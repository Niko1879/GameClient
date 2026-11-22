"""Package a frozen prepared-data selection and the learner for an NVIDIA cloud GPU.

The bundle holds no shards: data/ has only dataset.json, and bundle.json
names training-data/ as the shard source, from which scripts/cloud_upload.py
streams every shard into data/ on the instance. Nothing is copied or linked on
the workstation, so a full bundle takes no disk space. code/ holds every script
(tests included, so the GPU path can be checked on the instance); setup.sh runs
them and train.sh trains with SCC's supervised recipe (notes/plan.md §3).
Gameplay checks stay local: the native engine is Windows-only.

  python -B scripts/cloud_bundle.py --output D:/bundles/v7-all --validation 1000
  python -B scripts/cloud_bundle.py --output ... --training 15000 --validation 1000
  python -B scripts/cloud_bundle.py --output ... --widths '{"map_hidden": 64}'  # a model-only setting

bundle.json records the learner's parameter count, which changes as inputs and
heads are added; the preset names deliberately carry no size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
TORCH = '2.14.0'
NUMPY = '2.4.5'
PYTHON = '3.14'

SETUP = f"""#!/usr/bin/env bash
# Python {PYTHON}, torch {TORCH} (CUDA) and numpy {NUMPY}: the versions the workstation learner uses.
# CUDA wheel index: cu130 for recent drivers; use cu126 if `nvidia-smi` shows an older CUDA.
set -euo pipefail
cd "$(dirname "$0")"
INDEX=${{TORCH_INDEX:-https://download.pytorch.org/whl/cu130}}
if command -v uv >/dev/null; then
  uv venv --python {PYTHON} .venv
  uv pip install --python .venv "numpy=={NUMPY}" "torch=={TORCH}" --index-url "$INDEX" --extra-index-url https://pypi.org/simple
else
  python{PYTHON} -m venv .venv
  .venv/bin/pip install "numpy=={NUMPY}"
  .venv/bin/pip install "torch=={TORCH}" --index-url "$INDEX"
fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
# The learner's CPU reference, parity and continuation tests, on this GPU;
# test_compiled_learner checks the --compile encoder against eager updates;
# test_distributed_training checks NCCL ranks against one GPU on a multi-GPU host.
# Deterministic kernels for the tests only: the bitwise continuation/staging tests compare two runs, and
# CUDA's atomic scatter_add/index_add sums vary in the last bit between runs (12 Nov, 5090: 1e-10 in
# map_project_b). Training keeps the fast atomic kernels.
cd code && TEST_TORCH_DEVICE=cuda CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.venv/bin/python -c \\
  'import sys, torch, unittest; torch.use_deterministic_algorithms(True, warn_only=True); unittest.main(module=None, argv=sys.argv)' \\
  test_event_policy test_event_training_batches test_scheduled_capture test_policy \\
  test_unit_tasks test_command_actions test_build_placement test_resume_construction \\
  test_event_dataset test_compiled_learner test_enemy_and_clock test_prepared_training test_learner_bench \\
  test_distributed_training test_staged_updates test_alphagenerals
"""

TRAIN = """#!/usr/bin/env bash
# AlphaGenerals with SCC's supervised recipe (notes/plan.md Â§3), with cloud resource limits.
# Extra arguments are appended, e.g. --resume run/latest.npz, or for the winners' fine-tune
# --init <stage-2 best.npz> with LR=0.00001 LR_SCHEDULE=constant on its own bundle.
# Resume needs identical settings: keep PRECISION, COMPILE, READERS, WORLD, LANES, UNROLL, PART_LANES,
# CACHE_MIB and the learning-rate variables unchanged.
set -euo pipefail
cd "$(dirname "$0")"
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
WORLD=${WORLD:-1}  # One GPU (user decision, 9 Nov); WORLD=N for data-parallel ranks (lanes split between them).
# An update is LANES episodes x UNROLL consecutive decisions (truncated BPTT, LSTM state carried):
# 240 x 64 = 15,360, SCC's final batch (plan Â§3).
LANES=${LANES:-240}
UNROLL=${UNROLL:-64}
PART_LANES=${PART_LANES:-8}  # Reader preparation unit: 512 decisions, the old batch's shared-memory size.
CACHE_MIB=${CACHE_MIB:-256}  # Decoded chunks each reader keeps while its lanes read their episodes.
# Reader processes per rank: each rank's main thread needs a core of its own.
PER_RANK=$(( ($(nproc) - WORLD) / WORLD ))
# Inductor otherwise forks one ~0.5 GB compile worker per CPU thread (12 GB on a
# 24-thread host), which with the readers overran a 31 GB container (11 Nov).
export TORCHINDUCTOR_COMPILE_THREADS=1
# Each reader loads its own copy of the dataset index (~3.3 GB for 112.7k matches) plus its chunk
# cache, and the main process ~8 GB plus one update in host memory: cap readers by the memory limit.
MEM_LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
[ "$MEM_LIMIT" = max ] && MEM_LIMIT=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 ))
MEM_READERS=$(( (MEM_LIMIT / 1048576 - 16384) / (3584 + CACHE_MIB) / WORLD ))
CPU_READERS=$(( PER_RANK >= 8 ? 8 : (PER_RANK > 1 ? PER_RANK : 1) ))
READERS=${READERS:-$(( MEM_READERS < CPU_READERS ? (MEM_READERS > 1 ? MEM_READERS : 1) : CPU_READERS ))}
PRECISION=${PRECISION:-bf16}
COMPILE=${COMPILE:-1}  # The encoder; the core LSTM and the heads stay eager and FP32.
# Validation rounds per epoch; PATIENCE counts rounds. One round scores the 500 validation
# matches' whole episodes in order.
VALIDATIONS=${VALIDATIONS:-4}
PATIENCE=${PATIENCE:-4}  # One epoch without a 0.005 gain (user, 12 Nov: stop soon after validation flattens).
# SCC's recipe: 1e-3, then x0.1 after 154M decisions (its 40k steps at batch 3,850); warmup ours.
WARMUP=${WARMUP:-500}
LR=${LR:-0.001}
LR_SCHEDULE=${LR_SCHEDULE:-step}
LR_STEP_DECISIONS=${LR_STEP_DECISIONS:-154000000}
EPOCHS=${EPOCHS:-10}  # The cap on pool epochs (plan Â§3).
# Readers hand parts over in shared memory (~4 slots of >= 16 MiB each).
SHM_MB=$(df -m /dev/shm | awk 'NR==2 {print $4}')
if [ "$SHM_MB" -lt $(( READERS * WORLD * 400 )) ]; then
  echo "/dev/shm has $SHM_MB MB free; $(( READERS * WORLD )) readers need ~$(( READERS * WORLD * 400 )) MB: raise the container's --shm-size" >&2
  exit 1
fi
# Rating input ON (user, 15/17 Nov 2026: it helped the top-20% run). It is no config key (that would change
# the feature key), so the run directory carries the RATING_INPUT marker play_head_to_head checks.
export GENERALS_RATING_INPUT=${GENERALS_RATING_INPUT:-1}
mkdir -p run
if [ "$GENERALS_RATING_INPUT" = 1 ]; then touch run/RATING_INPUT; fi
EXTRA=()
if [ "$COMPILE" = 1 ]; then EXTRA+=(--compile); fi
if [ "$LR_SCHEDULE" = step ]; then EXTRA+=(--lr-step-decisions "$LR_STEP_DECISIONS"); fi
exec .venv/bin/python -B code/train_events.py train --dataset data/dataset.json --output run \\
  --device cuda --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --validations-per-epoch "$VALIDATIONS" \\
  --min-delta 0.005 --seed TRAIN_SEED --threads 1 \\
  --world-size "$WORLD" --lanes "$LANES" --unroll "$UNROLL" --part-lanes "$PART_LANES" --cache-mib "$CACHE_MIB" \\
  --pair-budget 8000000 --checkpoint-seconds 600 \\
  --lr "$LR" --lr-warmup-updates "$WARMUP" --lr-schedule "$LR_SCHEDULE" --precision "$PRECISION" "${EXTRA[@]}" \\
  --reader-processes "$READERS" --prefetch-batches 2 \\
  --min-free-ram-mib 4096 --max-private-mib 1048576 --min-free-disk-gib 10 \\
  --max-gpu-mib $(( GPU_MIB * 9 / 10 )) --stop-file run/stop "$@"
"""

# Model-only config overrides (event_store.MODEL_ONLY): the feature key and shards are unchanged.
# 'default' is AlphaGenerals' fixed widths (generals_policy.WIDTHS, 47.3M parameters; notes/plan.md §6),
# which cannot change without changing the feature key. Others are JSON, e.g. '{"map_hidden": 64}'.
WIDTHS = {'default': {}}


def check_shards(entries, source, report=None):
    """Bytes of the selected shards, each hashed in place: the upload sends exactly these."""
    from prepared_io import _digest
    total = 0
    for ordinal, entry in enumerate(entries, 1):
        path = Path(source)/entry['path']
        if not path.is_file() or _digest(path) != entry['archive_sha256']:
            raise ValueError('Shard missing or changed in '+str(source)+': '+entry['path'])
        total += path.stat().st_size
        if report and ordinal % 1000 == 0:
            report({'verified': ordinal, 'total': len(entries)})
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training', type=int, help='Training replays; omit to take every remaining one')
    parser.add_argument('--validation', type=int, default=1500,
                        help='Validation matches, stratified by length and rating (run_prepared_training.choose)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--validation-from', help='Reuse the validation matches of an earlier dataset.json')
    import run_prepared_training as selection_options
    selection_options.add_archive_arguments(parser)
    parser.add_argument('--train-seed', type=int, default=123, help='Learner seed baked into train.sh (initialisation, shuffling)')
    parser.add_argument('--widths', default='default',
                        help="Learner size: 'default' or a JSON object of model-only settings")
    args = parser.parse_args()
    from engine_runtime import disable_power_throttling
    disable_power_throttling()
    from event_store import MODEL_ONLY
    widths = WIDTHS[args.widths] if args.widths in WIDTHS else json.loads(args.widths)
    if not isinstance(widths, dict) or not set(widths) <= MODEL_ONLY:
        raise ValueError('Widths must be model-only settings: '+', '.join(sorted(MODEL_ONLY)))
    import run_prepared_training as experiment
    from event_dataset import EventDataset
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a new bundle directory')
    (output/'data').mkdir(parents=True); (output/'code').mkdir()
    started = time.time()
    doc, key = experiment.selection(args)
    # Strategy input z: a manifest without openers would train without it, silently.
    openers = doc['selection']['openers']
    if openers['with_opener'] < 0.8*openers['players']:
        raise ValueError(f'Only {openers["with_opener"]} of {openers["players"]} players have an opener')
    doc['config'] = {**doc['config'], **widths}
    from generals_policy import NeuralPolicy
    parameters = sum(value.data.size for value in NeuralPolicy(doc['config'], args.seed).parameters.values())
    experiment.save(output/'data/dataset.json', doc)
    EventDataset(output/'data/dataset.json')  # The manifest's own checks.
    shard_bytes = check_shards(doc['entries'], experiment.DATA, lambda item: print(json.dumps(item), flush=True))
    for script in sorted((ROOT/'scripts').glob('*.py')):
        shutil.copy2(script, output/'code'/script.name)
    # event_policy.SCIENCE_ORDERS resolves next to code/: the scripted science buyer every EventActor loads.
    (output/'replay-runtime').mkdir()
    shutil.copy2(ROOT/'replay-runtime/science-orders.json', output/'replay-runtime/science-orders.json')
    for name, text in (('setup.sh', SETUP), ('train.sh', TRAIN.replace('TRAIN_SEED', str(args.train_seed)))):
        (output/name).write_text(text, encoding='utf-8', newline='\n')
    summary = {'schema': 1, 'kind': 'generals_cloud_training_bundle', 'created': time.time(),
               'feature_key': key, 'dataset_sha256': experiment.digest(output/'data/dataset.json'),
               'training': sum(e['split'] == 'train' for e in doc['entries']),
               'validation': sum(e['split'] == 'validation' for e in doc['entries']),
               'enemy_unknown': sum(not e.get('faction_identities') for e in doc['entries']),
               'shard_source': str(experiment.DATA), 'shard_bytes': shard_bytes, 'widths': widths,
               'parameters': parameters, 'train_seed': args.train_seed, 'selection': doc.get('selection'),
               'python': PYTHON, 'torch': TORCH, 'numpy': NUMPY, 'seconds': time.time()-started}
    experiment.save(output/'bundle.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
