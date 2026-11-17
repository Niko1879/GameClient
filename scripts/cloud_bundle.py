"""Package a frozen prepared-data selection and the learner for an NVIDIA cloud GPU.

The bundle holds no shards: data/ has only dataset.json, and bundle.json
names training-data/ as the shard source, from which scripts/cloud_upload.py
streams every shard into data/ on the instance. Nothing is copied or linked on
the workstation, so a full bundle takes no disk space. code/ holds every script
(tests included, so the GPU path can be checked on the instance), and
setup.sh/train.sh run the same learner settings as run_prepared_training.py.
Gameplay checks stay local: the native engine is Windows-only.

  python -B scripts/cloud_bundle.py --output D:/bundles/v7-all --validation 1000
  python -B scripts/cloud_bundle.py --output ... --training 15000 --validation 1000
  python -B scripts/cloud_bundle.py --output ... --widths big     # v11 big run (52.1M; the map is not scaled)

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
  test_distributed_training test_staged_updates
"""

TRAIN = """#!/usr/bin/env bash
# The local experiment's learner settings (run_prepared_training.py), with cloud
# resource limits. Extra arguments are appended, e.g. --resume run/latest.npz.
# Resume needs identical settings: keep PRECISION, COMPILE, READERS, WORLD, BATCH, BUFFER_MIB and the
# learning-rate variables unchanged.
# Defaults from the RTX 5070 Ti benchmark (scripts/cloud_training.md): BF16 with
# the compiled encoder was fastest. 8 readers were the steadiest on a 5090 + 9950X (5 Nov).
set -euo pipefail
cd "$(dirname "$0")"
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# Data-parallel ranks, one per GPU (train_events --world-size); BATCH is the global
# batch, split across them. 512 for the v11 big run (15 Nov, with LR 4.2e-4 and a 1024 MiB
# window so batches really fill); the rating ladder used 256 with lr 3e-4 (BATCH=256 LR=0.0003).
WORLD=${WORLD:-1}  # One GPU (user decision, 9 Nov); WORLD=N for data-parallel ranks.
BATCH=${BATCH:-512}
# Reader processes per rank: each rank's main thread needs a core of its own.
PER_RANK=$(( ($(nproc) - WORLD) / WORLD ))
# Inductor otherwise forks one ~0.5 GB compile worker per CPU thread (12 GB on a
# 24-thread host), which with the readers overran a 31 GB container (11 Nov).
export TORCHINDUCTOR_COMPILE_THREADS=1
# Each reader loads its own copy of the dataset index (~3.3 GB for 112.7k matches)
# and the main process ~8 GB: cap readers by the container's memory limit too.
MEM_LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
[ "$MEM_LIMIT" = max ] && MEM_LIMIT=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 ))
MEM_READERS=$(( (MEM_LIMIT / 1048576 - 12288) / 3584 / WORLD ))
CPU_READERS=$(( PER_RANK >= 8 ? 8 : (PER_RANK > 1 ? PER_RANK : 1) ))
READERS=${READERS:-$(( MEM_READERS < CPU_READERS ? (MEM_READERS > 1 ? MEM_READERS : 1) : CPU_READERS ))}
PRECISION=${PRECISION:-bf16}
COMPILE=${COMPILE:-1}  # v9, 5090 + 9950X (5 Nov): +16% over eager; the first ~250 updates recompile
# Validation rounds per epoch; PATIENCE and LR_DECAY count rounds, so scale them with
# VALIDATIONS. All-data defaults (9 Nov): a 100k-match epoch is ~4.5 h on a 5090, so
# validate about hourly and stop after ~2 epochs without improvement.
VALIDATIONS=${VALIDATIONS:-4}
PATIENCE=${PATIENCE:-4}  # One epoch without a 0.005 gain (user, 12 Nov: stop soon after validation flattens).
# Learning rate (train_events.LearningRate, 12 Nov): a warmup ramp, then the rate halves
# only after LR_PATIENCE validation rounds in a row without improvement, and the round
# after a cut is not counted. v10's per-round 0.74 cut the rate on one loss spike in epoch 1.
# For a run of planned length: LR_SCHEDULE=cosine with EPOCHS set to that length
# (and LR_DECAY=1 to leave the cosine alone).
WARMUP=${WARMUP:-2000}
# Learning rate. v11 big run (15 Nov 2026, agent's call): batch 512 with 3e-4 x sqrt(2), the
# square-root rule for Adam; 256 / 3e-4 is the ladder runs' setting (top20R, the pilot).
LR=${LR:-0.00042}
LR_DECAY=${LR_DECAY:-0.5}
LR_PATIENCE=${LR_PATIENCE:-2}
LR_COOLDOWN=${LR_COOLDOWN:-1}
LR_SCHEDULE=${LR_SCHEDULE:-constant}
EPOCHS=${EPOCHS:-20}
# Shuffle window. A batch is cut from one window, and 64 MiB holds only ~290 rows of v6
# shards: BATCH=512 was never reached (v10 averaged 292 rows per update) and BATCH=256
# leaves a ~30-row update after every full one. 512 MiB is ~2,300 rows from ~140 game
# segments (measured on v6 shards, 12 Nov); it costs each reader well under 1 GB.
BUFFER_MIB=${BUFFER_MIB:-1024}
# Readers hand updates over in shared memory (~4 slots of >= 16 MiB each).
SHM_MB=$(df -m /dev/shm | awk 'NR==2 {print $4}')
if [ "$SHM_MB" -lt $(( READERS * WORLD * 100 )) ]; then
  echo "/dev/shm has $SHM_MB MB free; $(( READERS * WORLD )) readers need ~$(( READERS * WORLD * 100 )) MB: raise the container's --shm-size" >&2
  exit 1
fi
# Rating input ON (user, 15/17 Nov 2026: it helped the top-20% run). It is no config key (that would change
# the feature key), so the run directory carries the RATING_INPUT marker play_head_to_head checks.
export GENERALS_RATING_INPUT=${GENERALS_RATING_INPUT:-1}
mkdir -p run
if [ "$GENERALS_RATING_INPUT" = 1 ]; then touch run/RATING_INPUT; fi
EXTRA=()
if [ "$COMPILE" = 1 ]; then EXTRA+=(--compile); fi
exec .venv/bin/python -B code/train_events.py train --dataset data/dataset.json --output run \\
  --device cuda --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --validations-per-epoch "$VALIDATIONS" \\
  --min-delta 0.005 --seed TRAIN_SEED --threads 1 \\
  --world-size "$WORLD" --batch-size "$BATCH" --buffer-mib "$BUFFER_MIB" --pair-budget 8000000 --checkpoint-seconds 600 \\
  --lr-decay "$LR_DECAY" --lr-patience "$LR_PATIENCE" --lr-cooldown "$LR_COOLDOWN" \\
  --lr "$LR" --lr-warmup-updates "$WARMUP" --lr-schedule "$LR_SCHEDULE" --min-lr 0.00001 --precision "$PRECISION" "${EXTRA[@]}" \\
  --reader-processes "$READERS" --prefetch-batches 2 \\
  --min-free-ram-mib 4096 --max-private-mib 1048576 --min-free-disk-gib 10 \\
  --max-gpu-mib $(( GPU_MIB * 9 / 10 )) --stop-file run/stop "$@"
"""

# Model-only widths: the feature key and shards are unchanged. 'default' is
# replay-runtime/policy.json's own (29.0M parameters on 15 Nov, observation v11). 'big' is the
# v11 big run (user, 15 Nov 2026: do not scale the map; the capacity goes to the entity encoder,
# the core and the scalar encoder): 52.1M, entity+attention 12.1M (4x), core 15.1M (3x), map
# encoder and its 256-wide summary (map_fc 8.4M) unchanged. map_hidden is deliberately absent.
WIDTHS = {'default': {},
          'big': dict(hidden=2560, entity_hidden=512, scalar_hidden=512)}


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
                        help="Learner size: 'default', 'big' or a JSON object of model-only widths")
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
