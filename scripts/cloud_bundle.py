"""Package a frozen prepared-data selection and the learner for an NVIDIA cloud GPU.

The bundle is self-contained: data/ holds dataset.json and its shards (hard
links into training-data/ where the volume allows, else copies), code/ holds
every script (tests included, so the GPU path can be checked on the instance),
and setup.sh/train.sh run the same learner settings as run_prepared_training.py.
Upload the whole directory (e.g. rsync -a --partial bundle/ host:bundle/).
Gameplay checks stay local: the native engine is Windows-only.

  python -B scripts/cloud_bundle.py --output D:/bundles/v7-all --validation 1000
  python -B scripts/cloud_bundle.py --output ... --training 15000 --validation 1000
  python -B scripts/cloud_bundle.py --output ... --widths 50m     # 50.1M-parameter learner
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
TORCH = '2.14.0'
NUMPY = '2.4.5'
PYTHON = '3.14'

SETUP = f"""#!/usr/bin/env bash
# Python {PYTHON}, torch {TORCH} (CUDA) and numpy {NUMPY}: the versions the workstation learner uses.
# CUDA wheel index: cu130 for recent drivers; use cu128/cu126 if `nvidia-smi` shows an older CUDA.
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
# test_compiled_learner checks the --compile encoder against eager updates.
cd code && TEST_TORCH_DEVICE=cuda ../.venv/bin/python -m unittest \\
  test_event_policy test_event_training_batches test_scheduled_capture test_policy \\
  test_unit_tasks test_command_actions test_build_placement test_resume_construction \\
  test_event_dataset test_compiled_learner
"""

TRAIN = """#!/usr/bin/env bash
# The local experiment's learner settings (run_prepared_training.py), with cloud
# resource limits. Extra arguments are appended, e.g. --resume run/latest.npz.
# Resume needs identical settings: keep PRECISION, COMPILE and READERS unchanged.
# Defaults from the RTX 5070 Ti benchmark (scripts/cloud_training.md): BF16 with
# the compiled encoder was fastest; 4 readers supply ~20k rows/s on a modern CPU.
set -euo pipefail
cd "$(dirname "$0")"
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
READERS=${READERS:-$(( $(nproc) > 4 ? 4 : $(nproc) - 1 ))}
PRECISION=${PRECISION:-bf16}
COMPILE=${COMPILE:-1}
# Readers hand updates over in shared memory (~4 slots of >= 16 MiB each).
SHM_MB=$(df -m /dev/shm | awk 'NR==2 {print $4}')
if [ "$SHM_MB" -lt $(( READERS * 100 )) ]; then
  echo "/dev/shm has $SHM_MB MB free; $READERS readers need ~$(( READERS * 100 )) MB: raise the container's --shm-size" >&2
  exit 1
fi
EXTRA=()
if [ "$COMPILE" = 1 ]; then EXTRA+=(--compile); fi
exec .venv/bin/python -B code/train_events.py train --dataset data/dataset.json --output run \\
  --device cuda --epochs 20 --early-stopping-patience 3 --seed 123 --threads 1 \\
  --batch-size 256 --buffer-mib 64 --pair-budget 8000000 --checkpoint-seconds 600 \\
  --lr-decay 0.3 --min-lr 0.00001 --precision "$PRECISION" "${EXTRA[@]}" \\
  --reader-processes "$READERS" --prefetch-batches 2 \\
  --min-free-ram-mib 4096 --max-private-mib 1048576 --min-free-disk-gib 10 \\
  --max-gpu-mib $(( GPU_MIB * 9 / 10 )) --stop-file run/stop "$@"
"""

# Learner widths measured on the RTX 5070 Ti. Model-only: the feature key and
# shards are unchanged. '10m' is replay-runtime/policy.json's default.
WIDTHS = {'10m': {}, '50m': dict(hidden=3456, entity_hidden=512, map_hidden=256, scalar_hidden=512)}


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return 'linked'
    except OSError:
        shutil.copy2(source, destination)
        return 'copied'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training', type=int, help='Training replays; omit to take every remaining one')
    parser.add_argument('--validation', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--validation-from', help='Reuse the validation matches of an earlier dataset.json')
    parser.add_argument('--widths', default='10m',
                        help="Learner size: '10m', '50m' or a JSON object of model-only widths")
    args = parser.parse_args()
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
    doc['config'] = {**doc['config'], **widths}
    linked = {'linked': 0, 'copied': 0}
    for entry in doc['entries']:
        source = experiment.DATA/entry['path']
        linked[link_or_copy(source, output/'data'/entry['path'])] += 1
    experiment.save(output/'data/dataset.json', doc)
    dataset = EventDataset(output/'data/dataset.json')
    for ordinal, entry in enumerate(dataset.entries, 1):
        if experiment.digest(dataset.shard_path(entry)) != entry['archive_sha256']:
            raise ValueError('Bundled shard checksum mismatch: '+entry['path'])
        if ordinal % 1000 == 0:
            print(json.dumps({'verified': ordinal, 'total': len(dataset.entries)}), flush=True)
    for script in sorted((ROOT/'scripts').glob('*.py')):
        shutil.copy2(script, output/'code'/script.name)
    for name, text in (('setup.sh', SETUP), ('train.sh', TRAIN)):
        (output/name).write_text(text, encoding='utf-8', newline='\n')
    shards = [output/'data'/e['path'] for e in doc['entries']]
    summary = {'schema': 1, 'kind': 'generals_cloud_training_bundle', 'created': time.time(),
               'feature_key': key, 'dataset_sha256': experiment.digest(output/'data/dataset.json'),
               'training': sum(e['split'] == 'train' for e in doc['entries']),
               'validation': sum(e['split'] == 'validation' for e in doc['entries']),
               'shard_bytes': sum(p.stat().st_size for p in shards), **linked, 'widths': widths,
               'python': PYTHON, 'torch': TORCH, 'numpy': NUMPY, 'seconds': time.time()-started}
    experiment.save(output/'bundle.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
