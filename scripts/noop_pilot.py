"""The no-op snapshot pilot's fixed rules (notes/wait_snapshots.md), over one checkpoint's held-out rows.

  python -B noop_pilot.py score --dataset data/dataset.json --checkpoint run/best.npz --output rows.npz --device cuda
  python -B noop_pilot.py report --rows rows.npz --output report.json

score: the model's P(no-op) at every validation row, with the carried memory, with the memory reset at every row,
and (memory on) with the APM input set to 40 and 80.
report: rules 1-5 as pass/fail with their numbers. Rule 6 compares two reports, one per training snapshot rate. The table of rules 1 and 3 is fitted on half of the matches
and scored on the other, both ways round.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

APMS, APM_RATIO = (40, 80), 1.5
SINCE = (.5, 2.)            # Seconds since the last command.
MINUTES = (3., 10.)
SITUATIONS = ('combat', 'enemy visible', 'quiet')
MEMORY_GAIN, GRID_DIFFERENCE, CALIBRATION = .02, .02, .15


def situation(encoded):
    """0: an own unit was just hit or is firing; 1: an enemy is in view; 2: neither."""
    from generals_policy import ENEMY_COLUMN, ENTITY_FEATURES, OWN_COLUMN, UNDER_ATTACK_COLUMN, VISIBLE_COLUMN
    numeric = np.asarray(encoded['numeric'])
    own, firing = numeric[:, OWN_COLUMN] > 0, ENTITY_FEATURES-3
    if (own & ((numeric[:, UNDER_ATTACK_COLUMN] > 0) | (numeric[:, firing] > 0))).any():
        return 0
    return 1 if ((numeric[:, ENEMY_COLUMN] > 0) & (numeric[:, VISIBLE_COLUMN] > 0)).any() else 2


def score(dataset, reference, device, rows_at_once=128):
    """Per-row arrays of every validation episode, in game order."""
    import torch
    from generals_policy import apm_inputs
    from torch_policy_v3 import TorchPolicyV3
    from torch_training_inputs import cap_logits
    model = TorchPolicyV3(reference, device=device).eval()
    model.legality_key = dataset.key
    noop = reference.action_index[(0, -1)]
    result = {name: [] for name in ('match', 'player', 'frame', 'rate', 'noop', 'since', 'situation', 'apm',
                                    'memory', 'reset', *(f'apm{apm}' for apm in APMS))}

    def core_inputs(encoded):
        return torch.cat([model.encode_batch(encoded[i:i+rows_at_once])['core_input']
                          for i in range(0, len(encoded), rows_at_once)])

    def probabilities(inputs, masks, reset):
        resets = torch.full((1, len(inputs)), reset, dtype=torch.bool, device=model.device)
        core, _ = model.core(inputs[None].to(model.dtype), resets, model.initial_state(1))
        logits = model.action_logits(core[0])
        return cap_logits(logits, masks).masked_fill(~masks, -torch.inf).log_softmax(1)[:, noop].cpu().numpy()

    with torch.inference_mode():
        for number, (_, count) in enumerate(dataset.episodes('validation')):
            rows = dataset.episode_rows('validation', number, 0, count)
            encoded = [row['encoded'] for row in rows]
            masks = torch.as_tensor(np.stack([reference.action_mask(e, e['_training_legality']['verb']) for e in encoded]),
                                    device=model.device)
            inputs = core_inputs(encoded)
            result['memory'].append(probabilities(inputs, masks, False))
            result['reset'].append(probabilities(inputs, masks, True))
            for apm in APMS:
                shown = apm_inputs(apm)
                result[f'apm{apm}'].append(probabilities(core_inputs([{**e, 'apm': shown} for e in encoded]), masks, False))
            frames = np.array([row['frame'] for row in rows])
            noops = np.array([row['target']['kind'] == 'wait' for row in rows])
            last, since = None, []
            for frame, is_noop in zip(frames, noops):
                since.append(np.inf if last is None else (frame-last)/60)
                if not is_noop:
                    last = frame
            result['match'].append([row['match_key'] for row in rows]); result['player'].append([row['player'] for row in rows])
            result['frame'].append(frames); result['noop'].append(noops); result['since'].append(since)
            result['rate'].append([row.get('source_ticks_per_second', 60) for row in rows])
            result['situation'].append([situation(e) for e in encoded])
            result['apm'].append([np.nan if row['apm'] is None else row['apm'] for row in rows])
    return {name: np.concatenate(values) for name, values in result.items()}


def nll(logp_noop, noop):
    """Per-row act/no-op loss from ln P(no-op)."""
    p = np.clip(np.exp(logp_noop), 1e-6, 1-1e-6)
    return -np.where(noop, np.log(p), np.log(1-p))


def cells(rows):
    """Each row's table cell: time since the last command, minute, situation."""
    edges = (.25, .5, 1., 2., 4., 8.)
    return (np.digitize(rows['since'], edges)*3+np.digitize(rows['frame']/3600, MINUTES))*3+rows['situation']


def folds(rows):
    """0 or 1 per row: whole matches, alternating in sorted order."""
    names, index = np.unique(rows['match'], return_inverse=True)
    return index % 2


def table_loss(rows):
    """Cross-fitted loss of the (time since last command, minute, situation) table."""
    fold, cell = folds(rows), cells(rows)
    logp = np.zeros(len(cell))
    for part in (0, 1):
        fit, test = fold != part, fold == part
        total = np.bincount(cell[fit], minlength=63)+2
        hits = np.bincount(cell[fit], weights=rows['noop'][fit], minlength=63)+2*rows['noop'][fit].mean()
        logp[test] = np.log(hits/total)[cell[test]]
    return nll(logp, rows['noop'])


def reset_plus_table_loss(rows):
    """Cross-fitted loss of the memory-reset model corrected by the table: a slope on its logit and an offset per cell."""
    fold, cell, y = folds(rows), cells(rows), rows['noop'].astype(float)
    p = np.clip(np.exp(rows['reset']), 1e-6, 1-1e-6)
    logit = np.log(p/(1-p))
    result = np.zeros(len(y))
    for part in (0, 1):
        fit, test = fold != part, fold == part
        design = lambda m: np.concatenate([logit[m][:, None], np.eye(63)[cell[m]]], axis=1)
        x, target, w = design(fit), y[fit], np.r_[1., np.zeros(63)]
        prior, ridge = np.r_[1., np.zeros(63)], np.full(64, 1.)
        for _ in range(30):  # Newton steps of the ridge-regularised logistic fit.
            q = 1/(1+np.exp(-(x @ w)))
            step = np.linalg.solve((x*(q*(1-q))[:, None]).T @ x+np.diag(ridge), x.T @ (q-target)+ridge*(w-prior))
            w = w-step
            if np.abs(step).max() < 1e-8:
                break
        result[test] = nll(-np.logaddexp(0, -(design(test) @ w)), rows['noop'][test])
    return result


def bootstrap(values, matches, draws=2000, seed=1):
    """Mean and 95% interval of per-row values, resampling whole matches."""
    _, index = np.unique(matches, return_inverse=True)
    sums, counts = np.bincount(index, weights=values), np.bincount(index)
    picks = np.random.default_rng(seed).integers(0, len(counts), (draws, len(counts)))
    means = sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
    return float(values.mean()), [float(v) for v in np.percentile(means, (2.5, 97.5))]


def calibration(rows, groups):
    """{group: predicted commands / actual} over the rows of each group with at least 200 commands."""
    act, result = 1-np.exp(rows['memory']), {}
    for name, member in groups.items():
        actual = int((~rows['noop'] & member).sum())
        if actual >= 200:
            result[name] = {'commands': actual, 'predicted_over_actual': float(act[member].sum()/actual)}
    return result


def within(results):
    return bool(results) and all(abs(item['predicted_over_actual']-1) <= CALIBRATION for item in results.values())


def report(rows):
    noop, command = rows['noop'], ~rows['noop']
    memory, reset = nll(rows['memory'], noop), nll(rows['reset'], noop)
    table, corrected = table_loss(rows), reset_plus_table_loss(rows)
    on_grid = rows['frame'] % 6 == 0
    p = np.exp(rows['memory'])
    difference = float(p[command & on_grid].mean()-p[command & ~on_grid].mean())
    gain, interval = bootstrap(table-memory, rows['match'])
    minute = rows['frame']/3600
    groups = {f'situation: {name}': rows['situation'] == i for i, name in enumerate(SITUATIONS)}
    groups.update({'minute: 0-3': minute < 3, 'minute: 3-10': (minute >= 3) & (minute < 10), 'minute: 10+': minute >= 10})
    groups.update({'since: <0.5': rows['since'] < .5, 'since: 0.5-2': (rows['since'] >= .5) & (rows['since'] < 2),
                   'since: >2': (rows['since'] >= 2) & np.isfinite(rows['since'])})
    calibrated = calibration(rows, groups)
    # Regression to the mean: moments sorted by the model's own pace must be calibrated at both ends.
    cuts = np.quantile(p, (.1, .9))
    extremes = calibration(rows, {'fastest tenth': p <= cuts[0], 'slowest tenth': p >= cuts[1]})
    known = np.isfinite(rows['apm'])
    edges = np.quantile(rows['apm'][known], (1/3, 2/3)) if known.any() else (0, 0)
    band = np.digitize(np.nan_to_num(rows['apm']), edges)
    terciles = calibration(rows, {f'apm tercile {i+1}': known & (band == i) for i in range(3)})
    # No-op rows are a constant-rate sample of time, so the odds there average to the command rate.
    rate = {apm: float(((1-np.exp(rows[f'apm{apm}']))/np.exp(rows[f'apm{apm}']))[noop].mean()) for apm in APMS}
    quiet = noop & (rows['situation'] == 2)
    result = {
        'rows': len(noop), 'matches': len(np.unique(rows['match'])), 'commands': int(command.sum()), 'noops': int(noop.sum()),
        'act_noop_nll': {'memory': float(memory.mean()), 'reset': float(reset.mean()), 'table': float(table.mean()),
                         'reset_plus_table': float(corrected.mean())},
        'rule1_memory_leak': {'unexplained_gain': float((corrected-memory).mean()),
                              'passed': bool((corrected-memory).mean() <= MEMORY_GAIN)},
        'rule2_grid_leak': {'p_noop_on_minus_off_grid': difference, 'passed': abs(difference) < GRID_DIFFERENCE},
        'rule3_signal': {'gain_over_table': gain, 'interval95': interval, 'passed': interval[0] > 0,
                         'p_noop_quiet_5_50_95': np.percentile(p[quiet], (5, 50, 95)).tolist() if quiet.any() else None,
                         'p_noop_combat_5_50_95': (np.percentile(p[rows['situation'] == 0], (5, 50, 95)).tolist()
                                                   if (rows['situation'] == 0).any() else None)},
        'rule4_calibration': {'groups': calibrated, 'passed': within(calibrated)},
        # Both ends must hold enough commands to judge: a missing one fails.
        'rule5_extremes': {'groups': extremes, 'passed': len(extremes) == 2 and within(extremes)},
        # Reported, not rules: whether fast players are told from slow ones, and whether the APM input is a working knob.
        'player_pace': terciles,
        'apm_input': {'rate_ratio_80_over_40': rate[80]/rate[40], 'works': rate[80]/rate[40] >= APM_RATIO}}
    result['passed'] = all(value['passed'] for name, value in result.items() if name.startswith('rule'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('score')
    run.add_argument('--dataset', required=True)
    run.add_argument('--checkpoint', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--device', default='cuda')
    make = commands.add_parser('report')
    make.add_argument('--rows', required=True)
    make.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'score':
        from event_dataset import EventDataset
        from generals_policy import NeuralPolicy
        dataset = EventDataset(args.dataset)
        reference, _, _ = NeuralPolicy.load(args.checkpoint)
        if reference.snapshot_rate != dataset.snapshot_rate:
            raise ValueError('The checkpoint was trained on another snapshot rate than this dataset keeps')
        np.savez_compressed(args.output, **score(dataset, reference, args.device))
        return
    with np.load(args.rows) as archive:
        result = report(dict(archive))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
