"""Estimate a player's ladder rating from the replay command stream.

About 20% of players in the clean pool are unrated (custom lobbies, GenTool
retail games) and share one "unrated" policy input. This is a separate, small
model trained on rated Generals Online games that predicts the pre-match
Overall rating from how a player plays: order rates over time, the command
mix, hotkey use, burstiness, build/production timings, the outcome, and the
same features of the opponent.

Inputs are raw .rep files only (no engine, no shards, no catalogue reads), so
it covers every faction and both players of each game.

    features   one pass over the replays listed in cleaning-metrics.jsonl
    train      fit a ridge baseline, an MLP and LightGBM, report held-out
               accuracy, write an estimate (MLP/LightGBM mean) for every player

Labels: pre-match Overall from the Matches listing. Archive ratings
(gamereplays experts, GenTool) are synthetic and never used as labels; those
games only receive estimates. Splits are by player name, so a player's style
seen in training never inflates the held-out score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import re
from pathlib import Path
import struct
import sys

import numpy as np

from audit_replay_commands import message_names
from engine_runtime import disable_power_throttling
from harvest_china_replays import parse_replay
import map_slots
import replay_cleaning as rc

ROOT = rc.ROOT
OUT = ROOT/'replays/catalog'
FEATURES = OUT/'rating-features.npz'
MODEL = OUT/'rating-estimator.pt'
ESTIMATES = OUT/'rating-estimates.jsonl'
REPORT = OUT/'rating-estimator-report.json'
CENTER, SCALE = 1100.0, 300.0  # same normalisation as the policy's rating input

# The scan's faction labels (listing names; Random resolved). Before 13 Nov 2026 the general names here were
# spelled 'China Tank General' etc., which never matched, so every general's one-hot was all zero.
FACTIONS = ('China', 'China Tank', 'China Nuke', 'China Infantry',
            'USA', 'USA Super Weapon', 'USA Laser', 'USA Airforce',
            'GLA', 'GLA Toxin', 'GLA Demo', 'GLA Stealth')
# Command groups counted per minute (selection and camera messages handled apart).
GROUPS = {
    'move': ('MSG_DO_MOVETO',), 'attack': ('MSG_DO_ATTACK_OBJECT',),
    'attack_move': ('MSG_DO_ATTACKMOVETO',), 'force': ('MSG_DO_FORCE_ATTACK_GROUND', 'MSG_DO_FORCE_ATTACK_OBJECT'),
    'guard': ('MSG_DO_GUARD_POSITION', 'MSG_DO_GUARD_OBJECT'), 'stop': ('MSG_DO_STOP',),
    'scatter': ('MSG_DO_SCATTER',), 'waypoint': ('MSG_ADD_WAYPOINT',),
    'queue_unit': ('MSG_QUEUE_UNIT_CREATE',), 'cancel_unit': ('MSG_CANCEL_UNIT_CREATE',),
    'construct': ('MSG_DOZER_CONSTRUCT', 'MSG_DOZER_CONSTRUCT_LINE'),
    'cancel_construct': ('MSG_DOZER_CANCEL_CONSTRUCT',), 'resume': ('MSG_RESUME_CONSTRUCTION',),
    'upgrade': ('MSG_QUEUE_UPGRADE',), 'science': ('MSG_PURCHASE_SCIENCE',),
    'rally': ('MSG_SET_RALLY_POINT',), 'enter': ('MSG_ENTER',), 'exit': ('MSG_EXIT', 'MSG_EVACUATE'),
    'special': ('MSG_DO_SPECIAL_POWER', 'MSG_DO_SPECIAL_POWER_AT_OBJECT', 'MSG_DO_SPECIAL_POWER_AT_LOCATION',
                'MSG_DO_SPECIAL_POWER_OVERRIDE_DESTINATION'),
    'weapon': ('MSG_DO_WEAPON', 'MSG_DO_WEAPON_AT_LOCATION', 'MSG_DO_WEAPON_AT_OBJECT'),
    'sell': ('MSG_SELL',), 'repair': ('MSG_DO_REPAIR', 'MSG_GET_REPAIRED'), 'dock': ('MSG_DOCK',),
    'hack': ('MSG_INTERNET_HACK',), 'mode': ('MSG_ENABLE_RETALIATION_MODE', 'MSG_SET_MINE_CLEARING_DETAIL',
                                             'MSG_TOGGLE_OVERCHARGE'),
}
GROUP_OF = {m: g for g, names in GROUPS.items() for m in names}
WINDOWS = ((0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 1e9))  # minutes
MILESTONES = (('construct', 1), ('construct', 2), ('construct', 3), ('construct', 5), ('queue_unit', 2),
              ('queue_unit', 5), ('queue_unit', 10), ('upgrade', 1), ('science', 1), ('attack', 1),
              ('attack', 10), ('special', 1))
MISSING_SECONDS = 1800.0
OUTCOME = ('surrendered', 'last_to_order')  # who won, as far as the stream shows


def player_names():
    """Feature names of one player block (own and opponent blocks share them)."""
    names = ['log_orders_pm', 'log_selections_pm', 'log_hotkey_select_pm', 'log_hotkey_create_pm',
             'hotkey_teams', 'distinct_kinds', 'log_distinct_objects', 'mean_selection_size',
             'gap_p10', 'gap_p50', 'gap_p90', 'gap_under_0.2s', 'gap_under_0.5s', 'first_order_s',
             'active_fraction', 'surrendered', 'last_to_order']
    names += [f'log_{g}_pm' for g in GROUPS]
    names += [f'share_{g}' for g in GROUPS]
    names += [f'log_apm_{a}-{b}m' for a, b in WINDOWS]
    names += [f'log_t_{g}{n}' for g, n in MILESTONES] + [f'never_{g}{n}' for g, n in MILESTONES]
    names += [f'faction={f}' for f in FACTIONS]
    return names


GAME_NAMES = ['log_minutes', 'rate_30hz', 'retail']
BLOCK = len(player_names())


class Stream:
    def __init__(self):
        self.order_frames, self.kinds, self.groups, self.firsts = [], Counter(), Counter(), {}
        self.selections = self.hotkey_select = self.hotkey_create = 0
        self.teams, self.objects, self.selection_sizes = set(), set(), []

    def handle(self, frame, name, args):
        if name in ('MSG_CREATE_SELECTED_GROUP', 'MSG_CREATE_SELECTED_GROUP_NO_SOUND'):
            self.selections += 1
            self.objects.update(args[1:])
            self.selection_sizes.append(len(args)-1)
            return
        if name.startswith(('MSG_DESTROY_SELECTED_GROUP', 'MSG_REMOVE_FROM_SELECTED_GROUP')):
            return
        if name.startswith(('MSG_SELECT_TEAM', 'MSG_ADD_TEAM')):
            self.hotkey_select += 1
            self.teams.add(name[-1])
            return
        if name.startswith('MSG_CREATE_TEAM'):
            self.hotkey_create += 1
            self.teams.add(name[-1])
            return
        if name in rc.NOT_ORDERS or name == 'MSG_AREA_SELECTION_DEPRECATED':
            return
        self.order_frames.append(frame)
        self.kinds[name] += 1
        group = GROUP_OF.get(name)
        if group:
            self.groups[group] += 1
            self.firsts.setdefault(group, []).append(frame)


def block(stream, faction, rate, end, surrendered, last_to_order):
    minutes = max(end/rate/60, 0.5)
    orders = len(stream.order_frames)
    pm = lambda n: math.log1p(n/minutes)
    gaps = np.diff(np.asarray(stream.order_frames, dtype=np.float64))/rate if orders > 1 else np.zeros(1)
    first = stream.order_frames[0]/rate if orders else MISSING_SECONDS
    active = (stream.order_frames[-1]-stream.order_frames[0])/rate/60/minutes if orders > 1 else 0.0
    values = [pm(orders), pm(stream.selections), pm(stream.hotkey_select), pm(stream.hotkey_create),
              len(stream.teams), len(stream.kinds), math.log1p(len(stream.objects)),
              float(np.mean(stream.selection_sizes)) if stream.selection_sizes else 0.0,
              *(math.log1p(float(np.percentile(gaps, q))) for q in (10, 50, 90)),
              float(np.mean(gaps < 0.2)), float(np.mean(gaps < 0.5)), math.log1p(first),
              active, float(surrendered), float(last_to_order)]
    values += [pm(stream.groups[g]) for g in GROUPS]
    values += [stream.groups[g]/max(orders, 1) for g in GROUPS]
    frames = np.asarray(stream.order_frames, dtype=np.float64)/rate/60
    for a, b in WINDOWS:
        length = min(b, minutes)-a
        values.append(math.log1p(float(np.sum((frames >= a) & (frames < b)))/length) if length > 0 else 0.0)
    times, never = [], []
    for group, n in MILESTONES:
        hits = stream.firsts.get(group, ())
        seen = len(hits) >= n
        times.append(math.log1p(hits[n-1]/rate if seen else MISSING_SECONDS))
        never.append(0.0 if seen else 1.0)
    values += times+never
    values += [float(faction == f) for f in FACTIONS]
    return values


def extract(path, names):
    data = Path(path).read_bytes()
    header = parse_replay(data)
    streams = {}
    last_frame, last_player = 0, None
    for frame, name, index, args in rc.messages(data, header['commands_offset'], names):
        last_frame = max(last_frame, frame)
        streams.setdefault(index, Stream()).handle(frame, name, args)
        if name not in rc.NOT_ORDERS and not name.startswith(('MSG_CREATE_SELECTED', 'MSG_DESTROY_SELECTED')):
            last_player = index
    return header, streams, max(header['frames'], last_frame), last_player


GENTOOL_ID = re.compile(r'_([0-9A-Fa-f]{12})__')


def archive_aliases():
    """gamereplays name -> canonical name (player-names.csv alias_of, reviewed by the user)."""
    import csv
    with rc.ARCHIVE_ALIASES.open(encoding='utf-8-sig', newline='') as stream:
        return {r['name']: r['alias_of'] for r in csv.DictReader(stream) if r['alias_of']}


def identity(row, header, slot, aliases):
    """Who a player is across games, as reliably as each source allows.

    Generals Online names are accounts. A GenTool file is recorded by the player in
    its folder, whose 12-hex GenTool ID is stable; the opponent only has a name.
    gamereplays players go through the reviewed alias table.
    """
    source = row.get('archive')
    if source == 'gentool':
        found = GENTOOL_ID.search(Path(row['path']).name)
        if found and slot['slot'] == header['local_player']:
            return f'gentool:{found.group(1).upper()}'
        return f'gentool-name:{slot["name"]}'
    if source == 'gamereplays':
        return f'gamereplays:{aliases.get(slot["name"], slot["name"])}'
    return f'go:{slot["name"]}'


def featurize(row, names, opts):
    """One feature row per player of a clean-able 1v1 metrics row, or []."""
    if 'error' in row or len(row['players']) != 2:
        return []
    try:
        header, streams, end, last_player = extract(ROOT/row['path'], names)
    except (OSError, ValueError, struct.error):
        return []
    if len(header['players']) != 2:
        return []
    rate = row['rate']
    blocks, meta = [], []
    for slot, player in zip(header['players'], row['players']):
        stream = streams.get(player['player_index'], Stream())
        blocks.append(block(stream, player['faction'], rate, end, player['self_destruct'] is not None,
                            last_player == player['player_index']))
        overall = (player.get('rating') or {}).get('overall') if not row.get('archive') else None
        meta.append({'match_key': row['match_key'], 'player_index': player['player_index'],
                     'identity': identity(row, header, slot, opts['aliases']),
                     'name': slot['name'], 'faction': player['faction'], 'archive': row.get('archive'),
                     'collection': Path(row['path']).parts[1] if len(Path(row['path']).parts) > 2 else None,
                     'rating': overall, 'archive_rating': (player.get('rating') or {}).get('overall')
                     if row.get('archive') else None})
    game = [math.log(max(end/rate/60, 0.5)), float(rate == 30), float(bool(row.get('archive')))]
    clean = not rc.game_flags(row, opts['opts'], opts['allowed'], opts['slots'])
    out = []
    for i in (0, 1):
        out.append((blocks[i]+blocks[1-i]+game, {**meta[i], 'clean': clean}))
    return out


_STATE = {}


def _init():
    disable_power_throttling()
    _STATE['names'] = message_names()


def _batch(args):
    lines, allowed = args
    if 'opts' not in _STATE:
        _STATE['opts'] = {'opts': argparse.Namespace(**rc.DEFAULTS), 'allowed': allowed, 'slots': map_slots.load(),
                          'aliases': archive_aliases()}
    out = []
    for line in lines:
        out.extend(featurize(json.loads(line), _STATE['names'], _STATE['opts']))
    return out


def features(args):
    lines = Path(args.metrics).read_text(encoding='utf-8').splitlines()
    if args.limit:
        lines = lines[:args.limit]
    allowed = rc.ranked_map_crcs() | rc.ladder_map_crcs(args.metrics)
    batches = [(lines[i:i+200], allowed) for i in range(0, len(lines), 200)]
    X, meta = [], []
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for n, result in enumerate(pool.map(_batch, batches), 1):
            for x, m in result:
                X.append(x)
                meta.append(m)
            if n % 100 == 0:
                print(f'{n}/{len(batches)} batches, {len(X)} players', flush=True)
    X = np.asarray(X, dtype=np.float32)
    names = [f'own.{n}' for n in player_names()]+[f'opp.{n}' for n in player_names()]+GAME_NAMES
    np.savez_compressed(args.out, X=X, names=np.asarray(names), meta=np.asarray([json.dumps(m) for m in meta]))
    print(f'{len(X)} players, {X.shape[1]} features -> {args.out}')


# --- training -------------------------------------------------------------

def split_of(name):
    """0 train, 1 validation, 2 test; by player name (80/10/10)."""
    bucket = int(hashlib.sha1(name.encode('utf-8')).hexdigest()[:8], 16) % 10
    return 2 if bucket == 0 else 1 if bucket == 1 else 0


def metrics_of(pred, true):
    err = pred-true
    return {'n': int(len(true)), 'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err**2))),
            'pearson': float(np.corrcoef(pred, true)[0, 1]) if len(true) > 2 else None,
            'bias': float(np.mean(err))}


def bands(pred, true):
    out = {}
    for lo, hi in ((0, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 1700), (1700, 9999)):
        m = (true >= lo) & (true < hi)
        if m.sum():
            out[f'{lo}-{hi}'] = {'n': int(m.sum()), 'mean_estimate': float(pred[m].mean()),
                                 'mae': float(np.abs(pred[m]-true[m]).mean())}
    return out


def ridge(X, y, l2=1.0):
    A = np.c_[X, np.ones(len(X))]
    w = np.linalg.solve(A.T@A+l2*np.eye(A.shape[1]), A.T@y)
    return lambda Z: np.c_[Z, np.ones(len(Z))]@w


QUANTILES = np.linspace(0, 100, 201)


def calibrate(raw, quantile_map):
    """Monotone quantile matching (user, 15 Nov 2026): the ensemble's ordering is kept, its distribution is
    stretched to the labelled ratings' on the validation split, which undoes most of the regression to the mean
    in the tails. No clip: the strongest players stay apart (the old 0.5/99.5 clip tied the archive elite)."""
    return np.interp(raw, quantile_map[0], quantile_map[1])


def band_weights(rating, width=100.0, cap=10.0):
    """Inverse-frequency weights per rating band (user, 14 Nov 2026): every band counts about equally,
    so the bulk near the median no longer pulls the strong and weak players toward it. Capped, mean 1."""
    band = np.floor(rating/width)
    values, counts = np.unique(band, return_counts=True)
    weight = np.minimum(len(rating)/(len(values)*counts.astype(np.float64)), cap)
    w = weight[np.searchsorted(values, band)]
    return (w/w.mean()).astype(np.float32)


def mlp(Xtr, ytr, Xva, yva, opts, wtr=None):
    import torch
    torch.manual_seed(opts.seed)
    torch.set_num_threads(opts.threads)
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, opts.width), torch.nn.GELU(), torch.nn.Dropout(opts.dropout),
                              torch.nn.Linear(opts.width, opts.width), torch.nn.GELU(), torch.nn.Dropout(opts.dropout),
                              torch.nn.Linear(opts.width, 1))
    optim = torch.optim.AdamW(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    wt = torch.from_numpy(wtr) if wtr is not None else None
    xv = torch.from_numpy(Xva)
    best, best_state, stale = 1e9, None, 0
    steps = math.ceil(len(xt)/opts.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, opts.lr, total_steps=opts.epochs*steps)
    for epoch in range(opts.epochs):
        net.train()
        order = torch.randperm(len(xt))
        for i in range(steps):
            idx = order[i*opts.batch:(i+1)*opts.batch]
            losses = torch.nn.functional.huber_loss(net(xt[idx]).squeeze(1), yt[idx], delta=0.5, reduction='none')
            loss = (losses*wt[idx]).mean() if wt is not None else losses.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            mae = float(np.mean(np.abs(net(xv).squeeze(1).numpy()-yva)))*SCALE
        print(f'epoch {epoch+1}: validation MAE {mae:.1f}', flush=True)
        if mae < best-0.1:
            best, stale = mae, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= opts.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def gbm(Xtr, ytr, Xva, yva, names, opts, wtr=None, wva=None):
    import lightgbm as lgb
    params = dict(objective='huber', alpha=0.5, learning_rate=0.05, num_leaves=opts.leaves, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, metric='l1',
                  num_threads=opts.threads, seed=opts.seed, verbose=-1)
    train_set = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=[n.replace(' ', '_') for n in names])
    return lgb.train(params, train_set, num_boost_round=5000, valid_sets=[lgb.Dataset(Xva, yva, weight=wva, reference=train_set)],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])


def train(args):
    import torch
    data = np.load(args.features)
    X, names = data['X'].astype(np.float32), list(data['names'])
    meta = [json.loads(m) for m in data['meta']]
    if args.own_only:
        keep = [i for i, n in enumerate(names) if not n.startswith('opp.')]
        X, names = X[:, keep], [names[i] for i in keep]
    if not args.with_outcome:  # style only (default): `joint` adds results itself
        keep = [i for i, n in enumerate(names) if n.split('.')[-1] not in OUTCOME]
        X, names = X[:, keep], [names[i] for i in keep]
    rating = np.asarray([m['rating'] if m['rating'] is not None else np.nan for m in meta], dtype=np.float64)
    clean = np.asarray([m['clean'] for m in meta])
    split = np.asarray([split_of(m['name']) for m in meta])
    labelled = clean & ~np.isnan(rating)
    # The Tank/Nuke crawl (replays/china-generals) downloads highest-rated first, so its games are a rating-skewed
    # sample: as labels they would teach 'China Tank/Nuke player => strong'. Estimated, never labels (user, 13 Nov
    # 2026), until that crawl is complete. Its players are rated by their listing anyway.
    if not args.crawl_labels:
        labelled &= np.asarray([m.get('collection') != 'china-generals' for m in meta])
    if args.exclude_1000:
        labelled &= rating != 1000
    # Columns constant among rated training players (retail flags, 30 Hz) carry no
    # signal and would blow up the MLP on retail games after standardising.
    varying = X[labelled & (split == 0)].std(0) > 1e-3
    dropped = [n for n, v in zip(names, varying) if not v]
    X, names = X[:, varying], [n for n, v in zip(names, varying) if v]
    print(f'dropped constant columns: {dropped}')
    mean, std = X[labelled & (split == 0)].mean(0), X[labelled & (split == 0)].std(0)
    Z = ((X-mean)/std).astype(np.float32)
    y = ((rating-CENTER)/SCALE).astype(np.float32)
    tr, va, te = (labelled & (split == s) for s in (0, 1, 2))
    print(f'labelled clean players: train {tr.sum()}, validation {va.sum()}, test {te.sum()}')

    report = {'features': len(names), 'own_only': args.own_only, 'with_outcome': args.with_outcome, 'exclude_1000': args.exclude_1000}
    median = float(np.median(rating[tr]))
    report['baseline_median'] = metrics_of(np.full(te.sum(), median), rating[te])
    lin = ridge(Z[tr], y[tr], l2=args.l2)
    report['ridge'] = metrics_of(lin(Z[te])*SCALE+CENTER, rating[te])
    weights = band_weights(rating[labelled]) if args.band_weights else None
    if weights is not None:
        full = np.ones(len(rating), dtype=np.float32); full[labelled] = weights
        report['band_weights'] = True
    net = mlp(Z[tr], y[tr], Z[va], y[va], args, full[tr] if weights is not None else None)
    with torch.no_grad():
        mlp_pred = net(torch.from_numpy(Z)).squeeze(1).numpy().astype(np.float64)*SCALE+CENTER
    report['mlp'] = metrics_of(mlp_pred[te], rating[te])
    booster = gbm(X[tr], y[tr], X[va], y[va], names, args, *((full[tr], full[va]) if weights is not None else ()))
    gbm_pred = booster.predict(X, num_iteration=booster.best_iteration)*SCALE+CENTER
    report['gbm'] = metrics_of(gbm_pred[te], rating[te])
    report['gbm_iterations'] = int(booster.best_iteration)
    gain = booster.feature_importance('gain')
    report['gbm_top_features'] = [[names[i], round(float(gain[i]/gain.sum()), 4)] for i in np.argsort(-gain)[:30]]
    raw = (mlp_pred+gbm_pred)/2  # style estimate: mean of the two models
    # The target distribution is the unbiased labels' only: the Tank/Nuke crawl was collected highest-rated first.
    unbiased = va & np.asarray([m.get('collection') != 'china-generals' for m in meta])
    quantile_map = np.stack([np.percentile(raw[unbiased], QUANTILES), np.percentile(rating[unbiased], QUANTILES)])
    pred = calibrate(raw, quantile_map)
    report['ensemble_raw'] = metrics_of(raw[te], rating[te])
    report['ensemble'] = metrics_of(pred[te], rating[te])
    report['ensemble_bands'] = bands(pred[te], rating[te])
    china = np.asarray([m['faction'].startswith('China') for m in meta])
    report['ensemble_china'] = metrics_of(pred[te & china], rating[te & china])
    # Rating stability ceiling: the same name's rating in its nearest other rated game.
    report['same_name_other_game'] = name_lookup(meta, rating, labelled, te)
    known = KnownRatings(meta, rating)
    unrated = clean & np.isnan(rating)
    has_name = np.asarray([m['name'] in known for m in meta])
    report['unrated_with_rated_name'] = {str(s): [int((unrated & has_name & src).sum()), int((unrated & src).sum())]
                                         for s, src in ((s, np.asarray([m['archive'] == s for m in meta]))
                                                        for s in (None, 'gentool', 'gamereplays'))}
    experts = rc.archive_experts()
    gr = unrated & np.asarray([m['archive'] == 'gamereplays' for m in meta])
    is_expert = np.asarray([m['name'] in experts for m in meta])
    report['gamereplays_style_estimate_median'] = {
        'experts': float(np.median(pred[gr & is_expert])) if (gr & is_expert).any() else None,
        'others': float(np.median(pred[gr & ~is_expert])) if (gr & ~is_expert).any() else None}
    report['unrated_clean'] = {'n': int(unrated.sum()),
                               'estimate_percentiles': {str(q): float(np.percentile(pred[unrated], q))
                                                        for q in (10, 25, 50, 75, 90)} if unrated.any() else None,
                               'by_source': {str(s): int((unrated & np.asarray([m['archive'] == s for m in meta])).sum())
                                             for s in (None, 'gentool', 'gamereplays')}}
    report['rated_test_estimate_percentiles'] = {str(q): float(np.percentile(pred[te], q)) for q in (10, 25, 50, 75, 90)}
    print(json.dumps({k: report[k] for k in ('baseline_median', 'ridge', 'mlp', 'gbm', 'ensemble', 'ensemble_china',
                                             'same_name_other_game')}, indent=1))
    print(json.dumps(report['ensemble_bands'], indent=1))
    print(json.dumps(report['gbm_top_features'][:15]))
    print(json.dumps(report['unrated_clean'], indent=1))
    print('unrated with a rated name [with, all]:', report['unrated_with_rated_name'])
    print('gamereplays style estimate median:', report['gamereplays_style_estimate_median'])
    Path(args.report).write_text(json.dumps(report, indent=1), encoding='utf-8')
    torch.save({'state': net.state_dict(), 'mean': mean, 'std': std, 'names': names,
                'width': args.width, 'center': CENTER, 'scale': SCALE, 'quantile_map': quantile_map,
                'gbm': booster.model_to_string(num_iteration=booster.best_iteration)}, args.model)
    with Path(args.estimates).open('w', encoding='utf-8') as stream:
        for m, p, s, r in zip(meta, pred, split, raw):
            # Rule (user, 11 Nov): an unrated GO player resolved to their GO rating when the name has one;
            # gamereplays and GenTool players always take the model's rating (names there may be anyone).
            by_name = known.get(m['name'], m['match_key']) if m['archive'] is None else None
            estimate = m['rating'] if m['rating'] is not None else by_name if by_name is not None else round(float(p))
            stream.write(json.dumps({'match_key': m['match_key'], 'player_index': m['player_index'],
                                     'name': m['name'], 'identity': m['identity'], 'faction': m['faction'], 'archive': m['archive'],
                                     'clean': m['clean'], 'rating': m['rating'], 'style_estimate': round(float(p)),
                                     'raw_estimate': round(float(r), 1), 'collection': m.get('collection'),
                                     'name_rating': by_name, 'estimate': estimate,
                                     'split': ['train', 'validation', 'test'][s]}, ensure_ascii=False)+'\n')
    print(f'wrote {args.report}, {args.model}, {args.estimates}')


def match_number(key):
    """Generals Online match IDs grow with time; None for other keys."""
    return int(key[4:]) if key.startswith('pgo:') else None


class KnownRatings:
    """A Generals Online name's pre-match Overall in its rated game NEAREST IN TIME (by match ID).

    Ratings drift: leave-one-out over 240k rated games, the nearest game is off by
    24.6 on average, the median over all the name's games by 49.4. Any rated GO game
    counts (clean or not); only the rating is read from it.
    """
    def __init__(self, meta, rating):
        import bisect
        self._bisect = bisect.bisect_left
        games = {}
        for i, m in enumerate(meta):
            if m['archive'] is None and not np.isnan(rating[i]):
                games.setdefault(m['name'], []).append((match_number(m['match_key']), float(rating[i])))
        self.games = {}
        for name, items in games.items():
            numbered = sorted(x for x in items if x[0] is not None)
            self.games[name] = ([n for n, _ in numbered], [r for _, r in numbered],
                                round(float(np.median([r for _, r in items]))))

    def __contains__(self, name):
        return name in self.games

    def get(self, name, key):
        if name not in self.games:
            return None
        numbers, ratings, median = self.games[name]
        here = match_number(key)
        if here is None or not numbers:
            return median
        j = self._bisect(numbers, here)
        nearest = [k for k in (j-1, j) if 0 <= k < len(numbers)]
        return round(ratings[min(nearest, key=lambda k: abs(numbers[k]-here))])


def name_lookup(meta, rating, labelled, test):
    """Error of guessing a test player's rating from the same name's rating in another game."""
    by_name = {}
    for i in np.flatnonzero(labelled):
        by_name.setdefault(meta[i]['name'], []).append(i)
    guess, true = [], []
    for i in np.flatnonzero(test):
        others = [j for j in by_name.get(meta[i]['name'], ()) if j != i]
        if others:
            guess.append(np.median(rating[others]))
            true.append(rating[i])
    return metrics_of(np.asarray(guess), np.asarray(true)) if len(true) > 2 else None


def winners(X, names):
    """1.0 if the row's player won, by the stream: a lone surrender loses, else the last to order wins."""
    col = lambda n: X[:, names.index(n)]
    own, opp, last = col('own.surrendered'), col('opp.surrendered'), col('own.last_to_order')
    return np.where(own != opp, 1.0-own, last).astype(np.float64)


def fit_ratings(prior, a, b, won, prior_sd, error):
    """MAP ratings: Elo results (with heuristic error) around a Gaussian prior per identity."""
    import torch
    c = math.log(10)/400
    R = torch.tensor(prior, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(prior, dtype=torch.float64)
    A, B = torch.as_tensor(np.asarray(a)), torch.as_tensor(np.asarray(b))
    Y = torch.as_tensor(np.asarray(won), dtype=torch.float64)
    opt = torch.optim.LBFGS([R], max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        p = error+(1-2*error)*torch.sigmoid(c*(R[A]-R[B]))
        loss = -(Y*torch.log(p)+(1-Y)*torch.log(1-p)).sum()+(((R-mu)/prior_sd)**2).sum()/2
        loss.backward()
        return loss
    opt.step(closure)
    return R.detach().numpy()


def joint(args):
    """Pool each identity's games: mean style estimate as prior, then fit to results.

    Validated on Generals Online with true ratings hidden (held-out names, MAE vs each
    game's pre-match rating): per-game style 90.6, pooled style 78.6, joint 73.1.
    Ratings are static per identity, so a gamereplays player's years collapse to one.
    """
    data = np.load(args.features)
    names = list(data['names'])
    meta = [json.loads(m) for m in data['meta']]
    # split('\n'), not splitlines(): player names may hold U+2028 and similar separators.
    rows = [json.loads(line) for line in Path(args.estimates).read_text(encoding='utf-8').split('\n') if line]
    assert len(rows) == len(meta) and all(r['match_key'] == m['match_key'] for r, m in zip(rows, meta))
    won = winners(data['X'], names)
    index = {}
    ids = np.asarray([index.setdefault(m['identity'], len(index)) for m in meta])
    style = np.asarray([r['style_estimate'] for r in rows], dtype=np.float64)
    clean = np.asarray([m['clean'] for m in meta])
    # Prior: mean style over the identity's clean games, else over all its games.
    total, count = np.zeros(len(index)), np.zeros(len(index))
    np.add.at(total, ids[clean], style[clean])
    np.add.at(count, ids[clean], 1)
    every_total, every_count = np.zeros(len(index)), np.zeros(len(index))
    np.add.at(every_total, ids, style)
    np.add.at(every_count, ids, 1)
    prior = np.where(count > 0, total/np.maximum(count, 1), every_total/every_count)
    pairs = np.asarray([k for k in range(0, len(meta), 2)
                        if clean[k] and meta[k]['match_key'] == meta[k+1]['match_key']])
    fitted = fit_ratings(prior, ids[pairs], ids[pairs+1], won[pairs], args.prior_sd, args.winner_error)
    truth = np.asarray([r['rating'] if r['rating'] is not None else np.nan for r in rows], dtype=np.float64)
    test = np.asarray([r['split'] == 'test' for r in rows]) & clean & ~np.isnan(truth)
    report = json.loads(Path(args.report).read_text(encoding='utf-8')) if Path(args.report).exists() else {}
    report['joint'] = {'prior_sd': args.prior_sd, 'winner_error': args.winner_error, 'identities': len(index),
                       'games': len(pairs),
                       'go_test_mae': {'per_game_style': float(np.mean(np.abs(style[test]-truth[test]))),
                                       'pooled_style': float(np.mean(np.abs(prior[ids[test]]-truth[test]))),
                                       'joint': float(np.mean(np.abs(fitted[ids[test]]-truth[test])))}}
    retail = {}
    for source in ('gentool', 'gamereplays'):
        games = Counter(ids[i] for i, m in enumerate(meta) if m['archive'] == source and clean[i])
        members = sorted(games)
        values = fitted[members]
        retail[source] = {'identities': len(members), 'with_5_games': sum(1 for k in members if games[k] >= 5),
                          'percentiles': {str(q): round(float(np.percentile(values, q))) for q in (10, 25, 50, 75, 90)}}
    report['joint']['retail'] = retail
    print(json.dumps(report['joint'], indent=1))
    Path(args.report).write_text(json.dumps(report, indent=1), encoding='utf-8')
    with Path(args.estimates).open('w', encoding='utf-8') as stream:
        for r, m, i in zip(rows, meta, ids):
            r['identity'] = m['identity']
            r['joint_estimate'] = round(float(fitted[i]))
            r['games_pooled'] = int(every_count[i])
            r['estimate'] = (r['rating'] if r['rating'] is not None else
                             r['name_rating'] if r['name_rating'] is not None else r['joint_estimate'])
            stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(f'wrote {args.estimates}')


# --- inference only ---------------------------------------------------------

def predict(args):
    """Style estimates from the saved model for metrics rows under `--path-prefix`, written to a separate file.
    Nothing is retrained and rating-estimates.jsonl is untouched (user, 14 Nov 2026: keep the model as is; rate the
    new gamereplays download without changing the ratings existing bundles were selected with)."""
    import torch
    import lightgbm as lgb
    prefix = args.path_prefix.replace(chr(92), '/')
    lines = [line for line in Path(args.metrics).read_text(encoding='utf-8').splitlines()
             if json.loads(line).get('path', '').replace(chr(92), '/').startswith(prefix)]
    allowed = rc.ranked_map_crcs() | rc.ladder_map_crcs(args.metrics)
    X, meta = [], []
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for result in pool.map(_batch, [(lines[i:i+200], allowed) for i in range(0, len(lines), 200)]):
            for x, m in result:
                X.append(x)
                meta.append(m)
    if not X:
        raise SystemExit('no metrics rows under '+prefix)
    every = [f'own.{n}' for n in player_names()]+[f'opp.{n}' for n in player_names()]+GAME_NAMES
    saved = torch.load(args.model, weights_only=False)
    columns = [every.index(n) for n in saved['names']]
    raw = np.asarray(X, dtype=np.float32)[:, columns]
    net = torch.nn.Sequential(torch.nn.Linear(len(columns), saved['width']), torch.nn.GELU(), torch.nn.Dropout(0),
                              torch.nn.Linear(saved['width'], saved['width']), torch.nn.GELU(), torch.nn.Dropout(0),
                              torch.nn.Linear(saved['width'], 1))
    net.load_state_dict(saved['state'])
    net.eval()
    with torch.no_grad():
        z = torch.from_numpy(((raw-saved['mean'])/saved['std']).astype(np.float32))
        mlp_pred = net(z).squeeze(1).numpy().astype(np.float64)*saved['scale']+saved['center']
    booster = lgb.Booster(model_str=saved['gbm'])
    gbm_pred = booster.predict(raw)*saved['scale']+saved['center']
    pred = calibrate((mlp_pred+gbm_pred)/2, saved['quantile_map'])
    with Path(args.estimates).open('w', encoding='utf-8') as stream:
        for m, value in zip(meta, pred):
            stream.write(json.dumps({'match_key': m['match_key'], 'player_index': m['player_index'], 'name': m['name'],
                                     'faction': m['faction'], 'archive': m['archive'], 'clean': m['clean'],
                                     'rating': m['rating'], 'style_estimate': round(float(value)),
                                     'path_prefix': prefix}, ensure_ascii=False)+'\n')
    print(json.dumps({'players': len(meta), 'games': len(lines),
                      'estimates': str(args.estimates)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('features')
    f.add_argument('--metrics', default=str(rc.METRICS))
    f.add_argument('--out', default=str(FEATURES))
    f.add_argument('--workers', type=int, default=16)
    f.add_argument('--limit', type=int, default=0)
    t = sub.add_parser('train')
    t.add_argument('--features', default=str(FEATURES))
    t.add_argument('--model', default=str(MODEL))
    t.add_argument('--estimates', default=str(ESTIMATES))
    t.add_argument('--report', default=str(REPORT))
    t.add_argument('--own-only', action='store_true', help='ignore the opponent block')
    t.add_argument('--with-outcome', action='store_true',
                   help='keep surrender/last-order features (per-game MAE 89.7 vs 90.6); not for `joint`')
    t.add_argument('--exclude-1000', action='store_true', help='drop the provisional starting rating from labels')
    t.add_argument('--band-weights', action='store_true', help='inverse-frequency weights per 100-point rating band')
    t.add_argument('--crawl-labels', action='store_true',
                   help='also learn from the Tank/Nuke crawl (replays/china-generals; rating-selected, highest first)')
    t.add_argument('--width', type=int, default=256)
    t.add_argument('--dropout', type=float, default=0.1)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--weight-decay', type=float, default=1e-3)
    t.add_argument('--batch', type=int, default=512)
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--patience', type=int, default=6)
    t.add_argument('--l2', type=float, default=1.0)
    t.add_argument('--leaves', type=int, default=63)
    t.add_argument('--threads', type=int, default=8)
    t.add_argument('--seed', type=int, default=0)
    j = sub.add_parser('joint')
    j.add_argument('--features', default=str(FEATURES))
    j.add_argument('--estimates', default=str(ESTIMATES))
    j.add_argument('--report', default=str(REPORT))
    j.add_argument('--prior-sd', type=float, default=60.0, help="spread of a player around their mean style estimate")
    j.add_argument('--winner-error', type=float, default=0.05, help="share of games the winner heuristic gets wrong")
    q = sub.add_parser('predict', help='Rate metrics rows with the saved model (no training) into a separate file')
    q.add_argument('--path-prefix', required=True, help="metrics rows whose replay path starts with this, e.g. replays/gamereplays3/")
    q.add_argument('--metrics', default=str(rc.METRICS))
    q.add_argument('--model', default=str(MODEL))
    q.add_argument('--estimates', required=True, help='output .jsonl (never the main estimates file)')
    q.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == 'predict' and Path(args.estimates).resolve() == ESTIMATES.resolve():
        raise SystemExit('predict writes a separate file, not '+str(ESTIMATES))
    {'features': features, 'train': train, 'joint': joint, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
