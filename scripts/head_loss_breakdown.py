"""Teacher-forced per-head loss of a memory-free event-policy checkpoint on a cloud bundle's own rows.

Question it answers: where does the validation loss sit (act-or-wait, which action, unit selection, target,
click cell, click offset, ...), split by action, order kind, game minute and rating; whether any head shows a
train/validation gap; and, with an early and a final checkpoint, which heads still moved (still learnable)
versus which stayed flat (likely the noise floor of human play given what the model sees).

Rows come from the bundle's dataset manifest through event_dataset.EventDataset, so ratings, openers and enemy
identity are exactly as in training; the loss is train_events.loss_terms over the trainer's own teacher_heads,
so a row's summed terms equal its training loss (mean over a whole split = report.json's mean_nll).
Each row is scored alone (the core's state is not carried), which is exact for a memory-free core and is the
first decision of a game for an LSTM one.

The decision head is one flat softmax over actions (wait first). Its loss splits exactly into
  wait rows:   -log p(wait)
  order rows:  -log p(order) - log p(action | order)
reported as 'act_or_wait' and 'which_action'.

usage: python -I -B scripts/head_loss_breakdown.py --bundle .cache/cloud-bundles/top10k
           --dataset data/dataset-r25-memoryfree.json --checkpoint .cache/cloud-runs/memoryfree-10k/best.npz
           --split validation [--matches 100] --output .cache/head-loss/NAME.jsonl
Writes one JSON line per decision per checkpoint; summarise with --summarise FILE.jsonl [...].
"""
import argparse, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def select_matches(dataset, split, count):
    """`count` matches of `split`, a fixed pseudo-random sample by match-key hash (all when count is 0)."""
    keys = sorted((hashlib.sha256(e['match_key'].encode()).hexdigest(), i)
                  for i, e in enumerate(dataset.entries) if e['split'] == split)
    return {i for _, i in (keys[:count] if count else keys)}


def evaluate(args):
    import numpy as np
    import torch
    from engine_runtime import disable_power_throttling
    from event_dataset import EventDataset
    from generals_policy import KINDS, NeuralPolicy
    from torch_training_inputs import cap_logits, teacher_heads
    from train_events import EventLearner, loss_terms
    from training_preparation import prepare_update

    disable_power_throttling()
    torch.set_num_threads(args.threads)
    os.environ['GENERALS_RATING_INPUT'] = '1' if args.rating_input else '0'
    torch.manual_seed(0)
    bundle = Path(args.bundle)
    dataset = EventDataset(bundle/args.dataset)
    shards = Path(json.loads((bundle/'bundle.json').read_text())['shard_source'])
    dataset.shard_path = lambda entry: shards/entry['path']  # the bundle is manifest-only
    chosen = select_matches(dataset, args.split, args.matches)
    references = [r for r in dataset.chunks(args.split) if r[0] in chosen]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as stream:
        for checkpoint in args.checkpoint:
            policy, _, _ = NeuralPolicy.load(checkpoint)
            if tuple(policy.kinds) != tuple(KINDS):
                raise SystemExit('KINDS differ from the checkpoint')
            learner = EventLearner(policy, device=args.device, threads=args.threads)
            model, name = learner.model, Path(checkpoint).parent.name+'/'+Path(checkpoint).stem
            config = policy.config
            actions = [KINDS[k]+('' if o < 0 else ' '+(config['templates'][o] if KINDS[k] in ('build', 'train')
                                                      else config['commands'][o])) for k, o in policy.actions]
            started, total, pending = time.time(), 0, []

            def flush(rows):
                model.legality_key = rows[0]['encoded']['_training_legality']['key']
                prepared = prepare_update(model, rows, learner.pair_budget)
                with torch.no_grad():
                    for part in prepared['microbatches']:
                        plan, n = part['plan'], len(part['rows'])
                        staged = model.stager.stage(part['values'])
                        plan['staged'] = model.stager.stage(plan['values'])
                        selection = tuple(plan['staged'][k] for k in ('command_rows', 'selected_indices', 'selected_lengths'))
                        enc = model.encode_inputs(staged, selection)
                        core, _ = model.core(enc['core_input'][:, None].to(model.dtype),
                                             torch.ones((n, 1), dtype=torch.bool, device=model.device), model.initial_state(n))
                        core = core[:, 0]
                        output = teacher_heads(model, core, {'keys': enc['keys'], 'summary': enc['summary'], 'map': enc['map'],
                                                       'pixels': staged['scatter_pixels'], 'real': staged['real']},
                                               plan, loss_only=True, position_metrics=True)
                        heads = {k: v.float().cpu().numpy() for k, v in output['by_head'].items()}
                        counts = {k: np.asarray(c).reshape(-1) for k, c in output['head_counts'].items()}
                        # The flat action softmax again, for the act-or-wait / which-action split.
                        # network='paced' never scores no-op: orders only, and p(no-op) is what its pace implies.
                        paced = model.config.get('network') == 'paced'
                        mask = plan['staged']['decision_mask']
                        if paced:
                            mask = mask & (model.action_kinds != 0)[None]
                        logp = cap_logits(model.action_logits(core), mask).masked_fill(~mask, -torch.inf).log_softmax(dim=1)
                        target = plan['staged']['actions']
                        p_wait = (output['wait_probability'] if paced else logp[:, 0].exp()).float().cpu().numpy()
                        lp_target = logp.gather(1, target[:, None])[:, 0].float().cpu().numpy()
                        order = logp[:, 1:].masked_fill(~mask[:, 1:], -torch.inf)
                        best_order = (order.argmax(dim=1)+1).cpu().numpy()
                        best = logp.argmax(dim=1).cpu().numpy()
                        target = target.cpu().numpy()
                        # Direction vs distance per spatial row (torch_training_inputs.direction_errors).
                        direction = {}
                        if 'direction' in output:
                            spatial = plan['command_rows'][plan['spatial_rows']]
                            for row, values in zip(spatial.tolist(), output['direction'].cpu().numpy().tolist()):
                                direction[row] = dict(zip(('distance', 'sector_nll', 'mode_angle_error', 'log_distance_ratio'),
                                                          values))
                        for i, row in enumerate(part['rows']):
                            lp = {k: float(v[i]) for k, v in heads.items()}
                            count = {k: int(c[i]) for k, c in counts.items()}
                            terms = loss_terms(model.config, row['target'], lp, count)
                            wait = row['target']['kind'] == 'wait'
                            p_order = max(1.0-float(p_wait[i]), 1e-12)
                            split = ({'act_or_wait': -math.log(max(float(p_wait[i]), 1e-12))} if wait else
                                     {'act_or_wait': -math.log(p_order),
                                      'which_action': -float(lp_target[i])+(0 if paced else math.log(p_order))})
                            stream.write(json.dumps({
                                'checkpoint': name, 'split': args.split, 'match': row['match_key'],
                                'player': row['player'], 'general': row['general'], 'rating': row['rating'],
                                'minute': row['frame']/row['source_ticks_per_second']/60,
                                'kind': row['target']['kind'], 'action': actions[int(target[i])],
                                'p_wait': float(p_wait[i]), 'correct': bool(best[i] == target[i]),
                                'order_correct': (not wait) and bool(best_order[i] == target[i]),
                                'units': count.get('unit', 0),
                                'nll': {k: -v for k, v in lp.items() if v}, 'count': {k: c for k, c in count.items() if c},
                                'terms': terms, 'decision_split': split, 'loss': sum(terms.values()),
                                **({'direction': direction[i]} if i in direction else {})})+'\n')
                return len(rows)

            for reference in references:
                pending.extend(dataset.read(reference))
                while len(pending) >= args.batch:
                    total += flush(pending[:args.batch]); pending = pending[args.batch:]
                    if total % (args.batch*20) == 0:
                        print('%s %d rows %.0f rows/s' % (name, total, total/(time.time()-started)), flush=True)
            if pending:
                total += flush(pending)
            print('%s done: %d rows in %.0fs' % (name, total, time.time()-started), flush=True)


def summarise(paths, top):
    """Per checkpoint and split: loss by head (decision split into act-or-wait and which-action), by order kind
    and the heads inside each, by action, by game minute and by rating."""
    stats = defaultdict(lambda: {'rows': 0, 'loss': 0.0, 'terms': defaultdict(float), 'nll': defaultdict(float),
                                 'tokens': defaultdict(int), 'rows_with': defaultdict(int), 'split': defaultdict(float),
                                 'kind': defaultdict(lambda: defaultdict(float)), 'action': defaultdict(lambda: [0, 0.0, 0]),
                                 'minute': defaultdict(lambda: [0, 0.0]), 'rating': defaultdict(lambda: [0, 0.0]),
                                 'wait_p': [0.0, 0.0, 0, 0], 'correct': 0, 'order_correct': 0, 'orders': 0})
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                s = stats[(r['checkpoint'], r['split'])]
                s['rows'] += 1; s['loss'] += r['loss']; s['correct'] += r['correct']
                for k, v in r['terms'].items():
                    s['terms'][k] += v
                for k, v in r['nll'].items():
                    s['nll'][k] += v; s['tokens'][k] += r['count'].get(k, 1); s['rows_with'][k] += 1
                for k, v in r['decision_split'].items():
                    s['split'][k] += v
                kind = s['kind'][r['kind']]
                kind['rows'] += 1; kind['loss'] += r['loss']
                for k, v in r['terms'].items():
                    kind[k] += v
                for k, v in r['decision_split'].items():
                    kind['/'+k] += v
                a = s['action'][r['action']]; a[0] += 1; a[1] += r['loss']; a[2] += r['correct']
                m = s['minute'][min(int(r['minute']//5)*5, 30)]; m[0] += 1; m[1] += r['loss']
                rating = r['rating']
                band = 'unrated' if rating is None else '<1400' if rating < 1400 else '1400-1700' if rating < 1700 else '1700-2000' if rating < 2000 else '2000+'
                b = s['rating'][band]; b[0] += 1; b[1] += r['loss']
                wait = r['kind'] == 'wait'
                w = s['wait_p']; w[0 if wait else 1] += r['p_wait']; w[2 if wait else 3] += 1
                if not wait:
                    s['orders'] += 1; s['order_correct'] += r['order_correct']
    for (checkpoint, split), s in sorted(stats.items()):
        n, total = s['rows'], s['loss']
        print('\n== %s  %s  rows=%d  mean loss=%.4f  action top-1 %.1f%%  order top-1 (given an order) %.1f%%' % (
            checkpoint, split, n, total/n, 100*s['correct']/n, 100*s['order_correct']/max(s['orders'], 1)))
        w = s['wait_p']
        print('  wait rows %.1f%%; mean p(wait) on wait rows %.3f, on order rows %.3f' % (
            100*w[2]/n, w[0]/max(w[2], 1), w[1]/max(w[3], 1)))
        print('  %-14s %9s %7s %12s %10s' % ('head', 'loss/row', 'share', 'nll/token', 'rows_with'))
        rows = [(k, v) for k, v in s['terms'].items() if k != 'decision']+[('decision: '+k, v) for k, v in s['split'].items()]
        for head, value in sorted(rows, key=lambda kv: -kv[1]):
            key = head.split(': ')[-1]
            print('  %-24s %9.4f %6.1f%% %12s %10s' % (head, value/n, 100*value/total,
                  '%.4f' % (s['nll'][key]/max(s['tokens'][key], 1)) if key in s['nll'] else '',
                  s['rows_with'][key] if key in s['rows_with'] else ''))
        print('  by order kind: rows, mean loss, share of all loss; heads inside the kind (loss per row of the kind)')
        for kind, k in sorted(s['kind'].items(), key=lambda kv: -kv[1]['loss'])[:top]:
            c = k['rows']
            inside = ', '.join('%s %.2f' % (h.lstrip('/'), v/c) for h, v in sorted(k.items(), key=lambda kv: -kv[1])
                               if h not in ('rows', 'loss', 'decision') and v/c >= .05)
            print('    %-20s %7d %6.2f %5.1f%%   %s' % (kind, c, k['loss']/c, 100*k['loss']/total, inside))
        print('  by action (top %d by share): rows, mean loss, share, top-1' % top)
        for action, (c, l, ok) in sorted(s['action'].items(), key=lambda kv: -kv[1][1])[:top]:
            print('    %-48s %7d %6.2f %5.1f%% %5.1f%%' % (action[:48], c, l/c, 100*l/total, 100*ok/c))
        print('  by game minute: ' + ', '.join('%d+ %d %.2f' % (m, c, l/c) for m, (c, l) in sorted(s['minute'].items())))
        print('  by rating:      ' + ', '.join('%s %d %.2f' % (b, c, l/c) for b, (c, l) in sorted(s['rating'].items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle')
    parser.add_argument('--dataset', default='data/dataset.json', help='The manifest inside the bundle')
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--split', default='validation', choices=('train', 'validation'))
    parser.add_argument('--matches', type=int, default=0, help='0 = every match of the split')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--no-rating-input', dest='rating_input', action='store_false')
    parser.add_argument('--output')
    parser.add_argument('--summarise', nargs='+')
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()
    if args.summarise:
        summarise(args.summarise, args.top)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
