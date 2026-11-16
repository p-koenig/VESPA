"""Greedy return evaluation of CartPole policies (Fig. 3: 1,000 episodes per policy).

    python analysis/eval_returns.py --policy NAME=CKPT [--policy NAME=CKPT ...] --out-dir OUT [--episodes 1000] [--seed 12345]

For every policy it runs N greedy (argmax) episodes of ``VerifyCartPole-v0`` (CartPole-v1 dynamics, 500-step
cap), resetting episode i with seed ``rng.integers(0, 2**31 - 1)`` of ``numpy.random.default_rng(seed)`` -- the
same loop as the validation rollouts of the fine-tuning code -- and keeps per-episode data:
return, length, terminated, start state and final state. It also reports the validation routine's own means
over the first 30 and all N episodes (``official_mean30``/``official_meanN``) as a cross-check.
Writes OUT/ret<N>_<NAME>.json. CPU is sufficient (about 1 minute per policy).
"""
import argparse, json, os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax
import gymnasium as gym

from vespa.pretraining.checkpoint import load_checkpoint, rollout_mean_return
from vespa.sympol import SYMPOL_RL
from vespa.pretraining import envs  # registers VerifyCartPole-v0


def evaluate(pid, ckpt, N, SEED, env_id='VerifyCartPole-v0'):
    t0 = time.time()
    ck = load_checkpoint(ckpt); ac = ck['actor_config']
    actor = SYMPOL_RL(obs_dim=int(ac['obs_dim']), action_dim=int(ac['action_dim']), action_type=ck['action_type'],
                      depth=int(ac['depth']), n_estimators=int(ac['n_estimators']))
    idx = jax.tree.map(jax.numpy.asarray, ck['indices']); f = jax.jit(lambda p, o: actor.apply(p, o, indices=idx))
    P = jax.tree.map(jax.numpy.asarray, ck['params'])
    env = gym.make(env_id); cap = int(env.spec.max_episode_steps)
    rng = np.random.default_rng(SEED); rets = []; lens = []; term = []; s0 = []; sT = []
    for ep in range(N):
        rs = int(rng.integers(0, 2**31 - 1)); obs, _ = env.reset(seed=rs)
        s0.append([float(v) for v in np.asarray(env.unwrapped.state, dtype=np.float64)])
        R = 0.0; t = 0; te = False
        for _ in range(cap):
            o = f(P, np.asarray(obs, dtype=np.float32)[None, :]); a = int(np.argmax(np.asarray(o)[0]))
            obs, r, te, tr, _ = env.step(a); R += float(r); t += 1
            if te or tr: break
        rets.append(R); lens.append(t); term.append(bool(te))
        sT.append([float(v) for v in np.asarray(env.unwrapped.state, dtype=np.float64)])
    env.close()
    m30, _, _ = rollout_mean_return(f, P, 30, SEED)
    mN, sN, _ = rollout_mean_return(f, P, N, SEED)
    return dict(pid=pid, N=N, seed=SEED, cap=cap, rets=rets, lens=lens, term=term, s0=s0, sT=sT,
                official_mean30=float(m30), official_meanN=float(mN), official_stdN=float(sN),
                replica_mean30=float(np.mean(rets[:30])), replica_meanN=float(np.mean(rets)), sec=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--policy', action='append', required=True, help='NAME=checkpoint_dir')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for spec in a.policy:
        pid, ckpt = spec.split('=', 1)
        row = evaluate(pid, ckpt, a.episodes, a.seed)
        json.dump(row, open(os.path.join(a.out_dir, f'ret{a.episodes}_{pid}.json'), 'w'))
        print(pid, 'mean', row['replica_meanN'], 'min', min(row['rets']), 'official', row['official_meanN'], f"{row['sec']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
