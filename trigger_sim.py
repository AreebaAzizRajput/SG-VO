#!/usr/bin/env python3
"""Offline trigger simulation on recorded probe-loss traces.

Replays per-frame loss traces (from test_vo_online.py --probe-only) through
candidate trigger designs and reports how often each would have adapted.
On in-domain traces the ideal trigger fires (near) zero times; the same
designs can later be replayed on domain-shift traces (vKITTI) where they
SHOULD fire early and persistently.

Usage:
    python trigger_sim.py probe_losses_09.txt probe_losses_10.txt
"""
import sys
import numpy as np


def spike_ema(losses, theta=1.2, decay=0.9):
    """Current rule in test_vo_online.py: fire when loss > theta * EMA."""
    fires = np.zeros(len(losses), dtype=bool)
    ema = losses[0]
    for t, l in enumerate(losses):
        ema = decay * ema + (1 - decay) * l
        fires[t] = l > theta * ema
    return fires


def dual_ema(losses, theta=1.1, fast_decay=0.9, slow_decay=0.99):
    """Fire when the FAST average sits above theta x the SLOW average.
    Transients wash out of the fast average before it can dominate."""
    fires = np.zeros(len(losses), dtype=bool)
    fast = slow = losses[0]
    for t, l in enumerate(losses):
        fast = fast_decay * fast + (1 - fast_decay) * l
        slow = slow_decay * slow + (1 - slow_decay) * l
        fires[t] = fast > theta * slow
    return fires


def cusum(losses, kappa_sigmas=1.0, h_sigmas=8.0, ref_frames=100):
    """Classic one-sided CUSUM with a fixed reference from the first
    ref_frames (stand-in for training-domain statistics).
    kappa/h are expressed in units of the reference std."""
    ref = losses[:ref_frames]
    mu, sigma = ref.mean(), ref.std() + 1e-12
    kappa, h = kappa_sigmas * sigma, h_sigmas * sigma
    fires = np.zeros(len(losses), dtype=bool)
    s = 0.0
    for t, l in enumerate(losses):
        s = max(0.0, s + (l - mu - kappa))
        fires[t] = s > h
        if fires[t]:
            s = 0.0   # reset after firing: adaptation is assumed to run
    return fires


def cusum_selfcal(losses, kappa_sigmas=1.0, h_sigmas=8.0, slow_decay=0.999,
                  warmup=100):
    """Self-calibrating CUSUM: the reference mean is a very slow EMA the
    system maintains online (no offline statistics needed at deployment).
    Std is tracked the same way. No firing during the first `warmup`
    frames while the statistics stabilise (var starts at 0, so without a
    warmup the tiny initial sigma makes the trigger fire immediately)."""
    fires = np.zeros(len(losses), dtype=bool)
    mu = losses[0]
    var = 0.0
    s = 0.0
    for t, l in enumerate(losses):
        d = l - mu
        mu = slow_decay * mu + (1 - slow_decay) * l
        var = slow_decay * var + (1 - slow_decay) * d * d
        sigma = np.sqrt(var) + 1e-12
        s = max(0.0, s + (l - mu - kappa_sigmas * sigma))
        fires[t] = (s > h_sigmas * sigma) and t >= warmup
        if fires[t]:
            s = 0.0
    return fires


DESIGNS = {
    'spike theta=1.2 (current)':      lambda x: spike_ema(x, 1.2),
    'spike theta=1.5':                lambda x: spike_ema(x, 1.5),
    'dual-EMA theta=1.05':            lambda x: dual_ema(x, 1.05),
    'dual-EMA theta=1.10':            lambda x: dual_ema(x, 1.10),
    'dual-EMA theta=1.20':            lambda x: dual_ema(x, 1.20),
    'CUSUM k=0.5s h=6s':              lambda x: cusum(x, 0.5, 6.0),
    'CUSUM k=1.0s h=8s':              lambda x: cusum(x, 1.0, 8.0),
    'CUSUM k=1.0s h=12s':             lambda x: cusum(x, 1.0, 12.0),
    'selfcal-CUSUM k=1.0s h=8s':      lambda x: cusum_selfcal(x, 1.0, 8.0),
    'selfcal-CUSUM k=1.0s h=12s':     lambda x: cusum_selfcal(x, 1.0, 12.0),
}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    traces = {p: np.loadtxt(p) for p in sys.argv[1:]}

    for path, losses in traces.items():
        print(f'\n=== {path}: {len(losses)} frames | '
              f'mean {losses.mean():.4f}  std {losses.std():.4f}  '
              f'min {losses.min():.4f}  max {losses.max():.4f} ===')
        print(f'{"design":34s} {"fires":>6s} {"rate":>7s}  first-fire')
        for name, fn in DESIGNS.items():
            fires = fn(losses)
            n = int(fires.sum())
            first = int(np.argmax(fires)) if n else -1
            print(f'{name:34s} {n:6d} {100*n/len(losses):6.1f}%  '
                  f'{first if n else "never"}')


if __name__ == '__main__':
    main()
