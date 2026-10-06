# -*- coding: utf-8 -*-
"""Reproduce Tables 1-4 of

    "SURE-guided level-adaptive wavelet denoising with exponentially decaying
     shrinkage", submitted to Current Science.

Usage
-----
    python reproduce_tables.py                 # full run: 50 realisations
    python reproduce_tables.py --reps 5        # quick smoke run
    python reproduce_tables.py --no-ecg        # skip Part 3 (needs wfdb/PhysioNet)
    python reproduce_tables.py --nocap         # older, uncapped decomposition depth

Output (in --outdir, default ``results/``)
------------------------------------------
    repeat_stats.json    per-scenario means, SDs, win counts, paired differences
    repeat_stats.txt     human-readable report plus LaTeX-ready table rows

Design
------
Every setting is repeated N_REP times with independent, reproducible noise
draws, and all nine methods are evaluated on exactly the same noisy
realisation (paired design). For setting index i and trial t the noise comes
from ``numpy.random.default_rng([BASE_SEED, i, t])``, so results do not depend
on the order in which the settings are executed.
"""
import argparse
import json
import os

import numpy as np
import pywt
from scipy import stats
from scipy.signal import find_peaks

from core import (make_blocks, make_bumps, make_doppler, make_heavi_sine,
                  estimate_sigma, compute_metrics, wavelet_denoise)
from proposed import denoise_new_lam_only, denoise_new_full_robust

BASE_SEED = 20240919
METHODS = ['hard', 'soft', 'firm', 'bayes', 'bishrink',
           'pfixed', 'psure', 'pla', 'plar']
LABELS = {'hard': 'Hard', 'soft': 'Soft', 'firm': 'Firm',
          'bayes': 'BayesShrink', 'bishrink': 'BiShrink',
          'pfixed': 'Prop.-Fixed', 'psure': 'Prop.-SURE',
          'pla': 'Prop.-LA', 'plar': 'Prop.-LA-R'}


# ---------------------------------------------------------------- settings ---
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reps', type=int, default=50,
                   help='noise realisations per setting (default 50)')
    p.add_argument('--outdir', default='results',
                   help='where the JSON/report are written (default results/)')
    p.add_argument('--nocap', action='store_true',
                   help='use the plain log2 decomposition depth instead of the '
                        'dwt_max_level cap used in the manuscript')
    p.add_argument('--no-ecg', action='store_true',
                   help='skip Part 3 (MIT-BIH), which downloads data')
    return p.parse_args()


ARGS = parse_args()
N_REP = ARGS.reps
CAP_LEVEL = not ARGS.nocap


def auto_level(n):
    """Decomposition depth used throughout the manuscript."""
    j = max(1, min(int(np.floor(np.log2(n))) - 2, 8))
    if CAP_LEVEL:
        j = min(j, pywt.dwt_max_level(n, pywt.Wavelet('db8').dec_len))
    return j


# ------------------------------------------------------------ step / Gibbs ---
def make_step_signal(n=1024):
    """Piecewise-constant test signal with three plateaus."""
    s = np.zeros(n)
    s[300:310] = 1.0
    s[500:520] = 0.8
    s[700:750] = 0.6
    return s


def detect_gibbs_oscillations(clean, denoised, window=30, std_thresh=0.003):
    """Count band and total local energy of oscillations inside flat regions."""
    n = len(clean)
    flat = np.abs(clean) < 0.02
    regions, energy = 0, 0.0
    for i in range(window, n - window):
        if flat[i]:
            local_std = np.std(denoised[i - window:i + window])
            if local_std > std_thresh:
                regions += 1
                energy += local_std ** 2
    return regions, energy


# --------------------------------------------------------------- baselines ---
def soft_th(w, lam):
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)


def denoise_bayes(y, wavelet='db8', level=None):
    """Subband-adaptive BayesShrink (Chang, Yu & Vetterli, 2000)."""
    n = len(y)
    level = level or auto_level(n)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    c_a, c_ds = coeffs[0], coeffs[1:]
    sigma = estimate_sigma(np.asarray(c_ds[-1]).flatten())
    th = []
    for c_d in c_ds:
        c_d = np.asarray(c_d, dtype=float)
        sigma_y2 = np.mean(c_d ** 2)
        sigma_x = np.sqrt(max(sigma_y2 - sigma ** 2, 0.0))
        lam = sigma ** 2 / sigma_x if sigma_x > 1e-12 else np.max(np.abs(c_d))
        th.append(soft_th(c_d, lam))
    return pywt.waverec([c_a] + th, wavelet)[:n]


def _bivar_shrink(wc, wp, sigma):
    """Bivariate shrinkage map (Sendur & Selesnick, 2002)."""
    out = np.zeros_like(wc)
    win = 3
    n = wc.size
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        sigma_y2 = np.mean(wc[lo:hi] ** 2 + wp[lo:hi] ** 2)
        ss = np.sqrt(max(sigma_y2 - sigma ** 2, 0.0))
        if ss < 1e-12:
            continue
        mag = np.sqrt(wc[i] ** 2 + wp[i] ** 2)
        t = max(mag - np.sqrt(3.0) * sigma ** 2 / ss, 0.0) / max(mag, 1e-12)
        out[i] = wc[i] * t
    return out


def denoise_bishrink(y, wavelet='db8', level=None):
    """Bivariate shrinkage (Sendur & Selesnick, 2002)."""
    n = len(y)
    level = level or auto_level(n)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    c_a = coeffs[0]
    c_ds = [np.asarray(c, dtype=float) for c in coeffs[1:]]
    sigma = estimate_sigma(c_ds[-1].flatten())
    th = []
    for j, c_d in enumerate(c_ds):
        parent = np.zeros_like(c_d) if j == 0 else c_ds[j - 1][np.arange(len(c_d)) // 2]
        th.append(_bivar_shrink(c_d, parent, sigma))
    return pywt.waverec([c_a] + th, wavelet)[:n]


def denoise_all(y):
    """Run the nine compared methods on one noisy realisation."""
    lv = auto_level(len(y))
    out = {}
    for m in ['hard', 'soft', 'firm']:
        out[m] = wavelet_denoise(y, threshold_func=m, level=lv)[0]
    out['pfixed'] = wavelet_denoise(y, threshold_func='new', alpha=2.0, level=lv)[0]
    out['psure'] = wavelet_denoise(y, threshold_func='new', alpha='sure', level=lv)[0]
    out['pla'] = denoise_new_lam_only(y, alpha=2.0, level=lv)[0]
    out['plar'] = denoise_new_full_robust(y, level=lv)[0]
    out['bayes'] = denoise_bayes(y, level=lv)
    out['bishrink'] = denoise_bishrink(y, level=lv)
    return out


def snr_of(clean, x_hat):
    return compute_metrics(clean, x_hat)[1]


def summary(values):
    v = np.asarray(values, dtype=float)
    return {'mean': float(np.mean(v)),
            'sd': float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            'n': int(v.size)}


def paired_report(diffs_per_scenario):
    """Wilcoxon signed-rank test on the scenario-level mean differences."""
    per_mean = [float(np.mean(d)) for d in diffs_per_scenario]
    all_d = np.concatenate([np.asarray(d, dtype=float) for d in diffs_per_scenario])
    stat, p = stats.wilcoxon(per_mean) if len(per_mean) > 5 else (np.nan, np.nan)
    return {'mean_diff': float(np.mean(all_d)),
            'sd_diff': float(np.std(all_d, ddof=1)),
            'scenario_mean_diffs': [round(x, 4) for x in per_mean],
            'wilcoxon_p': float(p) if np.isfinite(p) else None}


# ================================================================= Part 1 ====
def run_synthetic(results):
    print('=' * 78)
    print('Part 1: synthetic signals, N_REP=%d' % N_REP)
    print('=' * 78)
    synth = [('Blocks', make_blocks), ('Bumps', make_bumps),
             ('Doppler', make_doppler), ('HeaviSine', make_heavi_sine)]
    sigmas = [0.1, 0.3, 0.5, 0.7]
    n = 2048
    rows, wins = [], {m: 0 for m in METHODS}
    diff_hard, diff_bayes = [], []
    all_means = {m: [] for m in METHODS}

    idx = 0
    for name, fn in synth:
        clean = fn(n)
        for sigma in sigmas:
            idx += 1
            acc = {m: [] for m in METHODS}
            for t in range(N_REP):
                rng = np.random.default_rng([BASE_SEED, idx, t])
                noisy = clean + sigma * rng.standard_normal(n)
                res = denoise_all(noisy)
                for m in METHODS:
                    acc[m].append(snr_of(clean, res[m]))
            row = {'signal': name, 'sigma': sigma,
                   'stats': {m: summary(acc[m]) for m in METHODS}}
            rows.append(row)
            for m in METHODS:
                all_means[m].append(row['stats'][m]['mean'])
            best = max(METHODS, key=lambda m: row['stats'][m]['mean'])
            wins[best] += 1
            row['best'] = best
            diff_hard.append(np.array(acc['pla']) - np.array(acc['hard']))
            diff_bayes.append(np.array(acc['pla']) - np.array(acc['bayes']))
            print('  %-10s sigma=%.1f best=%-12s ' % (name, sigma, LABELS[best])
                  + ' '.join('%s=%5.2f' % (m, row['stats'][m]['mean'])
                             for m in ['hard', 'bayes', 'pla', 'plar']))

    results['synthetic'] = {
        'n_rep': N_REP, 'rows': rows,
        'overall_mean': {m: float(np.mean(all_means[m])) for m in METHODS},
        'wins': {LABELS[k]: v for k, v in wins.items()},
        'paired_pla_vs_hard': paired_report(diff_hard),
        'paired_pla_vs_bayes': paired_report(diff_bayes)}
    print('\n  overall mean SNR: '
          + ' '.join('%s=%.2f' % (LABELS[m], np.mean(all_means[m])) for m in METHODS))
    print('  per-scenario wins: '
          + ' '.join('%s=%d' % (LABELS[m], wins[m]) for m in METHODS))


# ================================================================= Part 2 ====
def run_gibbs(results):
    print('\n' + '=' * 78)
    print('Part 2: ringing on the step signal, N_REP=%d' % N_REP)
    print('=' * 78)
    g_sigmas = [0.05, 0.10, 0.15, 0.20]
    n = 1024
    clean = make_step_signal(n)
    rows = []
    idx = 100
    for sigma in g_sigmas:
        idx += 1
        energy = {m: [] for m in METHODS}
        snrs = {m: [] for m in METHODS}
        for t in range(N_REP):
            rng = np.random.default_rng([BASE_SEED, idx, t])
            noisy = clean + sigma * rng.standard_normal(n)
            res = denoise_all(noisy)
            for m in METHODS:
                energy[m].append(detect_gibbs_oscillations(clean, res[m])[1])
                snrs[m].append(snr_of(clean, res[m]))
        rows.append({'sigma': sigma,
                     'energy': {m: summary(energy[m]) for m in METHODS},
                     'snr': {m: summary(snrs[m]) for m in METHODS}})
        print('  sigma=%.2f mean energy: ' % sigma
              + ' '.join('%s=%6.2f' % (m, np.mean(energy[m])) for m in METHODS))

    mean_energy = {m: float(np.mean([r['energy'][m]['mean'] for r in rows]))
                   for m in METHODS}
    mean_snr = {m: float(np.mean([r['snr'][m]['mean'] for r in rows]))
                for m in METHODS}
    vs_hard = {m: (1.0 - mean_energy[m] / mean_energy['hard']) * 100.0
               for m in METHODS}
    results['gibbs'] = {'n_rep': N_REP, 'rows': rows,
                        'mean_energy': mean_energy, 'mean_snr': mean_snr,
                        'reduction_vs_hard_percent': vs_hard}
    print('\n  vs Hard (energy): '
          + ' '.join('%s=%+.1f%%' % (LABELS[m], vs_hard[m]) for m in METHODS))


# ================================================================= Part 3 ====
def r_peak_metrics(clean, est, fs, height=0.35, dist_sec=0.4):
    """Sensitivity and RMSE (ms) of R-peak detection."""
    dist = int(dist_sec * fs)
    pk_c, _ = find_peaks(clean, height=height, distance=dist)
    pk_e, _ = find_peaks(est, height=height, distance=dist)
    if len(pk_c) == 0:
        return 0.0, float('nan')
    hits, errs = 0, []
    for p in pk_c:
        d = np.abs(pk_e - p)
        if d.size and d.min() <= dist:
            hits += 1
            errs.append(d.min())
    sens = hits / len(pk_c)
    rmse = np.sqrt(np.mean(np.square(errs))) / fs * 1000.0 if errs else float('nan')
    return sens, rmse


def run_ecg(results):
    print('\n' + '=' * 78)
    print('Part 3: MIT-BIH records 100/101, N_REP=%d' % N_REP)
    print('=' * 78)
    try:
        import wfdb
    except ImportError:
        print('  [SKIP] wfdb is not installed (pip install wfdb)')
        return

    rows = []
    idx = 200
    for rec in ['100', '101']:
        try:
            rec_data = wfdb.rdrecord(rec, pn_dir='mitdb')
        except Exception as exc:                                  # noqa: BLE001
            print('  [SKIP] record %s: %s' % (rec, exc))
            continue
        fs = rec_data.fs
        start = int(5 * fs)
        seg = rec_data.p_signal[start:start + int(10 * fs), 0]
        clean = seg - np.mean(seg)
        clean = clean / np.max(np.abs(clean))
        n = len(clean)
        for sigma in [0.1, 0.3, 0.5]:
            idx += 1
            snrs = {m: [] for m in METHODS}
            sens = {m: [] for m in METHODS}
            rmse = {m: [] for m in METHODS}
            for t in range(N_REP):
                rng = np.random.default_rng([BASE_SEED, idx, t])
                noisy = clean + sigma * rng.standard_normal(n)
                res = denoise_all(noisy)
                for m in METHODS:
                    snrs[m].append(snr_of(clean, res[m]))
                    s, e = r_peak_metrics(clean, res[m], fs)
                    sens[m].append(s)
                    rmse[m].append(e)
            rows.append({'record': rec, 'sigma': sigma, 'n_samples': int(n),
                         'snr': {m: summary(snrs[m]) for m in METHODS},
                         'sensitivity': {m: summary(sens[m]) for m in METHODS},
                         'rmse_ms': {m: summary(rmse[m]) for m in METHODS}})
            last = rows[-1]
            print('  ECG%s sigma=%.1f ' % (rec, sigma)
                  + ' '.join('%s=%5.2f' % (m, last['snr'][m]['mean'])
                             for m in ['hard', 'bayes', 'pla', 'plar']))

    if not rows:
        return
    overall = {m: float(np.mean([r['snr'][m]['mean'] for r in rows]))
               for m in METHODS}
    results['ecg'] = {'n_rep': N_REP, 'rows': rows,
                      'overall_mean_snr': overall, 'fs': None}
    print('\n  overall mean SNR: '
          + ' '.join('%s=%.2f' % (LABELS[m], overall[m]) for m in METHODS))


# ================================================================= report ====
def write_report(results, json_path, txt_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    lines = []
    a = lines.append
    a('Repeated-noise statistics (%d independent realisations per setting)' % N_REP)
    a('Paired design: all methods see the same noise draw in every trial.')
    a('')

    if 'synthetic' in results:
        s = results['synthetic']
        a('--- Table 1: synthetic signals (mean SNR +/- SD, dB) ---')
        for r in s['rows']:
            a('%-10s sigma=%.1f' % (r['signal'], r['sigma']))
            for m in METHODS:
                st = r['stats'][m]
                a('    %-12s %6.2f +/- %5.2f' % (LABELS[m], st['mean'], st['sd']))
        a('')
        a('  LaTeX rows:')
        for r in s['rows']:
            a('    %s & %s & %s \\\\'
              % (r['signal'], r['sigma'],
                 ' & '.join('%.2f' % r['stats'][m]['mean'] for m in METHODS)))
        a('  overall mean: '
          + ' '.join('%s=%.2f' % (LABELS[m], s['overall_mean'][m]) for m in METHODS))
        a('  per-scenario wins: %s' % s['wins'])
        for key, label in (('paired_pla_vs_hard', 'Prop.-LA - Hard'),
                           ('paired_pla_vs_bayes', 'Prop.-LA - BayesShrink')):
            a('  paired %s: mean %.2f dB, Wilcoxon p=%s'
              % (label, s[key]['mean_diff'], s[key]['wilcoxon_p']))
        a('')

    if 'gibbs' in results:
        g = results['gibbs']
        a('--- Table 2: ringing energy and SNR on the step signal ---')
        a('  LaTeX rows (oscillation energy):')
        for r in g['rows']:
            a('    %s & %s \\\\'
              % (r['sigma'],
                 ' & '.join('%.2f' % r['energy'][m]['mean'] for m in METHODS)))
        a('  mean energy: '
          + ' '.join('%s=%.2f' % (LABELS[m], g['mean_energy'][m]) for m in METHODS))
        a('  mean SNR:    '
          + ' '.join('%s=%.2f' % (LABELS[m], g['mean_snr'][m]) for m in METHODS))
        a('  vs Hard:     '
          + ' '.join('%s=%+.1f%%' % (LABELS[m], g['reduction_vs_hard_percent'][m])
                     for m in METHODS))
        a('')

    if 'ecg' in results:
        e = results['ecg']
        a('--- Tables 3 and 4: MIT-BIH records 100 and 101 ---')
        for r in e['rows']:
            a('  ECG%s sigma=%.1f' % (r['record'], r['sigma']))
            for m in METHODS:
                a('    %-12s SNR %6.2f +/- %5.2f   sens %.2f   RMSE %8.1f ms'
                  % (LABELS[m], r['snr'][m]['mean'], r['snr'][m]['sd'],
                     r['sensitivity'][m]['mean'], r['rmse_ms'][m]['mean']))
        a('  overall mean SNR: '
          + ' '.join('%s=%.2f' % (LABELS[m], e['overall_mean_snr'][m])
                     for m in METHODS))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    results = {'meta': {'n_rep': N_REP, 'base_seed': BASE_SEED,
                        'methods': METHODS, 'labels': LABELS,
                        'level_cap': CAP_LEVEL}}
    run_synthetic(results)
    run_gibbs(results)
    if not ARGS.no_ecg:
        run_ecg(results)
    os.makedirs(ARGS.outdir, exist_ok=True)
    suffix = '' if CAP_LEVEL else '_nocap'
    write_report(results,
                 os.path.join(ARGS.outdir, 'repeat_stats%s.json' % suffix),
                 os.path.join(ARGS.outdir, 'repeat_stats%s.txt' % suffix))
    print('\nDONE -> %s' % os.path.join(ARGS.outdir, 'repeat_stats%s.json' % suffix))


if __name__ == '__main__':
    main()
