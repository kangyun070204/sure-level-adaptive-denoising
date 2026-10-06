"""The proposed level-adaptive shrinkage rule.

Two variants are used in the manuscript:

``denoise_new_lam_only``     (Prop.-LA)
    Every detail subband selects its own threshold by minimising a closed-form
    SURE of the MSE, the decay parameter alpha being fixed.

``denoise_new_full_robust``  (Prop.-LA-R)
    As above, with two safeguards: subbands judged to be noise dominated fall
    back to the global threshold and the globally selected alpha, and the
    threshold search range is bounded from both sides.

``denoise_new_full`` is the unconstrained joint (lambda, alpha) search per
subband; it is kept because Prop.-LA-R is derived from it.
"""

import numpy as np
import pywt
from scipy.optimize import minimize_scalar

from core import (estimate_sigma, visu_shrink_threshold, new_threshold,
                  select_alpha_sure)


def sure_value_paper(alphas, coeffs, sigma, lam):
    """SURE of Eq. (8) of the manuscript,

        SURE(a) = sum_{|w| <= lam} (w^2 - sigma^2)
                  + sum_{|w| > lam} [ lam^2 e^{-2 a d}
                                      + sigma^2
                                      + 2 sigma^2 a lam e^{-a d} ],
        d = |w| - lam > 0.

    The first term does not depend on a but is included so that the returned
    values are comparable across alpha.  Returns an array shaped like
    ``alphas``.
    """
    w = np.asarray(coeffs).flatten()
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    aw = np.abs(w)
    mask_le = aw <= lam
    mask_gt = aw > lam
    const_sum_le = np.sum(w[mask_le] ** 2 - sigma ** 2)
    const = (const_sum_le + np.sum(sigma ** 2) * mask_gt.sum()
             if np.any(mask_gt) else const_sum_le)
    if not np.any(mask_gt):
        return np.full(alphas.shape, float(const))
    wgt = w[mask_gt]
    awgt = aw[mask_gt]
    d = awgt - lam                              # (K,)
    e_a = np.exp(-np.outer(alphas, d))          # (M, K)
    e_2a = e_a ** 2
    term = lam ** 2 * e_2a + 2 * sigma ** 2 * alphas[:, None] * lam * e_a
    return const + term.sum(axis=1)


def select_alpha_sure_single(coeffs, sigma, lam, lo=0.001, hi=1000.0,
                             alpha_max=None):
    """Select alpha for one subband from the SURE above.

    A coarse logarithmic grid localises the global minimum (so that a
    non-unimodal SURE cannot trap the search), then a bounded Brent refinement
    polishes it.  ``alpha_max`` clips the admissible range.  Returns ``None``
    when the subband has too few supra-threshold coefficients to identify
    alpha, leaving the fallback decision to the caller.
    """
    c = np.asarray(coeffs).flatten()
    n = c.size
    active = int(np.count_nonzero(np.abs(c) > lam))
    if active < max(2, 0.005 * n):
        return None

    hi_eff = alpha_max if alpha_max is not None else hi
    grid = np.logspace(np.log10(lo), np.log10(hi_eff), 60)
    vals = sure_value_paper(grid, c, sigma, lam)
    i = int(np.argmin(vals))
    a_lo = grid[i - 1] if i > 0 else lo
    a_hi = grid[i + 1] if i < len(grid) - 1 else hi_eff
    try:
        res = minimize_scalar(
            lambda a: float(sure_value_paper(a, c, sigma, lam)),
            bounds=(a_lo, a_hi), method='bounded',
            options={'xatol': 1e-3, 'maxiter': 80})
        return float(res.x)
    except Exception:
        return float(grid[i])


def select_lam_alpha_single(coeffs, sigma, lam0, n_lam=15, alpha_lo=0.01,
                            alpha_hi=50.0):
    """Joint per-subband search for (lambda, alpha) minimising that SURE.

    lambda is scanned on a logarithmic grid over [0.3 sigma, lam0] and alpha on
    a logarithmic grid over [alpha_lo, alpha_hi].  Returns ``(None, None)`` if
    every candidate threshold kills the whole subband.
    """
    c = np.asarray(coeffs).flatten()
    if np.count_nonzero(np.abs(c) > 0.3 * sigma) < 2:
        return None, None
    lam_grid = np.logspace(np.log10(0.3 * sigma), np.log10(max(lam0, 0.3 * sigma * 1.1)),
                           n_lam)
    alpha_grid = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), 30)
    best_val = np.inf
    best = (None, None)
    for lam in lam_grid:
        active = int(np.count_nonzero(np.abs(c) > lam))
        if active < max(2, 0.005 * c.size):
            continue
        vals = sure_value_paper(alpha_grid, c, sigma, lam)
        i = int(np.argmin(vals))
        if vals[i] < best_val:
            best_val = vals[i]
            best = (lam, float(alpha_grid[i]))
    return best


def select_lam_single(coeffs, sigma, lam0, alpha, n_lam=15, min_active_ratio=0.005):
    """Optimise lambda only, with alpha fixed.

    Used by Prop.-LA to separate the contribution of per-level thresholds from
    that of per-level decay parameters.  Returns None when too few coefficients
    are active.
    """
    c = np.asarray(coeffs).flatten()
    if np.count_nonzero(np.abs(c) > 0.3 * sigma) < 2:
        return None
    lam_grid = np.logspace(np.log10(0.3 * sigma), np.log10(max(lam0, 0.3 * sigma * 1.1)),
                           n_lam)
    best_val, best_lam = np.inf, None
    for lam in lam_grid:
        active = int(np.count_nonzero(np.abs(c) > lam))
        if active < max(2, min_active_ratio * c.size):
            continue
        val = float(sure_value_paper(alpha, c, sigma, lam)[0])
        if val < best_val:
            best_val, best_lam = val, lam
    return best_lam


def select_lam_alpha_single_robust(coeffs, sigma, lam0, n_lam=15, alpha_lo=0.01,
                                   alpha_hi=50.0, min_active0_ratio=0.001,
                                   lam_lo_sigma=0.5, lam_hi_ratio=1.0):
    """Robust joint per-subband search used by Prop.-LA-R.

    1. Subband test: if the fraction of coefficients above the global threshold
       lam0 is below ``min_active0_ratio`` the subband is treated as pure noise
       and ``(None, None)`` is returned, so the caller falls back to the global
       rule.  (A noise-only subband has P(|w| > lam0) around 1e-4, a subband
       carrying signal has markedly more.)
    2. Range limit: lambda is restricted to
       [lam_lo_sigma * sigma, lam_hi_ratio * lam0], so that SURE cannot
       overfit an extreme threshold on a nearly empty subband.
    """
    c = np.asarray(coeffs).flatten()
    n = c.size
    active0 = int(np.count_nonzero(np.abs(c) > lam0))
    if active0 < max(1, min_active0_ratio * n):
        return None, None
    lo = max(0.3 * sigma, lam_lo_sigma * sigma)
    hi = max(lo * 1.05, lam_hi_ratio * lam0)
    lam_grid = np.logspace(np.log10(lo), np.log10(hi), n_lam)
    alpha_grid = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), 30)
    best_val, best = np.inf, (None, None)
    for lam in lam_grid:
        active = int(np.count_nonzero(np.abs(c) > lam))
        if active < max(2, 0.005 * n):
            continue
        vals = sure_value_paper(alpha_grid, c, sigma, lam)
        i = int(np.argmin(vals))
        if vals[i] < best_val:
            best_val, best = vals[i], (float(lam), float(alpha_grid[i]))
    return best


def new_threshold_safe(w, lam, alpha):
    """Numerically safe form of the decaying rule.

    Coefficients with |w| <= lam are set to zero directly, so that the
    exponential never has to be evaluated for large argument; the result is
    mathematically identical to ``new_threshold``.
    """
    w = np.asarray(w, dtype=float)
    aw = np.abs(w)
    mask = aw > lam
    out = np.zeros_like(w)
    if np.any(mask):
        t = aw[mask] - lam                     # > 0
        decay = lam * np.exp(-alpha * t)       # <= lam, no overflow
        shrunk = aw[mask] - decay
        out[mask] = np.sign(w[mask]) * np.maximum(shrunk, 0.0)
    return out


def denoise_new_full(y, wavelet='db8', level=None, fallback_alpha=2.0):
    """Joint per-level optimisation of (lambda, alpha), unconstrained.

    Returns ``(x_hat, sigma_hat, lam0, per_level_paras)``, where
    ``per_level_paras`` holds one ``(lam_j, alpha_j)`` per detail subband, in
    the same order as the detail coefficients.
    """
    N = len(y)
    if level is None:
        level = int(np.floor(np.log2(N))) - 2
        level = max(1, min(level, 8))
        level = min(level, pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len))

    coeffs_orig = pywt.wavedec(y, wavelet, level=level)
    cA = coeffs_orig[0]
    cDs = coeffs_orig[1:]
    sigma_hat = estimate_sigma(cDs[-1])
    lam0 = visu_shrink_threshold(sigma_hat, N)

    paras = []
    cDs_th = []
    for cD in cDs:
        lam_opt, alpha_opt = select_lam_alpha_single(cD, sigma_hat, lam0)
        if lam_opt is None:                      # the whole level is killed
            lam_j, alpha_j = lam0, fallback_alpha
            th = np.zeros_like(np.asarray(cD, dtype=float))
        else:
            lam_j, alpha_j = lam_opt, alpha_opt
            th = new_threshold_safe(cD, lam_j, alpha_j)
        paras.append((lam_j, alpha_j))
        cDs_th.append(th)

    coeffs_th = [cA] + cDs_th
    x_hat = pywt.waverec(coeffs_th, wavelet)[:N]
    return x_hat, sigma_hat, lam0, paras


def denoise_new_full_robust(y, wavelet='db8', level=None, fallback_alpha=None,
                            min_active0_ratio=0.001, lam_lo_sigma=0.5,
                            lam_hi_ratio=1.0):
    """Prop.-LA-R: joint per-level (lambda, alpha) search with the safeguards
    described in :func:`select_lam_alpha_single_robust`.

    Noise-dominated subbands fall back to the global threshold lam0 with the
    alpha selected globally by SURE.  Returns
    ``(x_hat, sigma_hat, lam0, per_level_paras)``.
    """
    N = len(y)
    if level is None:
        level = int(np.floor(np.log2(N))) - 2
        level = max(1, min(level, 8))
        level = min(level, pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len))

    coeffs_orig = pywt.wavedec(y, wavelet, level=level)
    cA = coeffs_orig[0]
    cDs = coeffs_orig[1:]
    sigma_hat = estimate_sigma(cDs[-1])
    lam0 = visu_shrink_threshold(sigma_hat, N)

    if fallback_alpha is None:
        a_glob = select_alpha_sure_single(np.concatenate([c.flatten() for c in cDs]),
                                          sigma_hat, lam0)
        fallback_alpha = 2.0 if a_glob is None else a_glob

    paras = []
    cDs_th = []
    for cD in cDs:
        lam_opt, alpha_opt = select_lam_alpha_single_robust(
            cD, sigma_hat, lam0, min_active0_ratio=min_active0_ratio,
            lam_lo_sigma=lam_lo_sigma, lam_hi_ratio=lam_hi_ratio)
        if lam_opt is None:
            lam_j, alpha_j = lam0, fallback_alpha
            th = new_threshold_safe(cD, lam_j, alpha_j)
        else:
            lam_j, alpha_j = lam_opt, alpha_opt
            th = new_threshold_safe(cD, lam_j, alpha_j)
        paras.append((lam_j, alpha_j))
        cDs_th.append(th)

    coeffs_th = [cA] + cDs_th
    x_hat = pywt.waverec(coeffs_th, wavelet)[:N]
    return x_hat, sigma_hat, lam0, paras


def denoise_new_lam_only(y, alpha=2.0, alpha_glob_sure=False, wavelet='db8',
                         level=None, fallback_lam=None):
    """Prop.-LA: per-level thresholds, fixed (or globally selected) alpha.

    Every detail subband chooses its own lambda by minimising its SURE while
    alpha stays common, which isolates the effect of level-adaptive
    thresholding from that of a level-adaptive decay parameter.  Returns
    ``(x_hat, sigma_hat, lam0, per_level_lams)``.
    """
    N = len(y)
    if level is None:
        level = int(np.floor(np.log2(N))) - 2
        level = max(1, min(level, 8))
        level = min(level, pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len))

    coeffs_orig = pywt.wavedec(y, wavelet, level=level)
    cA = coeffs_orig[0]
    cDs = coeffs_orig[1:]
    sigma_hat = estimate_sigma(cDs[-1])
    lam0 = visu_shrink_threshold(sigma_hat, N)

    if alpha_glob_sure:
        a = select_alpha_sure_single(np.concatenate([c.flatten() for c in cDs]),
                                     sigma_hat, lam0)
        alpha = 2.0 if a is None else a

    lams = []
    cDs_th = []
    for cD in cDs:
        lam_opt = select_lam_single(cD, sigma_hat, lam0, alpha)
        lam_j = lam0 if lam_opt is None else lam_opt
        lams.append(lam_j)
        cDs_th.append(new_threshold_safe(cD, lam_j, alpha))

    coeffs_th = [cA] + cDs_th
    x_hat = pywt.waverec(coeffs_th, wavelet)[:N]
    return x_hat, sigma_hat, lam0, np.array(lams)
