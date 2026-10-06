"""Classical wavelet thresholding machinery and evaluation metrics.

Contents
--------
* the four standard test signals (Blocks, Bumps, Doppler, HeaviSine);
* the hard, soft, firm and exponentially decaying ("new") threshold rules;
* the MAD noise estimator and the universal (VisuShrink) threshold;
* the SURE machinery used to select the decay parameter;
* ``wavelet_denoise``  -- the common decompose / shrink / reconstruct pipeline;
* ``compute_metrics``  -- RMSE and output SNR.

The notation follows the manuscript "SURE-guided level-adaptive wavelet
denoising with exponentially decaying shrinkage".  No plotting code is
included, so this module depends only on NumPy and PyWavelets.
"""

import numpy as np
import pywt


def make_blocks(N):
    """``Blocks``: piecewise constant, tests edge preservation."""
    t = np.arange(1, N + 1) / N
    heights = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]
    positions = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81]
    x = np.zeros(N)
    for i, (h, p) in enumerate(zip(heights, positions)):
        idx = int(p * N)
        x[idx:] = h
    return x


def make_bumps(N):
    """``Bumps``: spikes of different widths."""
    t = np.arange(1, N + 1) / N
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array([0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005])
    x = np.zeros(N)
    for j in range(len(pos)):
        x += hgt[j] / (1 + np.abs((t - pos[j]) / wth[j])) ** 4
    return x


def make_doppler(N):
    """``Doppler``: time-varying frequency, tests non-stationary behaviour."""
    t = np.arange(1, N + 1) / N
    return np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * 1.05 / (t + 0.05))


def make_heavi_sine(N):
    """``HeaviSine``: a sinusoid plus two jumps."""
    t = np.arange(1, N + 1) / N
    return 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)


def add_noise(signal, sigma):
    """Add white Gaussian noise (legacy helper, uses the global RNG)."""
    noise = sigma * np.random.randn(len(signal))
    return signal + noise, noise


def hard_threshold(w, lam):
    """Hard rule: eta(w) = w * 1(|w| > lam)."""
    return w * (np.abs(w) > lam)


def soft_threshold(w, lam):
    """Soft rule: eta(w) = sign(w) * max(|w| - lam, 0)."""
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)


def firm_threshold(w, lam1, lam2=3.0):
    """Firm shrinkage (Gao & Bruce, 1997); linear transition on [lam1, lam2]."""
    lam2 = max(lam2, lam1 * 2)  # keep lam2 > lam1
    abs_w = np.abs(w)
    result = np.zeros_like(w)
    mask_large = abs_w > lam2                      # keep the coefficient
    result[mask_large] = w[mask_large]
    mask_mid = (abs_w > lam1) & (abs_w <= lam2)    # linear transition
    result[mask_mid] = np.sign(w[mask_mid]) * lam2 * (abs_w[mask_mid] - lam1) / (lam2 - lam1)
    # |w| <= lam1: zeroed, which is the initial value
    return result


def new_threshold(w, lam, alpha):
    """Exponentially decaying shrinkage:

        eta(w) = sign(w) * max(|w| - lam * exp(-alpha * (|w| - lam)), 0)

    Properties: continuous at +-lam; asymptotically unbiased (eta(w)/w -> 1 as
    |w| -> inf); ``alpha`` sets how fast the soft-hard transition happens.
    """
    abs_w = np.abs(w)
    t = abs_w - lam                       # t >= 0 in the active branch
    decay = lam * np.exp(-alpha * t)      # exponential decay factor
    shrink = np.maximum(abs_w - decay, 0)
    return np.sign(w) * shrink


def estimate_sigma(coeffs_detail_level1):
    """MAD noise estimate, sigma = median(|w_1|) / 0.6745.

    ``coeffs_detail_level1`` holds the finest-scale detail coefficients.
    """
    return np.median(np.abs(coeffs_detail_level1)) / 0.6745


def visu_shrink_threshold(sigma, N):
    """Universal VisuShrink threshold, lambda = sigma * sqrt(2 log N)."""
    return sigma * np.sqrt(2 * np.log(N))


def sure_value(alpha, coeffs, lam):
    """SURE(alpha) of the decaying rule (Eq. (6) of the manuscript).

        SURE(a) = n + sum_{|w_i| > lam} [ (w_i^2 - 2)
                  - 2 lam (|w_i| + a) e^{-a(|w_i| - lam)}
                  + lam^2 e^{-2a(|w_i| - lam)} ]
    """
    n = len(coeffs)
    S = n
    for w in coeffs:
        abs_w = abs(w)
        if abs_w > lam:
            d = abs_w - lam                    # |w| - lambda
            e_a = np.exp(-alpha * d)           # e^{-alpha(|w|-lambda)}
            e_2a = np.exp(-2 * alpha * d)      # e^{-2 alpha(|w|-lambda)}
            S += (w**2 - 2) - 2 * lam * (abs_w + alpha) * e_a + lam**2 * e_2a
    return S


def sure_derivative(alpha, coeffs, lam):
    """Central finite-difference derivative of ``sure_value``."""
    eps = 1e-6
    return (sure_value(alpha + eps, coeffs, lam) - sure_value(alpha - eps, coeffs, lam)) / (2 * eps)


def select_alpha_sure(coeffs_list, lam, lo=0.01, hi=500.0, max_iter=50):
    """Bisection search for the alpha minimising SURE.

    Parameters
    ----------
    coeffs_list : list of ndarray
        Detail coefficients of every level.
    lam : float
        Universal threshold.
    lo, hi : float
        Bracketing interval of alpha.
    max_iter : int
        Number of bisection steps.

    Returns
    -------
    (alpha_opt, sure_min)
    """
    all_coeffs = np.concatenate([c.flatten() for c in coeffs_list])
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        d = sure_derivative(mid, all_coeffs, lam)
        if d < 0:            # SURE still decreasing -> minimum to the right
            lo = mid
        else:                # SURE increasing -> minimum to the left
            hi = mid
    alpha_opt = (lo + hi) / 2
    sure_min = sure_value(alpha_opt, all_coeffs, lam)
    return alpha_opt, sure_min


def wavelet_denoise(y, wavelet='db8', level=None, threshold_func='soft', alpha=2.0):
    """Full wavelet-thresholding pipeline.

    Parameters
    ----------
    y : ndarray
        Noisy observation.
    wavelet : str
        Wavelet name, e.g. ``'db8'``.
    level : int or None
        Decomposition depth; when None it follows
        ``J = max(1, min(floor(log2 N) - 2, 8))`` further capped by
        ``pywt.dwt_max_level`` (the setting used in the manuscript).
    threshold_func : {'hard', 'soft', 'firm', 'new'}
        Shrinkage rule.
    alpha : float or 'sure'
        Decay parameter of the 'new' rule; 'sure' selects it by SURE.

    Returns
    -------
    x_hat, coeffs_orig, coeffs_th, sigma_hat, lam, alpha_used
    """
    N = len(y)

    if level is None:
        level = int(np.floor(np.log2(N))) - 2
        level = max(1, min(level, 8))   # at least 1 level, at most 8
        # additionally bounded by the deepest level pywt supports, so that no
        # coefficient is contaminated by the boundary extension
        level = min(level, pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len))

    coeffs_orig = pywt.wavedec(y, wavelet, level=level)
    cA = coeffs_orig[0]                # approximation coefficients
    cDs = coeffs_orig[1:]              # detail coefficients, fine to coarse

    sigma_hat = estimate_sigma(cDs[-1])
    lam = visu_shrink_threshold(sigma_hat, N)

    alpha_used = alpha
    if alpha == 'sure':
        alpha_used, _ = select_alpha_sure(cDs, lam)

    cDs_th = []
    for cD in cDs:
        if threshold_func == 'hard':
            cDs_th.append(hard_threshold(cD, lam))
        elif threshold_func == 'soft':
            cDs_th.append(soft_threshold(cD, lam))
        elif threshold_func == 'firm':
            cDs_th.append(firm_threshold(cD, lam, lam * 2))
        elif threshold_func == 'new':
            cDs_th.append(new_threshold(cD, lam, alpha_used))
        else:
            raise ValueError(f"Unknown threshold_func: {threshold_func}")

    coeffs_th = [cA] + cDs_th
    x_hat = pywt.waverec(coeffs_th, wavelet)
    x_hat = x_hat[:N]                  # waverec may return a longer array

    return x_hat, coeffs_orig, coeffs_th, sigma_hat, lam, alpha_used


def compute_metrics(x_true, x_hat):
    """Return (MSE, SNR in dB), with SNR = 10 log10(Var(x_true) / MSE)."""
    mse = np.mean((x_true - x_hat) ** 2)
    var_signal = np.var(x_true)
    if mse > 0:
        snr = 10 * np.log10(var_signal / mse)
    else:
        snr = np.inf
    return mse, snr
