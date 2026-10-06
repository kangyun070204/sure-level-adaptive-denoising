# Reproduction code — SURE-guided level-adaptive wavelet denoising

This repository contains the code that produced every number reported in

> **SURE-guided level-adaptive wavelet denoising with exponentially decaying shrinkage**
> Yun Kang, Mingyu Li
> submitted to *Current Science*

Running one command reproduces Tables 1–4 of that manuscript (and
Supplementary Tables S1–S4) from fixed random seeds.

## What the method does

Each detail subband of a `db8` wavelet decomposition picks **its own**
threshold by minimising a closed-form Stein unbiased risk estimate (SURE) of
the mean-squared error, instead of sharing one global threshold. Two safeguards
— a fallback to the global threshold on subbands judged noise dominated, and a
bounded threshold search range — are switched on in the robust variant
(Prop.-LA-R).

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Only NumPy, SciPy and PyWavelets are needed for Parts 1 and 2. Part 3
additionally uses `wfdb` to fetch two records of the MIT-BIH Arrhythmia
Database from PhysioNet; without it — or without network access — Part 3 is
skipped automatically.

## Reproduce the tables

```bash
python reproduce_tables.py
```

This writes `results/repeat_stats.json` and `results/repeat_stats.txt`
(human-readable report plus LaTeX-ready rows). Each setting is repeated with 50
independent noise realisations, and all nine methods see exactly the same
realisation, so every comparison is paired.

Useful options:

| flag | effect |
|---|---|
| `--reps N` | use N realisations instead of 50 (a few seconds with `--reps 3`) |
| `--outdir DIR` | write output elsewhere |
| `--no-ecg` | skip Part 3 (no download, no `wfdb` needed) |
| `--nocap` | use the plain `log2` decomposition depth instead of the capped depth reported in the paper |

Determinism: for setting index `i` and trial `t` the noise comes from
`numpy.random.default_rng([20240919, i, t])`, so the results do not depend on
the order in which settings are executed.

**Runtime** about 8 minutes end to end on a laptop (dominated by the
reference implementation of BiShrink).

## Mapping to the manuscript

| output | manuscript |
|---|---|
| Part 1, synthetic signals | Table 1 / Table S1 |
| Part 2, ringing on the step signal | Table 2 / Table S2 |
| Part 3, MIT-BIH records 100 and 101 | Tables 3 and 4 / Tables S3 and S4 |

`results/` in this repository holds the exact output obtained on the authors'
machine; it can be diffed against your own run to confirm reproducibility.

## Decomposition depth

The reported setting is

```python
J = max(1, min(floor(log2(N)) - 2, 8))
J = min(J, pywt.dwt_max_level(N, pywt.Wavelet('db8').dec_len))
```

so `J = 7` for the `N = 2048` test signals. The cap keeps every coefficient
free of the boundary extension effects that arise when a decomposition is
carried deeper than the filter supports. `reproduce_tables.py --nocap` restores
the uncapped depth used in an earlier draft.

## Layout

| file | contents |
|---|---|
| `reproduce_tables.py` | experiment driver: repeats, metrics, paired tests, report |
| `core.py` | test signals, threshold rules, MAD estimator, SURE for the decay parameter, metrics |
| `proposed.py` | the proposed level-adaptive rule (Prop.-LA, Prop.-LA-R) |
| `results/` | reference output (`repeat_stats.json`, `repeat_stats.txt`) |

Baselines implemented in `reproduce_tables.py`: hard, soft and firm
thresholding with the universal threshold, BayesShrink
(Chang, Yu & Vetterli, 2000) and BiShrink (Sendur & Selesnick, 2002).

## Data

The MIT-BIH Arrhythmia Database is publicly available from PhysioNet
(https://physionet.org/content/mitdb/1.0.0/) and is downloaded on demand by
`wfdb`. Records `100` and `101` are used, taking the 10 s segment that starts
5 s into the recording, mean-removed and amplitude-normalised to unit peak
value. No other data are required; the synthetic signals are generated.

## Verified with

Python 3 with NumPy 1.24.4, SciPy 1.10.1 and PyWavelets 1.4.1; the reference
output reproduces bit for bit on that combination. Other versions may give
differences in the last decimal place.

## Licence and citation

MIT; see `LICENSE`. If you reuse the code, please cite both the repository
(`CITATION.cff`) and the manuscript above.
