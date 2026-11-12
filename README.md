# Roblox Up-and-Coming Top-10 Forecaster

Entry for the make.fun bounty **Predict Roblox's Up-and-Coming top 10**
(`roblox-up-and-coming-forecast-01`). A deterministic, offline, standard-library-only
Node 22 program that reads hourly snapshots of Roblox's Up-and-Coming chart
and writes one forecast for the 72 hours after entries close.

## How it works

Each game's chart rank is treated as a latent value that drifts and diffuses.
A seeded Monte Carlo (6,000 paths, fixed seed) simulates every game forward
over the scored window:

- **Drift and volatility** are estimated per game from its rank history
  (recency-weighted, so recent reads matter more).
- **Rank diffusion** — each hour a game's rank moves by its drift plus
  Gaussian noise scaled by its volatility.
- **Leaving the chart** — a per-hour exit hazard, estimated from how often
  games disappear from the chart, rising with the game's age (the chart only
  lists games published in the last 28 days).
- **New entrants** — a small unnamed reserve at every place covers games the
  history never names (the forecast leaves that share unnamed, and the scorer
  spreads it over the games on each read's chart).

At each simulated hour the games are ordered by their simulated rank and the
top 10 recorded. A game's probability at a place is the fraction of
(path, hour) pairs where it holds that place, averaged over the whole window —
which is what the log score rewards, since one forecast is scored against
every hourly read of the window.

`top10` is the 10 games most likely to be in the top 10, ordered by their
expected rank.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads `task.json` and `history.ndjson` from the input directory and writes
`forecast.json` to the output directory. No network, no model calls, no
dependencies. Two runs on the same input produce byte-identical output
(seeded PRNG; no wall-clock or unseeded randomness).

Runtime is ~4 seconds on one CPU, well inside the 300 s limit.

## Backtesting (kit development data)

Rolling-origin backtests on the kit's 34-read practice history, scored with
the official log-score rule:

| history reads | held-out reads | horizon | log score | house baseline |
|---:|---:|---:|---:|---:|
| 24 | 10 | 62 h | -48.24 | -56.91 |
| 28 | 6 | 36 h | -32.25 | -37.71 |
| 30 | 4 | 24 h | -34.15 | -30.01 |

The model beats the house baseline at long horizons (where most of the
72-hour window's signal lives) and is competitive at short horizons. Scores
vary a lot by window because the chart churns heavily; the numbers above are
the honest backtest results, not a promise about the final window.

## Files

- `main.ts` — the forecaster (the entrypoint).
- `package.json` — marks the package as an ES module.
- `LICENSE` — MIT.
- `.makefun/owner.json` — claim file naming the wallet that signs this entry.
