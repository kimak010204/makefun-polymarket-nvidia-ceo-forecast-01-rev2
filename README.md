# Polymarket NVIDIA CEO price forecaster — revision 2

Entry for make.fun bounty `polymarket-nvidia-ceo-forecast-01`: forecast the
Yes price of every trading market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after each checkpoint, as five
quantiles (5%, 25%, 50%, 75%, 95%), scored by mean pinball loss.

## Why revision 2

Revision 1 (persistence median + empirical widths) scored **0.285119** on
the six practice cases — and the dev leaderboard reproduced that number
exactly (rank 1646 of 2289, loss 0.285). So the leaderboard *is* the
practice set: the defect was in the model, not the data. Our median error
(0.7675) was far above the leaders' (0.60–0.68). A persistence median
throws away the two predictable signals in this event, so revision 2
adds them (details in `main.py` and `BUILD_LOG.md`):

- **Board-sum reversion.** The 24 Yes prices are mutually exclusive, so
  their sum mean-reverts toward 1. Each price is pulled toward its
  sum-normalised level, fast (most of it within the first hour).
- **Asymmetric bounce.** A market that spiked *up* in the last hour
  usually gives the spike back (~50%); down moves do not bounce back —
  they are information.
- **Pooled cross-market widths.** A single market's own 8-hour history
  has often never seen a collapse; the pooled distribution across all
  markets has, including the fat downside tail.
- **Regime gate.** When the board sum is far from 1 (market launch:
  every candidate near 0.45, sum ~2.4) the model falls back to
  revision 1 exactly, so a rotation cannot blow up the forecast.

## Method (one paragraph)

`main.py` (Python 3, standard library only, no network, deterministic)
reads each case in `/input` (`$MAKEFUN_INPUT_DIR` locally) and writes one
`<id>.out` per case to `/output` (`$MAKEFUN_OUTPUT_DIR` locally). Median =
last price + clipped drift (sum reversion + up-spike give-back). Width =
empirical h-step quantiles (last 8 h, recentered), blended 85/15 with the
house baseline normal quantiles and 70/30 with pooled cross-market
quantiles, scaled by conditional vol (recent 2 h vs lookback 10-min vol,
^0.5, clip [0.7, 1.6]) and by a board-dependent multiplier (0.7 base,
+1.5·|S−1|, cap 1.6; inner quantiles ×0.7, outer ×1.2). Short history
(<60 prices) falls back to the house baseline widths.

## Validation (local backtest, official pinball loss)

Actual prices fetched from Polymarket's public price history
(`clob.polymarket.com/prices-history`, fidelity=1). The harness
reproduces the house baseline's published practice loss exactly
(0.363589).

| model | practice (6 cases) | fresh Nov 12–13 (4 cases) | chaotic Nov 10–11 (11 cases) |
|---|---|---|---|
| house baseline | 0.363589 | — | — |
| revision 1 (shipped) | 0.285119 | 0.2463 | 2.7161 |
| **this program (rev 2)** | **0.259477** | **0.2195** | **2.6853** |

Per practice case, rev 2 vs rev 1: p00 0.420 vs 0.491, p01 0.274 vs 0.312,
p02 0.363 vs 0.372, p03 0.291 vs 0.282 (slightly worse), p04 0.164 vs
0.199, p05 0.104 vs 0.121. It beats revision 1 on 4 of 6 practice cases,
ties 1, and is slightly worse on 1 — and it also beats revision 1 on
out-of-sample checkpoints in both calm and chaotic regimes.

Runtime: ~0.22 s per case locally (limit 8 s). Two runs on the same
cases write byte-identical output. Output files 192–260 KB (limit 1 MiB).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
