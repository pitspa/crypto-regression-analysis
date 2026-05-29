# Crypto Rolling Regression Analysis

Rolling **alpha** and **beta** of the largest cryptocurrencies measured against **Bitcoin**, estimated over multiple trailing windows and refreshed daily. The project pulls real market data from CoinGecko, computes the regressions in Rust, and publishes an interactive dashboard to GitHub Pages.

**Live site:** https://pitspa.github.io/crypto-regression-analysis

Bitcoin is treated as the "market" factor. For every other top-cap coin the project estimates two specifications on a rolling basis:

1. A standard single-factor (CAPM-style) market model — one beta to Bitcoin.
2. A **segmented** model with separate betas for Bitcoin's **up** days and **down** days, plus a formal test of whether the two betas differ.

\---

## Architecture at a glance

```
CoinGecko API
     │  (Python: fetch\\\_crypto\\\_data.py)
     ▼
backend/data/                      ← per-coin CSVs, combined\\\_data.csv, metadata.json
     │  (Rust: rust\\\_analyzer, driven by rust\\\_analyzer\\\_wrapper.py)
     ▼
regression\\\_results\\\_window\\\_{w}.json
segmented\\\_regression\\\_results\\\_window\\\_{w}.json
     │  (GitHub Actions copies JSON + CSV into frontend/data/)
     ▼
frontend/  → GitHub Pages (Chart.js dashboard)
```

The whole chain runs unattended once per day via GitHub Actions (`02:00 UTC`), and also on push to `main` or manual dispatch.

\---

## Repository layout

```
backend/
  data\\\_fetcher/
    fetch\\\_crypto\\\_data.py        # CoinGecko pull → CSVs + metadata
    rust\\\_analyzer\\\_wrapper.py    # builds \\\& runs the Rust analyzer, tracks status
    pipeline\\\_status.py          # step-by-step status written to pipeline\\\_status.json
    requirements.txt            # requests, pandas, numpy
    test\\\_coingecko\\\_api.py       # connectivity / sanity checks
    test\\\_api\\\_key\\\_usage.py
  rust\\\_analyzer/
    src/main.rs                 # rolling OLS + segmented OLS, CIs, asymmetry test
    Cargo.toml                  # nalgebra, csv, serde, serde\\\_json, chrono
  data/                         # (generated at runtime; not committed)
frontend/
  index.html                   # Chart.js dashboard
  status.html                  # pipeline status viewer
.github/workflows/
  deploy.yml                    # fetch → analyze → deploy → cleanup
```

\---

## 1\. The data pipeline

### 1.1 Source and universe

Data comes exclusively from the public **CoinGecko REST API** (`https://api.coingecko.com/api/v3`) — there is no synthetic or back-filled data anywhere in the pipeline.

* The universe is the **top 20 coins by market capitalisation**, taken from `/coins/markets` (`vs\\\_currency=usd`, `order=market\\\_cap\\\_desc`). The ranking is re-pulled on every run, so the set of coins can change over time.
* **Bitcoin is the reference asset.** The run aborts if Bitcoin is missing from the fetched set, or if fewer than two coins are retrieved successfully.

### 1.2 Historical prices

For each coin, daily history is fetched from `/coins/{id}/market\\\_chart` (`interval=daily`):

* **365 days** when a CoinGecko API key is supplied (sent via the `x-cg-demo-api-key` header, i.e. the free/"demo" tier), or
* **90 days** when no key is present (the run still works, just with a shorter sample).

The key is read from the `COINGECKO\\\_API\\\_KEY` environment variable / GitHub secret. Requests are spaced out (1.0 s with a key, 2.5 s without) to respect rate limits, and HTTP 429s are caught and logged rather than silently dropped. Because the regressions use trailing windows of up to 180 days (§2), the 90-day no-key mode cannot fill the longest windows — a key is needed for the full set of window sizes to be populated.

### 1.3 Cleaning and returns

Per coin, the raw price series is cleaned as follows:

* Timestamps are converted to dates; **duplicate dates are collapsed to one observation per day** (the last price of the day is kept).
* Non-positive prices are discarded.
* A coin is only kept if it has **more than 10 valid days** of data.

Daily **log returns** are then computed:

$$
r\_t = \\ln\\left(\\frac{P\_t}{P\_{t-1}}\\right)
$$

(The first observation is `NaN` by construction and is handled downstream.)

### 1.4 Output files

The fetcher writes everything to `backend/data/`:

* `{coin\\\_id}\\\_data.csv` — per-coin `date, price, log\\\_return`.
* `combined\\\_data.csv` — a **wide** table with one row per date and, for each coin, the columns `{coin\\\_id}\\\_price` and `{coin\\\_id}\\\_log\\\_return`. Coins are outer-joined on date, so gaps appear as blanks (handled as missing in the analyzer).
* `metadata.json` — fetch timestamp, the ranked coin list (id, symbol, name, market cap, rank), `reference\\\_coin: "bitcoin"`, success/failure counts, and a `data\\\_coverage` block (date range, API tier).
* `pipeline\\\_status.json` — written incrementally by `pipeline\\\_status.py`; records each step's status (`success`/`failed`/`running`), errors, and warnings, and backs the `status.html` page.

\---

## 2\. The regression models

All regressions are **rolling** and run **per coin against Bitcoin** over a set of trailing window sizes. The reference coin (Bitcoin) is skipped as a dependent variable. Estimation uses ordinary least squares solved through the normal equations with `nalgebra`:

$$
\\hat{\\boldsymbol\\theta} = (X^\\top X)^{-1} X^\\top y
$$

Each rolling estimate is dated with the **last day of its window** (the window for date $D$ spans the `window\\\_size` trading days up to and including $D$). A window is only fitted if it contains at least `window\\\_size / 2` finite, paired (coin, BTC) observations; otherwise a null result is emitted for that date so the time series stays aligned. Windows run from one week up to **180 days (six months)** — the longest the analysis produces. Although the fetcher pulls up to a year of price history, no rolling window exceeds six months.

**On inference.** Confidence intervals and the asymmetry p-value use deliberately lightweight machinery: a tabulated $t$ critical value (falling back to the normal approximation $z = 1.645$ beyond $\\mathrm{df} = 120$) and a normal-CDF p-value rather than exact $t$-distribution tails. The standard errors are the classical OLS ones — no heteroskedasticity- or autocorrelation-robust (HAC / Newey–West) correction is applied. Since daily crypto returns are both heteroskedastic and serially correlated, the reported intervals are best read as indicative rather than exact: fine for a dashboard, but worth bearing in mind for formal inference.

> \\\*\\\*A note on terminology.\\\*\\\* Both models regress the coin's \\\*raw\\\* log return on Bitcoin's \\\*raw\\\* log return — neither subtracts a risk-free rate. So these are strictly \\\*\\\*single-factor market-model\\\*\\\* regressions with Bitcoin as the market proxy, rather than textbook CAPM (which uses excess returns). In a crypto setting where Bitcoin stands in for "the market" and the risk-free rate is negligible at daily frequency, the labels are used interchangeably here, but the distinction is worth keeping in mind when interpreting alpha.

### 2.1 Standard market model (single beta)

For coin returns $y\_t$ and Bitcoin returns $x\_t$ within a window:

$$
y\_t = \\alpha + \\beta x\_t + \\varepsilon\_t
$$

* **$\\alpha$ (alpha)** — average return of the coin not explained by its co-movement with Bitcoin (the intercept).
* **$\\beta$ (beta)** — sensitivity to Bitcoin. $\\beta > 1$ amplifies BTC moves, $\\beta < 1$ dampens them.

Standard errors come from the OLS variance–covariance matrix $\\widehat{\\mathrm{Var}}(\\hat{\\boldsymbol\\theta}) = s^2 (X^\\top X)^{-1}$, with $s^2 = \\mathrm{RSS}/(n-2)$ and $n-2$ degrees of freedom; **90% confidence intervals** are reported for both $\\alpha$ and $\\beta$.

**Window sizes:** 7 (1 week), 14 (2 weeks), 30 (1 month), 60 (2 months), 90 (3 months), 120 (4 months), and 180 (6 months) days.

### 2.2 Segmented model (up-beta vs down-beta)

This specification splits Bitcoin's beta by the **sign of the Bitcoin return**, sharing a single intercept:

$$
y\_t = \\alpha + \\beta^{+}\\max(x\_t, 0) + \\beta^{-}\\min(x\_t, 0) + \\varepsilon\_t
$$

Equivalently, the design matrix is $X = \[1; x\_t\\cdot\\mathbb{1}(x\_t>0); x\_t\\cdot\\mathbb{1}(x\_t<0)]$. This is a continuous, "kinked" regression — the two arms meet at $x\_t = 0$ — and corresponds to the **dual-beta / downside-beta** idea: an asset can track Bitcoin differently on up days than on down days.

* **$\\beta^{+}$ (upside beta)** — sensitivity on days Bitcoin rises.
* **$\\beta^{-}$ (downside beta)** — sensitivity on days Bitcoin falls.

Guards: the window must contain **at least 5 positive and 5 negative** Bitcoin-return days, or the segmented fit is skipped for that window. Degrees of freedom are $n-3$; 90% CIs are reported for $\\alpha$, $\\beta^{+}$, and $\\beta^{-}$.

**Asymmetry test.** For each window the model tests whether the two betas are genuinely different:

$$
H\_0: \\beta^{+} = \\beta^{-}
\\qquad
t = \\frac{\\hat\\beta^{+} - \\hat\\beta^{-}}{\\sqrt{\\widehat{\\mathrm{Var}}(\\hat\\beta^{+}) + \\widehat{\\mathrm{Var}}(\\hat\\beta^{-}) - 2,\\widehat{\\mathrm{Cov}}(\\hat\\beta^{+}, \\hat\\beta^{-})}}
$$

The denominator is exactly $\\widehat{\\mathrm{Var}}(\\hat\\beta^{+} - \\hat\\beta^{-})$, with all three terms taken directly from the OLS coefficient covariance matrix (verified against the source). Although written as a $t$-ratio, the statistic is compared against the **standard normal** distribution — the two-tailed p-value is $p = 2,\[,1 - \\Phi(|t|),]$, with $\\Phi$ evaluated through an Abramowitz–Stegun `erf` approximation — so in practice it is a Wald / z test. The betas are flagged **significantly different at the 90% level** when $p < 0.10$.

**Window sizes:** 60 (2 months), 90 (3 months), 120 (4 months), and 180 (6 months) days — segmented analysis is intentionally restricted to the longer windows so each sign regime has enough observations.

### 2.3 Result files

For every window the Rust analyzer writes pretty-printed JSON to `backend/data/`:

* `regression\\\_results\\\_window\\\_{w}.json` — `metadata` block (`reference\\\_coin`, `window\\\_size`, `analysis\\\_date`, `total\\\_coins`, `analysis\\\_type: "standard"`) plus a `results` array of `{ coin\\\_id, date, alpha, beta, alpha\\\_lower\\\_90, alpha\\\_upper\\\_90, beta\\\_lower\\\_90, beta\\\_upper\\\_90, window\\\_size }`.
* `segmented\\\_regression\\\_results\\\_window\\\_{w}.json` — same metadata shape (`analysis\\\_type: "segmented"`) plus per-date `{ alpha, beta\\\_positive, beta\\\_negative, …\\\_lower\\\_90 / …\\\_upper\\\_90, asymmetry\\\_test\\\_statistic, asymmetry\\\_p\\\_value, betas\\\_significantly\\\_different, window\\\_size }`.

\---

## 3\. Frontend

`frontend/index.html` is a static, dependency-light dashboard built on **Chart.js** (with the moment date adapter, the zoom plugin, and hammer.js for touch/gestures). At runtime it loads `data/metadata.json` and `data/combined\\\_data.csv`, then fetches the relevant `regression\\\_results\\\_window\\\_${w}.json` or `segmented\\\_regression\\\_results\\\_window\\\_${w}.json` on demand as the controls change. The controls are:

* **Select Cryptocurrency** — the primary coin to analyse against Bitcoin (e.g. XRP).
* **Comparison Mode** — *Single Coin*, or *Compare Two Coins* to overlay a second asset.
* **Second Cryptocurrency** — shown only in compare mode (e.g. Cardano).
* **Window Size (days)** — 7 (1 week), 14 (2 weeks), 30 (1 month), 60 (2 months), 90 (3 months), 120 (4 months), or 180 (6 months).
* **Chart Type** — *Alpha and Beta* (the standard model) or *Alpha and Segmented Betas* (the up/down-beta model). Segmented betas require a window of **at least 60 days (2 months)**, matching the analyzer; pairing them with a shorter window (7/14/30 days) is caught in the UI, which shows an inline notice prompting for a larger window instead of rendering a chart.

A **Price Comparison with Bitcoin** panel plots the selected coin(s) alongside Bitcoin, with a *Toggle Normalize* option and zoom / *Reset Zoom* controls. `frontend/status.html` separately reads `pipeline\\\_status.json` to surface the health of the latest pipeline run.

\---

## 4\. Running it

### Prerequisites

* Python 3.11+
* Rust (stable toolchain, `cargo`)
* *(Optional but recommended)* a CoinGecko API key for the full 365-day history

### Local run

```bash
# 1) Fetch data
cd backend/data\\\_fetcher
pip install -r requirements.txt
export COINGECKO\\\_API\\\_KEY=your\\\_key\\\_here   # optional; omit for 90-day mode
python fetch\\\_crypto\\\_data.py              # writes ../data/

# 2) Build + run the analyzer (the wrapper calls cargo build/run --release)
python rust\\\_analyzer\\\_wrapper.py          # writes ../data/\\\*.json

# 3) Serve the frontend
cd ../../frontend
cp ../backend/data/\\\*.json ./data/        # mirror the CI "copy to frontend" step
cp ../backend/data/combined\\\_data.csv ./data/
python -m http.server 8000               # then open http://localhost:8000
```

The Rust analyzer can also be built and run directly:

```bash
cd backend/rust\\\_analyzer
cargo run --release        # expects ../data/metadata.json and ../data/combined\\\_data.csv
```

### Automated deployment

`.github/workflows/deploy.yml` runs three sequential jobs (plus a cleanup job):

1. **fetch-data** — installs Python deps, runs the connectivity test and `fetch\\\_crypto\\\_data.py`, verifies `combined\\\_data.csv` and `metadata.json` exist, and uploads `backend/data/` as a build artifact.
2. **analyze-data** — downloads that artifact, sets up Rust (with a cargo cache), runs the analyzer through the wrapper, asserts every expected window file is present, then copies the JSON + `combined\\\_data.csv` into `frontend/data/` and uploads the Pages artifact.
3. **deploy** — publishes to GitHub Pages.
4. **cleanup** — deletes the intermediate data artifact.

Set `COINGECKO\\\_API\\\_KEY` as a repository **secret** to enable the full-history path in CI. The schedule is daily at `02:00 UTC`.

\---

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

