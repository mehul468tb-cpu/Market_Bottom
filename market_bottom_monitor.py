#!/usr/bin/env python3
"""Daily Market Bottom Monitor (standalone; does not alter other analysis pipelines).

Data: Yahoo Finance (via yfinance), FRED public CSV (no API key), optional
unfiltered stock-universe CSV for *sample* breadth. Indicators are observed
at the latest shared completed market session. Scores are experimental,
not calibrated probabilities or investment advice.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import math
import sys
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import yfinance as yf

LOG = logging.getLogger("bottom_monitor")
# ETFs/benchmarks deliberately limited to market-bottom monitoring.
TICKERS = ["SPY", "QQQ", "IWM", "RSP", "SPMO", "HYG", "JNK", "LQD",
           "XLF", "KRE", "KBE", "XLI", "XLK", "XLV", "XLP", "XLU", "XLY",
           "XLE", "XLB", "^VIX"]
FRED = {"hy_oas": "BAMLH0A0HYM2", "treasury_10y": "DGS10",
        "treasury_2y": "DGS2", "financial_conditions": "NFCI"}
GROUP_WEIGHTS = {"breadth": 25, "credit": 20, "price": 20,
                 "momentum": 15, "volatility": 10, "leadership": 10}


def get_ohlcv(tickers: list[str], start: str, end: str, batch_size: int = 40) -> dict[str, pd.DataFrame]:
    out = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, start=start, end=end, interval="1d", auto_adjust=True,
                              progress=False, group_by="ticker", threads=True,
                              multi_level_index=True)
            if raw is None or raw.empty:
                raise ValueError("Empty Yahoo response")
            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        # group_by=ticker -> (ticker, OHLCV); defend against swapped levels
                        if t in raw.columns.get_level_values(0):
                            frame = raw[t].copy()
                        elif t in raw.columns.get_level_values(1):
                            frame = raw.xs(t, axis=1, level=1).copy()
                        else:
                            continue
                    else:
                        frame = raw.copy() if len(batch) == 1 else pd.DataFrame()
                    frame.columns = [str(c).title() for c in frame.columns]
                    if "Close" not in frame.columns:
                        continue
                    frame = frame.dropna(subset=["Close"]).sort_index()
                    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
                    frame = frame.loc[~frame.index.duplicated(keep="last")]
                    if len(frame) >= 25:
                        out[t] = frame
                except Exception as exc:
                    LOG.warning("Cannot parse %s: %s", t, exc)
        except Exception as exc:
            LOG.warning("Yahoo batch %s failed: %s", batch, exc)
    return out


def get_fred(start: str) -> dict[str, pd.Series]:
    result = {}
    for name, series_id in FRED.items():
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series_id, "cosd": start})
            response = requests.get(url, timeout=25, headers={"User-Agent": "MarketBottomMonitor/1.0"})
            response.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            date_col = "observation_date" if "observation_date" in df else df.columns[0]
            value_col = series_id if series_id in df else df.columns[-1]
            dates = pd.to_datetime(df[date_col], errors="coerce")
            vals = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
            s = pd.Series(vals.to_numpy(), index=dates, name=name).dropna()
            result[name] = s[~s.index.isna()].sort_index()
        except Exception as exc:
            LOG.warning("FRED %s unavailable: %s", series_id, exc)
    return result


def valid(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def last_at(series: pd.Series | None, date: pd.Timestamp, max_age_days: int = 7):
    if series is None or series.empty:
        return None, None
    eligible = series.loc[series.index <= date].dropna()
    if eligible.empty:
        return None, None
    dt = eligible.index[-1]
    if (date - dt).days > max_age_days:
        return None, str(dt.date())
    return float(eligible.iloc[-1]), str(dt.date())


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    c = x["Close"].astype(float)
    for n in [20, 50, 200]:
        x[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = (100 - (100 / (1 + rs))).where(loss != 0, 100)
    x["macd_hist"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean())
    x["macd_hist"] = x["macd_hist"] - x["macd_hist"].ewm(span=9, adjust=False).mean()
    x["ret5"] = c.pct_change(5)
    return x


def value_at(frames: dict, ticker: str, date: pd.Timestamp, col: str = "Close"):
    f = frames.get(ticker)
    if f is None or col not in f or date not in f.index:
        return None
    v = f.at[date, col]
    return float(v) if valid(v) else None


def ratio_change(frames, top, bottom, date, periods=5):
    a, b = frames.get(top), frames.get(bottom)
    if a is None or b is None:
        return None
    pair = pd.concat([a["Close"], b["Close"]], axis=1, join="inner").dropna()
    pair = pair.loc[pair.index <= date]
    if len(pair) <= periods or pair.iloc[-1].name != date:
        return None
    r = pair.iloc[:, 0] / pair.iloc[:, 1]
    return float(r.iloc[-1] / r.iloc[-1-periods] - 1)


def sample_breadth(frames: dict, symbols: list[str], date: pd.Timestamp):
    """Sample breadth, NOT NYSE/S&P500 breadth. Excludes benchmark ETFs."""
    observed = []
    for ticker in symbols:
        f = frames.get(ticker)
        if f is None or len(f) < 202 or date not in f.index:
            continue
        close = f["Close"].loc[:date]
        if len(close) < 202:
            continue
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        # New lows only over prior 252 trading sessions; if insufficient, exclude.
        if len(close) < 253:
            continue
        prev_252 = close.iloc[-253:-1]
        observed.append({"above20": close.iloc[-1] > ma20.iloc[-1],
                         "above50": close.iloc[-1] > ma50.iloc[-1],
                         "above200": close.iloc[-1] > ma200.iloc[-1],
                         "new_low": close.iloc[-1] < prev_252.min(),
                         "new_high": close.iloc[-1] > prev_252.max(),
                         "up": close.iloc[-1] > close.iloc[-2]})
    if not observed:
        return {}
    df = pd.DataFrame(observed)
    return {"sample_size": len(df), **{f"sample_{c}_pct": round(100 * float(df[c].mean()), 2)
                                           for c in df.columns}}


def evaluate(frames: dict, fred: dict, date: pd.Timestamp, breadth: dict, min_breadth: int):
    measurements = {"date": str(date.date()), **breadth}
    # Data as-of market close; FRED is not guaranteed released by this same close.
    for t in ["SPY", "QQQ", "IWM", "RSP", "SPMO", "HYG", "JNK", "LQD", "^VIX"]:
        key = t.replace("^", "")
        measurements[f"{key}_close"] = value_at(frames, t, date)
    for series_name, s in fred.items():
        measurements[series_name], measurements[f"{series_name}_observation_date"] = last_at(s, date)
    for t in ["SPY", "QQQ", "IWM"]:
        for c in ["ma20", "ma50", "ma200", "rsi14", "macd_hist", "ret5"]:
            measurements[f"{t}_{c}"] = value_at(frames, t, date, c)
    for a, b in [("RSP", "SPY"), ("IWM", "SPY"), ("SPMO", "SPY"), ("KRE", "XLF")]:
        measurements[f"{a}_{b}_ratio_5d_change"] = ratio_change(frames, a, b, date)

    components = {}
    notes = []
    def group(name, checks):
        # Each check: bool if data exists; None -> don't silently award points.
        ready = [bool(x) for x in checks if x is not None]
        if len(ready) < (len(checks) + 1) // 2:
            components[name] = None
        else:
            components[name] = round(100 * sum(ready) / len(ready), 1)

    p = measurements
    def above(a, b):
        return a > b if valid(a) and valid(b) else None
    def below(a, b):
        return a < b if valid(a) and valid(b) else None
    def positive(a):
        return a > 0 if valid(a) else None
    group("price", [above(p.get("SPY_close"), p.get("SPY_ma20")),
                    above(p.get("SPY_close"), p.get("SPY_ma50")),
                    above(p.get("SPY_close"), p.get("SPY_ma200")),
                    above(p.get("QQQ_close"), p.get("QQQ_ma20")),
                    above(p.get("IWM_close"), p.get("IWM_ma20"))])
    group("momentum", [above(p.get("SPY_rsi14"), 50),
                       positive(p.get("SPY_macd_hist")),
                       positive(p.get("SPY_ret5")),
                       above(p.get("QQQ_rsi14"), 50)])
    # A spread level alone is not enough: rising credit stress is a negative.
    hy = fred.get("hy_oas")
    hy_historical = hy.loc[hy.index <= date].dropna() if hy is not None else pd.Series(dtype=float)
    hy_change = None
    if len(hy_historical) >= 6 and (date - hy_historical.index[-1]).days <= 7:
        hy_change = float(hy_historical.iloc[-1] - hy_historical.iloc[-6])
    p["hy_oas_change_5obs_pp"] = hy_change
    h = frames.get("HYG")
    hyg_ret = value_at(frames, "HYG", date, "ret5")
    jnk_ret = value_at(frames, "JNK", date, "ret5")
    group("credit", [below(hy_change, 0), positive(hyg_ret), positive(jnk_ret)])
    # Volatility: falling VIX and a low current reading support stabilization;
    # elevated VIX itself indicates stress, never automatically a bottom.
    v = frames.get("^VIX")
    vix_delta = None
    if v is not None and date in v.index and len(v.loc[:date]) >= 6:
        close = v.loc[:date, "Close"]
        vix_delta = float(close.iloc[-1] - close.iloc[-6])
    p["VIX_5d_change"] = vix_delta
    group("volatility", [below(vix_delta, 0), below(p.get("VIX_close"), 25)])
    group("leadership", [positive(p.get("RSP_SPY_ratio_5d_change")),
                         positive(p.get("IWM_SPY_ratio_5d_change")),
                         positive(p.get("SPMO_SPY_ratio_5d_change"))])
    if breadth.get("sample_size", 0) >= min_breadth:
        group("breadth", [above(p.get("sample_above20_pct"), 50),
                          above(p.get("sample_above50_pct"), 40),
                          below(p.get("sample_new_low_pct"), 5),
                          above(p.get("sample_up_pct"), 50)])
        notes.append("Breadth is from a user-defined SAMPLE universe, not broad-exchange official data.")
    else:
        components["breadth"] = None
        notes.append(f"Insufficient breadth coverage: need {min_breadth} eligible stocks, have {breadth.get('sample_size', 0)}.")
    covered = sum(GROUP_WEIGHTS[k] for k in GROUP_WEIGHTS if components[k] is not None)
    score = round(sum(GROUP_WEIGHTS[k] * components[k] / 100 for k in GROUP_WEIGHTS
                      if components[k] is not None) / covered * 100, 1) if covered else None
    # Unscaled score only when at least 75% of intended groups are actually populated.
    if covered < 75:
        classification = "INSUFFICIENT DATA"
    elif score <= 25:
        classification = "DOWNTREND / HIGH RISK"
    elif score <= 45:
        classification = "STRESS / POSSIBLE CAPITULATION"
    elif score <= 65:
        classification = "BOTTOM FORMATION CANDIDATE"
    elif score <= 80:
        classification = "RECOVERY CONFIRMATION CANDIDATE"
    else:
        classification = "BROAD RECOVERY SIGNAL"
    if covered < 100:
        notes.append("Score rescales covered groups; missing groups reduce confidence. No predictive calibration.")
    notes.append("FRED observation dates may precede public release; avoid treating this as a point-in-time backtest.")
    return {"date": str(date.date()), "score": score, "coverage_pct": covered,
            "classification": classification, "components": components,
            "metrics": measurements, "notes": notes}


def symbols_from_csv(path: Path) -> list[str]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    col = next((c for c in data.columns if str(c).strip().lower() in ("ticker", "symbol")), data.columns[0])
    symbols = [str(x).strip().upper().replace(".", "-") for x in data[col].dropna()]
    return sorted({x for x in symbols if x and x not in TICKERS and len(x) <= 12})


def write_outputs(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    date = result["date"]
    with (output_dir / f"market_bottom_{date}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    flat = {"date": date, "score": result["score"], "coverage_pct": result["coverage_pct"],
            "classification": result["classification"],
            **{f"component_{k}": v for k, v in result["components"].items()},
            **result["metrics"]}
    history_path = output_dir / "market_bottom_history.csv"
    if history_path.exists():
        existing = pd.read_csv(history_path)
        history = pd.concat([existing, pd.DataFrame([flat])], ignore_index=True)
        history = history.drop_duplicates(subset="date", keep="last").sort_values("date")
    else:
        history = pd.DataFrame([flat])
    history.to_csv(history_path, index=False)
    hist = history.loc[history["date"] < date].sort_values("date")
    previous = hist.iloc[-1] if not hist.empty else None
    def fmt(value, suffix=""):
        return f"{value:.2f}{suffix}" if valid(value) else "Unavailable"
    lines = [f"# Market Bottom Monitor — {date}", "",
             f"**Classification:** {result['classification']}",
             f"**Experimental score:** {fmt(result['score'], '/100')} | **Coverage:** {result['coverage_pct']}%", ""]
    if previous is not None:
        lines.append(f"Previous run ({previous['date']}): {fmt(pd.to_numeric(previous.get('score'), errors='coerce'), '/100')} ")
        lines.append("")
    lines += ["## Components", "", "| Component | Score / 100 | Weight |",
              "|---|---:|---:|"]
    for k, v in result["components"].items():
        lines.append(f"| {k.title()} | {fmt(v)} | {GROUP_WEIGHTS[k]}% |")
    lines += ["", "## Key observations", ""]
    keys = ["SPY_close", "SPY_ma20", "SPY_ma50", "SPY_ma200", "SPY_rsi14",
            "VIX_close", "VIX_5d_change", "hy_oas", "hy_oas_change_5obs_pp",
            "RSP_SPY_ratio_5d_change", "IWM_SPY_ratio_5d_change",
            "sample_size", "sample_above20_pct", "sample_above50_pct", "sample_new_low_pct"]
    for k in keys:
        value = result["metrics"].get(k)
        lines.append(f"- {k}: {fmt(value) if isinstance(value, (int, float)) else value or 'Unavailable'}")
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in result["notes"]]
    lines += ["- Score thresholds and weights are preliminary heuristics and require historical validation.",
              "- No independent NYSE/Nasdaq breadth, 52-week lows, advance/decline volume, earnings revisions or VIX futures term structure are fetched.",
              "- A scored recovery is not proof of a final market low; no allocation instruction is generated."]
    (output_dir / f"market_bottom_{date}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame([flat]).to_csv(output_dir / f"market_bottom_{date}.csv", index=False)
    return history_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", type=Path, help="Optional CSV with Ticker/Symbol column for sample breadth")
    ap.add_argument("--output", type=Path, default=Path("market_bottom_output"))
    ap.add_argument("--min-breadth", type=int, default=100,
                    help="Minimum eligible ticker count for sample breadth (default 100)")
    ap.add_argument("--years", type=int, default=2, help="Historical download lookback (default 2)")
    ap.add_argument("--asof", help="Date YYYY-MM-DD, default latest completed common SPY session")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    if args.min_breadth < 1 or args.years < 2:
        ap.error("--min-breadth must be >=1 and --years >=2")
    # End is exclusive. Avoid intraday partial current-session bars: cut off at
    # yesterday unless NY time >= 16:30; holidays handled by Yahoo index.
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    max_date = now.date() if (now.hour, now.minute) >= (16, 30) else now.date() - timedelta(days=1)
    if args.asof:
        requested = datetime.strptime(args.asof, "%Y-%m-%d").date()
        if requested > max_date:
            ap.error("--asof cannot be beyond the latest safely completed US trading session")
        max_date = requested
    start = (max_date - timedelta(days=365 * args.years + 40)).isoformat()
    end = (max_date + timedelta(days=1)).isoformat()
    symbols = symbols_from_csv(args.universe) if args.universe else []
    LOG.info("Downloading core indicators (%d ETFs/indexes)", len(TICKERS))
    frames = get_ohlcv(TICKERS, start, end)
    if "SPY" not in frames:
        LOG.error("SPY data unavailable. No report generated (avoids misleading snapshots).")
        return 2
    market_date = frames["SPY"].index[-1]
    # Only same-session comparison, not stale prices filled forward.
    frames = {k: indicators(v) for k, v in frames.items()}
    breadth = {}
    if symbols:
        LOG.info("Downloading user sample universe (%d tickers)", len(symbols))
        sample_frames = get_ohlcv(symbols, start, end)
        breadth = sample_breadth(sample_frames, symbols, market_date)
    fred = get_fred(start)
    result = evaluate(frames, fred, market_date, breadth, args.min_breadth)
    history_path = write_outputs(result, args.output)
    print(f"Date: {result['date']} | {result['classification']} | Score: {result['score']} | Coverage: {result['coverage_pct']}%")
    print(f"Outputs: {args.output.resolve()} | History: {history_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
