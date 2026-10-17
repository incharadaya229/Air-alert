"""Step 4b: replay readings hour by hour like a live stream, with timing for the IBM Z story."""
import time

import numpy as np

from src.decide import ADVICE, tier


def pick_window(test, hours=48):
    """Pick the stretch of the test set that contains the most real spikes (best for a demo)."""
    t = test.sort_values("timestamp").reset_index(drop=True)
    score = t["label"].rolling(hours, min_periods=hours).sum()
    end = int(score.idxmax()) if score.notna().any() else min(hours, len(t)) - 1
    return t.iloc[max(end - hours + 1, 0): end + 1]


def replay(model, window, features, threshold=200, verbose=True):
    X = window[features].to_numpy(float)
    times = []
    rows = []
    for i in range(len(window)):
        t0 = time.perf_counter()
        p = float(model.predict_proba(X[i:i + 1])[0, 1])
        times.append((time.perf_counter() - t0) * 1000)
        r = window.iloc[i]
        tr = tier(p, r["aqi"], r["rate_3h"], threshold)
        rows.append((r["timestamp"], r["aqi"], p, tr))
        if verbose:
            adv = ADVICE[tr].get("Schools", "") if tr != "Low" else ""
            print(f"{r['timestamp']}  AQI={r['aqi']:6.1f}  spike_prob={p:4.2f}  [{tr:5s}] {adv}")
    t0 = time.perf_counter()
    model.predict_proba(X)
    batch = time.perf_counter() - t0
    stats = {
        "events": len(times),
        "avg_ms_per_event": round(float(np.mean(times)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "batch_events_per_sec": int(len(X) / max(batch, 1e-9)),
    }
    return rows, stats
