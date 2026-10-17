"""Step 2: merge AQI + weather, flag bad sensor readings, build features and the spike label."""
from io import StringIO

import pandas as pd

HORIZON = 2        # hours ahead we warn about
THRESHOLD = 200    # AQI above this = "spike" (CPCB 'Poor' starts at 201 - verify on CPCB site)
WEATHER_COLS = ["temp", "humidity", "wind"]


def load_weather(path):
    """Read an Open-Meteo CSV (skips the location lines before the 'time' header)."""
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("time,"))
    w = pd.read_csv(StringIO("\n".join(lines[start:])))
    w["timestamp"] = pd.to_datetime(w["time"])
    rename = {}
    for c in w.columns:
        lc = c.lower()
        if lc.startswith("temperature"):
            rename[c] = "temp"
        elif lc.startswith("relative_humidity"):
            rename[c] = "humidity"
        elif lc.startswith("wind_speed"):
            rename[c] = "wind"
    w = w.rename(columns=rename)
    return w[["timestamp"] + [c for c in WEATHER_COLS if c in w.columns]]


def build_table(aqi, weather=None):
    """Continuous hourly table (gaps become NaN) with optional weather columns."""
    full = pd.date_range(aqi["timestamp"].min(), aqi["timestamp"].max(), freq="h")
    df = aqi.set_index("timestamp")["value"].reindex(full).rename("aqi").to_frame()
    df.index.name = "timestamp"
    df = df.reset_index()
    if weather is not None:
        df = df.merge(weather, on="timestamp", how="left")
    return df


def clean(df, stuck_hours=6, max_valid=500):
    """Sensor trust check: blank out impossible values and 'stuck' flat-lined readings."""
    v = df["aqi"]
    bad_range = (v < 0) | (v > max_valid)
    run_id = (v != v.shift()).cumsum()
    run_len = v.groupby(run_id).transform("size")
    stuck = v.notna() & (run_len >= stuck_hours)
    out = df.copy()
    out["flag_stuck"] = stuck
    out["flag_range"] = bad_range
    out.loc[stuck | bad_range, "aqi"] = float("nan")
    return out


def add_features(df, horizon=HORIZON, threshold=THRESHOLD):
    d = df.copy()
    a = d["aqi"]
    for k in (1, 2, 3, 6, 24):
        d[f"aqi_lag{k}"] = a.shift(k)
    d["roll_mean_3"] = a.rolling(3, min_periods=2).mean()
    d["roll_mean_6"] = a.rolling(6, min_periods=3).mean()
    d["roll_max_6"] = a.rolling(6, min_periods=3).max()
    d["rate_1h"] = a - d["aqi_lag1"]
    d["rate_3h"] = a - d["aqi_lag3"]
    d["hour"] = d["timestamp"].dt.hour
    d["month"] = d["timestamp"].dt.month
    d["dow"] = d["timestamp"].dt.dayofweek
    # label: does AQI go above the threshold at any point in the next `horizon` hours?
    future_max = a.rolling(horizon, min_periods=horizon).max().shift(-horizon)
    d["label"] = (future_max > threshold).astype(float)
    d.loc[future_max.isna(), "label"] = float("nan")
    return d


def feature_names(d):
    base = ["aqi", "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag6", "aqi_lag24",
            "roll_mean_3", "roll_mean_6", "roll_max_6", "rate_1h", "rate_3h",
            "hour", "month", "dow"]
    return base + [c for c in WEATHER_COLS if c in d.columns]


def training_rows(d, threshold=THRESHOLD):
    """Onset prediction: only hours where AQI is currently at/below the threshold."""
    return d.dropna(subset=["aqi", "label"]).query("aqi <= @threshold").reset_index(drop=True)
