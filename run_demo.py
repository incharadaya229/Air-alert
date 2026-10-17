"""End-to-end run: python run_demo.py hebbal.csv weather.csv"""
import sys

from src.decide import DISCLAIMER
from src.loader import coverage_report, load_wide_hourly
from src.prepare import (THRESHOLD, add_features, build_table, clean, feature_names,
                         load_weather, training_rows)
from src.replay import pick_window, replay
from src.train import baseline_predict, evaluate, time_split, train_model

aqi_path = sys.argv[1]
weather_path = sys.argv[2] if len(sys.argv) > 2 else None

raw = load_wide_hourly(aqi_path, station=aqi_path.rsplit("/", 1)[-1])
print("COVERAGE:", coverage_report(raw))
weather = load_weather(weather_path) if weather_path else None
table = clean(build_table(raw, weather))
print(f"SENSOR CHECKS: stuck readings removed={int(table.flag_stuck.sum())}, "
      f"out-of-range removed={int(table.flag_range.sum())}")

d = add_features(table)
features = feature_names(d)
rows = training_rows(d)
train, test, cut = time_split(rows)
print(f"TRAIN rows={len(train)}  TEST rows={len(test)}  test starts {cut}")

model = train_model(train, features)
p = model.predict_proba(test[features].to_numpy(float))[:, 1]
y = test["label"].to_numpy()
print("MODEL (warn at prob>=0.25):", evaluate(y, p >= 0.25))
print("BASELINE (AQI already near threshold):", evaluate(y, baseline_predict(test, THRESHOLD)))

print("\nLIVE REPLAY (busiest 48h of test period)")
_, stats = replay(model, pick_window(test), features)
print("TIMING:", stats)
print(DISCLAIMER)
