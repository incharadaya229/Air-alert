"""Step 3: time-based split, model, honest metrics, and a simple baseline."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def time_split(d, test_frac=0.25):
    cut = d["timestamp"].sort_values().iloc[int(len(d) * (1 - test_frac))]
    return d[d["timestamp"] < cut].copy(), d[d["timestamp"] >= cut].copy(), cut


def train_model(train, features):
    y = train["label"].astype(int).to_numpy()
    pos = y.mean()
    if pos == 0 or pos == 1:
        raise ValueError("Training data has only one class - check threshold/date range.")
    w = np.where(y == 1, (1 - pos) / pos, 1.0)   # balance rare spikes
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=5, random_state=0)
    model.fit(train[features].to_numpy(float), y, sample_weight=w)
    return model


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    return {
        "spikes_in_test": int(y_true.sum()),
        "recall": round(tp / max(tp + fn, 1), 3),            # share of real spikes we warned about
        "precision": round(tp / max(tp + fp, 1), 3),         # share of warnings that were real
        "false_alarm_rate": round(fp / max(fp + tn, 1), 3),  # false warnings among calm hours
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def baseline_predict(test, threshold):
    """Naive rule: warn when AQI is already close to the threshold."""
    return (test["aqi"] >= 0.85 * threshold).to_numpy()
