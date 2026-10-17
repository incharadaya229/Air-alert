"""Step 4a: turn a spike probability into Low / Watch / Alert plus group-specific advice."""
WATCH_PROB = 0.25
ALERT_PROB = 0.50
FAST_RISE = 40  # AQI points gained over 3 hours counts as "rising fast"

ADVICE = {
    "Alert": {
        "Schools": "Move PE and outdoor activities indoors now.",
        "Elderly / care homes": "Keep residents indoors; keep windows closed.",
        "People with asthma or heart conditions": "Avoid going out; keep medication within reach.",
        "Outdoor workers": "Wear an N95-type mask; take regular breaks in cleaner indoor air.",
    },
    "Watch": {
        "Schools": "Plan to shorten or shift outdoor activity.",
        "Elderly / care homes": "Limit time outdoors in the next few hours.",
        "People with asthma or heart conditions": "Reduce exertion outdoors; have medication handy.",
        "Outdoor workers": "Keep a mask ready and watch for further rise.",
    },
    "Low": {"Everyone": "No special precautions expected."},
}
DISCLAIMER = "General guidance based on published AQI health recommendations - not medical advice."


def tier(prob, aqi, rate_3h, threshold=200):
    if aqi > threshold:
        return "Alert"
    if prob >= ALERT_PROB:
        return "Alert"
    if prob >= WATCH_PROB or (rate_3h == rate_3h and rate_3h >= FAST_RISE):  # NaN-safe
        return "Watch"
    return "Low"
