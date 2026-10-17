# AirAlert: Early Warning for Air Quality Spikes

**[Team Name]_IBM Z Datathon 2026 Project**
Theme: Real-Time AI for Critical Decisions (Detect. Decide. Respond.)
Team: [Member 1], [Member 2], [Member 3], [Member 4]

## Problem

Air quality apps report pollution after it happens. Children, elderly people, people with asthma or heart conditions and outdoor workers need warning **before** air quality gets bad, and they need to know what to do about it.

## Solution

AirAlert reads hourly air quality readings, checks for faulty sensor data, and predicts whether the Air Quality Index (AQI) will cross 200 within the next 2 hours. Each hour is rated **Low / Watch / Alert**, with plain-language advice for schools, elderly people and care homes, people with asthma or heart conditions, and outdoor workers. Alerts are by area and use no personal data.

A "spike" means AQI rises above 200 within 2 hours, starting from an hour where AQI is at or below 200 (CPCB's "Poor" category begins above 200). The model therefore answers "will it cross the line soon?", not "is it already high?".

## Data

- **Air quality:** Hebbal station (Bengaluru), hourly AQI values, June 2018 to December 2023, from the Central Pollution Control Board (CPCB) via OpenCity.in (Public Domain). Source: https://data.opencity.in/dataset/bengaluru-hourly-air-quality-reports
- 48,312 hourly rows, 15.4% missing, longest outage 662 hours.
- **Sensor checks:** 119 stuck (flat-lined) readings removed before training.
- The files hold AQI index values (the pollutant is not labelled in the file), so this project refers to AQI, not PM2.5 concentration.
- **Weather:** the code can merge Open-Meteo weather (temperature, humidity, wind), but **the results below use AQI history and time features only**. Weather is future work.

## How it works

1. `src/loader.py`: converts the calendar-style CSV into an hourly table.
2. `src/prepare.py`: sensor checks, features (recent values, rolling averages, rate of rise, hour, month, day of week) and the spike label.
3. `src/train.py`: scikit-learn HistGradientBoostingClassifier, trained on earlier dates and tested on later dates only, compared with a simple baseline.
4. `src/decide.py`: probability to Low / Watch / Alert, plus advice.
5. `src/replay.py`: replays readings hour by hour as a live stream and times each score.

## Results

Trained on data up to 15 June 2022 (29,433 hours). Tested on later data the model never saw (9,811 hours, 95 real spikes).

| Level | Probability cutoff | Spikes caught (recall) | Warnings that were real (precision) | False alarms among calm hours |
|---|---|---|---|---|
| Watch (early heads-up) | 0.25 | 50.5% (48 of 95) | 5.2% | 9.0% |
| Alert (higher confidence) | 0.50 | 32.6% (31 of 95) | 8.5% | 3.4% |
| Stricter Alert | 0.70 | 25.3% (24 of 95) | 13.2% | 1.6% |
| Baseline: "AQI already near 200" | n/a | 17.9% (17 of 95) | 25.0% | 0.5% |

**Reading these honestly:** spikes are rare (about 1% of hours). The model catches more spikes than the baseline at every level, but it also raises more false alarms: it trades precision for earlier warning. False alarm figures are per hour, not per warning episode.

**Speed on LinuxONE:** each reading was scored in under 1 millisecond (about 0.64 to 0.70 ms average), and batch scoring ran at roughly 57,000 to 71,000 readings per second across two runs.

## Platform: IBM LinuxONE

Developed and run on an IBM LinuxONE instance (s390x, Big Endian, Python 3.12, 3 cores, about 8 GB memory). The environment runs in a container (`/.dockerenv` present). Screenshots are in `screenshots/`.

## Run it

```bash
pip install pandas numpy scikit-learn
python3 run_demo.py hebbal.csv            # AQI only
python3 run_demo.py hebbal.csv weather.csv   # optional Open-Meteo weather
```

Download the data from the OpenCity link above (Hebbal hourly AQI 2017-2023). Data files are not stored in this repo.

## Limitations

- One station (Hebbal), one threshold (AQI 200), one horizon (2 hours).
- Many false alarms at the Watch level; Alert is more selective but catches fewer spikes.
- 15% of readings are missing and some months are empty.
- Advice is general guidance based on published AQI health recommendations, not medical advice.

## Next steps

Add weather features, more stations, a live data feed and a dashboard.

## Credits

- Data: CPCB / OpenCity.in (Public Domain). Weather (planned): Open-Meteo.com (CC BY 4.0), "Weather data by Open-Meteo.com".
- Open-source libraries: pandas, NumPy, scikit-learn.
- Built during the IBM Z Datathon 2026 hacking window, with help from an AI assistant (Claude) for code and planning.
