# cycle-tracking Development Plan

## Project Overview

**Goal:** Load, preprocess, and analyze cycle tracking data; build features and models for insights
**Stack:** Python 3.12, pandas, scikit-learn, matplotlib, loguru, typer, ruff, pipenv

## Current Phase: Data Loading & Preprocessing

<!-- Status markers: [ ] pending, [~] in progress, [x] complete -->

### Phase 1: Data Loading & Preprocessing

- [x] Implement data loading in `src/dataset.py`
  - Intraday data: heart_rate, active_zone_minutes, glucose
  - Interday data: hormones_and_selfreport, sleep, resting_heart_rate, heart_rate_variability_details, computed_temperature, active_minutes
    - Hormones: keep lh, estrogen and pdg
    - Sleep data: keep total sleep duration and sleep stage durations
    - Computed temperature: keep nightly temperature and baseline_relative_sample_sum variables
    - Heart rate variability details: keep daily mean, min and max of rmssd variable
- [x] Filter data keeping 2022 study interval, start from first day of sleep data
- [x] Remove cycles with 4 consecutive days of hormone data missing or more than 40% of hormone data missing (consider only estrogen and lh)
- [x] Remove days with less than 18 hours of intraday data
- [x] Sample intraday data to 5-minute intervals using mean aggregation, linearly interpolate missing values for heart rate and glucose, fill with 0 for active zone minutes
- [x] Save processed intraday and interday data as separate CSV files in `data/processed/`
- [x] Create notebook for initial data exploration (`notebooks/initial-exploration.ipynb`)

### Phase 2: Feature Engineering

Interday and intraday features:

- [x] Cycle count, cycle day count and % of cycle progress

Interday features:

- [x] Estrogen-to-lh ratio
- [ ] LH deviation from baseline (5-day average, 5 days before LH surge)
- [x] Total active minutes
- [x] Sleep efficiency ratios - deep/total and rem/total
- [x] Convert reports to numeric values following Likert-type scale (0-5)

Intraday features:

- [x] Time of day - morning/afternoon/evening/night
- [x] Add cycle phase labels from interday dataset

- [x] Implement feature engineering in `src/features.py`

### Phase 3: Visualization & Analysis

- [x] Create two notebooks: `subject-visualization.ipynb` and `group-visualization.ipynb` (in subject visualizations, plot variables for 10 select subjects in subplots, in group visualizations, plot means and 95% confidence intervals across subjects)
- [x] Visualize cycle length distribution and cycle phase duration distributions (histograms and boxplots)
- [x] Visualize hormones across cycles and cycle progression (two subplots; line plot)
- [x] Visualize heart rate and glucose across day for each cycle phase (rows for cycle phase; line plots)
- [x] Visualize report counts for each cycle phase (phase in x-axis; stacked bar plot)
- [x] Visualize sleep ratios and total sleep duration across cycle phases (boxplots)

### Phase 4: Modeling

- [ ] Define modeling objective
- [ ] Implement training pipeline in `src/modeling/train.py`
- [ ] Implement inference in `src/modeling/predict.py`

## Notes

<!-- Add implementation notes, decisions, and blockers here -->
