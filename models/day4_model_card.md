---
license: mit
library_name: pytorch
pipeline_tag: tabular-regression
tags:
  - pytorch
  - mlp
  - tabular-regression
  - housing
  - affordability
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - mae
  - rmse
  - r2
---

# Housing affordability forecast (MLP): next year's affordability from this year's indicators

This model forecasts **next year's housing affordability** of a country (house prices relative to household income, as an index) from 17 housing and income indicators known **this year**. It is a small **multi-layer perceptron** (PyTorch, 26 → 32 → 32 → 1, 1,953 parameters) trained from scratch on 26 countries, 1977–2009, that predicts the **change** of affordability from this year to next. It is meant for housing policy analysts who want a quick one-year-ahead early warning. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27), as the neural-network counterpart of our [Ridge forecaster](https://huggingface.co/iaac-maai/day2-group8-affordability).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/Group08_Day04_AI-Methods
- **Team:** Group 8: Akshat Dayal, Bhavini Manot, Bo De Roeck

**Part of one project.** The same dataset runs through our earlier work: [data cleaning and exploration](https://github.com/iaac-maai/Group8-day1-assignment) produced the inputs; a [Ridge forecaster](https://huggingface.co/iaac-maai/day2-group8-affordability) (supervised learning) is the baseline this network is compared with, on the same test set; and a [clustering of housing-market regimes](https://huggingface.co/datasets/iaac-maai/day3-group8-housing-market-regimes) (affordable steady growth, price boom, expensive plateau, price correction) is used below to show **in which market phase** the forecasts fail.

## Model at a glance

![Test results against the baselines](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/06_test_comparison.png)

*On the test years 2015–2024 (used once), the network's mean error is 3.22 index points, against 3.28 for our Ridge forecaster and 3.74 for "next year = this year". But the Ridge forecaster has more forecasts within ±5% (83.1% vs 81.2%): the network **ties** with the linear model, it is not clearly better.*

| Model | MAE on the test set (index points) | R² (log scale) | Within ±5% |
|---|---|---|---|
| Baseline: next year = this year | 3.74 | 0.770 | 75.4% |
| Baseline: our Ridge forecaster | 3.28 | 0.826 | **83.1%** |
| **This model (MLP)** | **3.22** | **0.827** | 81.2% |

![Forecast vs actual and error distribution](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/07_predicted_vs_actual.png)

*Left: every test forecast against the actual value; the labelled points are the largest misses, all forecasts of 2023 made with 2022 data (and Canada's 2021 pandemic boom). Right: the errors of the network (orange) and of Ridge (green) have almost the same distribution.*

## Intended use

- **Intended users and use:** housing policy analysts, students and researchers who follow several countries and want a quick **one-year-ahead** estimate of affordability per country, as an early warning before official statistics are published, or to explore what-if scenarios (e.g. faster house price growth). Use it **next to** a simple baseline and the Ridge forecaster: a forecast where they disagree deserves a closer look.
- **Out-of-scope uses:**
  - countries outside the 26 in the training data, single cities or regions inside a country;
  - forecasts more than one year ahead;
  - comparing or ranking affordability **levels** between countries (the inputs are indexes with 2024 = 100 for every country);
  - investment decisions, and any decision about individual households or properties;
  - periods with shocks that the inputs cannot see (interest-rate jumps, pandemics): the model extrapolates momentum and misses turning points (see *Evaluation*).

## Model details

| | |
|---|---|
| **Task** | Tabular regression: forecast of `affordability` for year + 1 |
| **Architecture** | MLP: `Linear(26, 32)` → ReLU → Dropout(0.2) → `Linear(32, 32)` → ReLU → Dropout(0.2) → `Linear(32, 1)` |
| **Parameters** | 1,953 (all trainable) |
| **Base model** | None: trained from scratch |
| **Input** | One country-year with 17 columns (table below), preprocessed into 26 numbers with the fitted `ColumnTransformer` in `scaler.joblib` |
| **Output** | One number `z`: the standardised change of log10 affordability. Forecast = `10 ** (log10(affordability) + z × target_std + target_mean)` (both stored in `scaler.joblib`) |
| **Files** | `model.pt` (the `state_dict`, 10 kB); `scaler.joblib` (a dictionary: `preprocess` = the fitted scikit-learn `ColumnTransformer`, `columns` = the 17 input columns in order, `target_mean`, `target_std`); the figures of this card |
| **Library versions** | torch 2.14.1, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0, Python 3.12 |

### Input features

One row = one country in one year. These 17 columns, **in this order** (`bundle["columns"]`):

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `year` | numeric | year | Year of the indicators (the forecast is for `year + 1`) | median imputation, standard scaling |
| `decade` | ordinal | text | `"1970s"` … `"2020s"` | ordinal encoding 0–5 (not scaled) |
| `affordability` | numeric | index | Real house price index ÷ real disposable income index × 100 (higher = less affordable) | median imputation, standard scaling |
| `real_house_price_index` | numeric | index | House prices adjusted for inflation | median imputation, standard scaling |
| `real_personal_disposable_income` | numeric | index | Household income after tax, adjusted for inflation | median imputation, standard scaling |
| `real_hpi_change_pct` | numeric | % | Change of real house prices vs the previous year | median imputation, standard scaling |
| `real_income_change_pct` | numeric | % | Change of real income vs the previous year | median imputation, standard scaling |
| `affordability_lag1` | numeric | index | Affordability one year earlier | median imputation, standard scaling |
| `affordability_lag2` | numeric | index | Affordability two years earlier | median imputation, standard scaling |
| `affordability_3yr_avg` | numeric | index | Average affordability of the last three years | median imputation, standard scaling |
| `nominal_zero` | 0/1 | flag | Data-cleaning flag: original nominal price or income was 0 (hyperinflation artefact). Always 0 in this data | median imputation, standard scaling |
| `hyperinflation_country` | 0/1 | flag | Data-cleaning flag: Israel, Croatia or Slovenia | median imputation, standard scaling |
| `price_shock_up` | 0/1 | flag | Data-cleaning flag: real house prices rose more than 30% this year | median imputation, standard scaling |
| `price_shock_down` | 0/1 | flag | Data-cleaning flag: real house prices fell more than 30% this year. Always 0 in this data | median imputation, standard scaling |
| `gfc_period` | 0/1 | flag | Data-cleaning flag: global financial crisis and aftermath (2008–2012) | median imputation, standard scaling |
| `log_affordability` | numeric | log10 | `log10(affordability)` | median imputation, standard scaling |
| `region` | categorical | text | One of 10 world regions (e.g. `"Southern Europe"`, `"Nordic"`) | most-frequent imputation, one-hot encoding (`handle_unknown="ignore"`) |

## Training data

| | |
|---|---|
| **Dataset** | Housing affordability, 26 countries, 1977–2024, cleaned by our team (`housing_affordability_model_ready.csv`) |
| **Source** | Kaggle notebook [Forecasting housing affordability](https://www.kaggle.com/code/tolstoyjustin/forecasting-housing-affordability); cleaned file in our [data-cleaning repository](https://github.com/iaac-maai/Group8-day1-assignment/tree/main/data) |
| **Licence** | CC BY-SA 4.0 |
| **Size** | 1,210 country-years: training 820 (1977–2009), validation 130 (2010–2014), test 260 (2015–2024); split **by time**, the same test set as the Ridge forecaster |
| **Coverage** | 26 mostly high-income countries (Europe, North America, Oceania, East Asia, plus Colombia, Israel and South Africa); national averages; all series are indexes with 2024 = 100 |

## Training procedure

- **Target:** the change `log10(affordability_next) − log10(affordability)`, standardised with the training mean (−0.0005) and standard deviation (0.0291). Predicting the change instead of the level was the decisive choice (validation MAE 0.0098 vs 0.0154).
- **Settings:** MSE loss, Adam, learning rate 1e-4, weight decay 1e-3, dropout 0.2, batch size 32, seed 42, at most 300 epochs.
- **Model selection:** the weights of the epoch with the lowest validation loss (**epoch 69**); the test set was used once, at the end. The final model learned from 1977–2009 only (the Ridge forecaster from 1977–2014).
- **Hardware and time:** a laptop CPU, about **10–30 seconds** for 300 epochs.

![Training curves](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/02_training_curves.png)

*Left: the validation loss is lowest at epoch 69 and then rises slowly while the training loss keeps falling: the start of overfitting, so we keep epoch 69. Right: the validation error falls below "next year = this year" after 7 epochs and below the Ridge forecaster after 50; trained for 300 epochs, it ends up worse than Ridge again.*

![One change at a time](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/05_experiments.png)

*Each bar changes one setting (mean and spread of 5 random seeds, validation 2010–2014). Predicting the level directly is worse than the naive baseline; dropping early stopping makes it worse than Ridge; dropout, learning rate and network size make no measurable difference.*

![Inside the network](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/04_one_example_through_the_network.png)

*Spain in 2012 travelling through the network: 26 inputs (prices and incomes falling), 15 and 20 of the 32 hidden neurons active, and a forecast of −5.5% (112.4 for 2013; actual 111.9; "next year = this year" would have said 118.9).*

## Evaluation

| Metric (test 2015–2024, 260 country-years) | Next year = this year | Ridge forecaster | **This model** |
|---|---|---|---|
| MAE (index points) | 3.74 | 3.28 | **3.22** |
| RMSE (index points) | 5.26 | 4.75 | **4.74** |
| R² (log10 scale) | 0.770 | 0.826 | **0.827** |
| Median absolute error | – | 2.09% | **2.06%** |
| Forecasts within ±5% | 75.4% | **83.1%** | 81.2% |
| Misses above 10% | – | 15 | **13** |

The differences between the network and Ridge are smaller than what changes with a different random seed: **treat them as equal**. Both are about 12–14% better than "next year = this year".

**Where it fails.** The errors are concentrated in a few years:

| Forecast for | 2016–2019 | 2020 | 2021 | 2022 | **2023** | 2024–2025 |
|---|---|---|---|---|---|---|
| Mean absolute error (MLP) | 1.4–2.6% | 3.0% | 4.0% | 4.7% | **7.2%** | 2.1–2.3% |
| Mean error (MLP), sign = direction | −0.2 to +2.1% | −1.5% | −1.7% | +2.6% | **+7.1%** | +0.1 / −0.4% |

- **2023:** affordability improved by 5.4% on average (the year after interest rates rose sharply, which is not in the inputs). The 2022 indicators gave little warning (real prices still rising, or only 1–2% down), so the network forecast affordability to stay about level or worsen, while it fell 10–14% in Germany, Sweden, the Netherlands, South Korea and Canada: 6 of its 13 misses above 10% are forecasts for 2023 (errors of +14 to +16%).
- **2021:** the pandemic housing boom (+4.9% on average) was not visible in the 2020 data: under-forecasts (Canada 2020 → 2021: −15%).
- **Per country:** most accurate for South Africa and Colombia (median error below 1%), least for **Sweden, New Zealand and Canada** (about 4%). Better than Ridge for Japan and Korea, worse for Spain and Portugal.
- **All 8 of the network's largest misses are also misses of 13–16% for Ridge:** the two models fail at the same turning points.

**By market regime.** We grouped the test errors by the regime of the forecast year, from our clustering analysis (the regimes are not an input: they use each country's average over the whole period, which would leak the future into a forecast):

| Regime of the forecast year | Test rows | Next year = this year | Ridge forecaster | **This model** | Within ±5% |
|---|---|---|---|---|---|
| Affordable steady growth | 68 | 2.9% | 2.7% | **2.4%** | 88% |
| Price boom | 38 | 5.4% | 4.8% | **4.8%** | **63%** |
| Expensive plateau | 133 | 3.5% | **3.0%** | 3.2% | 82% |
| Price correction | 21 | 3.8% | 2.8% | **2.6%** | 86% |

![Errors by market regime](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/10_errors_by_regime.png)

*Mean absolute error by regime (left) and the direction of the network's errors (right). **Booms are the hardest phase** for every model: after a boom year affordability worsens by 3.6% on average, and the network forecasts even more (mean error +2.1%). Corrections are forecast well. The differences between the network and Ridge within each regime (0.0–0.4 points) are small: a forecast made in a boom year deserves the most scrutiny, whichever model made it.*

![Failures per country and over time](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/08_failures.png)

*Left: median error per country, network vs Ridge forecaster. Right: the two hardest countries; the forecasts (dashed) follow the actual values one year late and overshoot the 2022–2023 turn.*

## What it does: a scenario

![Scenarios for Spain](https://huggingface.co/iaac-maai/day4-group8-affordability-mlp/resolve/main/09_scenarios_spain.png)

*Spain's 2024 indicators with different house price growth in 2024, and the forecast for 2025. With the actual data the network forecasts +2.5% (Ridge +1.5%). Both turn faster price growth into a worse forecast and agree for moderate scenarios (+10%: 116.7 vs 116.9); for extreme ones the network's response flattens (+15%: 123.5 vs 125.1).*

## Limitations and biases

- **Booms:** forecasts made in a price-boom year are the least reliable (4.8% mean error, 63% within ±5%) and tend to be too high.
- **Missing drivers:** no interest rates, credit, construction, supply or migration. The model extrapolates momentum and **misses turning points**, which is exactly when an early warning matters most. "No deterioration forecast" does not mean "no risk".
- **Indexes, not levels:** every country's series is an index with 2024 = 100; the model cannot say which country is more affordable.
- **Small data:** 820 training and 260 test country-years; differences of a few percent between models are noise.
- **Coverage:** 26 mostly high-income countries and national averages only (a city boom can hide in a calm national average). The early years of South Korea, Israel and Croatia are the least reliable (see the data-cleaning repository).
- **Training period:** learned from 1977–2009 only; the structure of housing markets after 2010 (low interest rates, then a sharp rise) is under-represented.

## How to use

```python
# pip install torch scikit-learn pandas joblib huggingface_hub
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from huggingface_hub import hf_hub_download

REPO = "iaac-maai/day4-group8-affordability-mlp"

# 1. Build EXACTLY the trained architecture (the Dropout layers are needed for the layer names), then load
model = nn.Sequential(
    nn.Linear(26, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 1),
)
model.load_state_dict(torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu"))
model.eval()
bundle = joblib.load(hf_hub_download(REPO, "scaler.joblib"))   # preprocessing, column order, target mean and std

# 2. One country-year with the 17 input columns (here: Spain 2024 from our cleaned data)
url = "https://raw.githubusercontent.com/iaac-maai/Group8-day1-assignment/main/data/housing_affordability_model_ready.csv"
df = pd.read_csv(url)
row = df[(df["iso"] == "ESP") & (df["year"] == 2024)].copy()
row["log_affordability"] = np.log10(row["affordability"])
row["region"] = row.filter(like="region_").idxmax(axis=1).str.replace("region_", "")
row["decade"] = row["decade"].astype(str) + "s"
X = row[bundle["columns"]]

# 3. Forecast: the network predicts the standardised change of log10(affordability)
with torch.no_grad():
    z = model(torch.tensor(bundle["preprocess"].transform(X), dtype=torch.float32)).item()
change = z * bundle["target_std"] + bundle["target_mean"]
forecast = 10 ** (X["log_affordability"].item() + change)
print(f"Spain 2024: {X['affordability'].item():.1f} -> forecast for 2025: {forecast:.1f} ({(10 ** change - 1) * 100:+.1f}%)")
# Spain 2024: 100.0 -> forecast for 2025: 102.5 (+2.5%)
```

⚠️ `.pt` and `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; `scaler.joblib` may not load with other versions of scikit-learn.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/Group08_Day04_AI-Methods. Run `notebook/neural_network.ipynb` with the packages in `requirements.txt`; it downloads the data, trains in about 10–30 seconds on a CPU (the whole notebook, with the 35 trainings of the experiments, in about 5 minutes) and reproduces every number of this card (seed 42).

## Authors and licence

Akshat Dayal, Bhavini Manot, Bo De Roeck (Group 8). Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the **MIT License**. The training data is subject to its own licence: CC BY-SA 4.0 (credit the original Kaggle source and share derived data under the same licence).
