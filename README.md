# Credit card fraud detection and transaction decisions

How much fraud could a payment team flag while intervening on fewer than 1% of transactions? On a six-month holdout of 555,719 transactions, a class-weighted XGBoost model and a frozen decision policy flagged **1,975 of 2,145 fraud cases** and **2,962 legitimate transactions**. That amounts to approximately nine interventions per 1,000 transactions.

| Measure | Result |
| --- | ---: |
| Validation average precision | 0.9145 |
| Best SMOTENC challenger, validation average precision | 0.8925 |
| Holdout average precision | 0.8624 |
| Holdout intervention rate | 0.89% |
| Fraud-case recall | 92.07% |
| Precision among interventions | 40.00% |
| Share of observed fraud dollars flagged | 97.00% |

The notebooks label scikit-learn's average precision as PR-AUC. The dollar measure divides the value of fraudulent transactions flagged by the total value of fraudulent transactions. It measures exposure covered in an offline simulation. Actual losses prevented would depend on verification outcomes, recovery and operating costs.

![Holdout ROC and precision–recall curves with the policy confusion matrix](results/holdout-performance.png)

## Read the results

The following command prints case and dollar coverage and checks that three published tables agree. It runs with Python's standard library and the included aggregate CSVs.

```sh
python3 scripts/summarize_results.py
```

## Reproduce the study

The reference environment used Python 3.9.6 on macOS. From the repository root, create a virtual environment and install the reference package versions. LightGBM and XGBoost on macOS use the OpenMP runtime. A Mac with Homebrew can install it with `brew install libomp` if the library is missing.

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and unzip version 1 of Kartik Shenoy's simulated transaction dataset. Supply the directory containing `fraudTrain.csv` and `fraudTest.csv` to the preparation command. [Credit Card Transactions Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

```sh
python scripts/prepare_data.py "/path/to/unzipped/dataset"
python run.py all
```

The preparation script checks both source file hashes. The historical project used the original training file and a separately formatted test export with day-first dates and minute-resolution timestamps. The script recreates the fields used by the historical experiment from the public files. All 555,719 prepared test rows matched the original export in order across the 17 checked columns, including every input used for scoring and routing, the label and transaction ID. If you already have the project CSVs, place them directly at `Data/raw/fraud_train.csv` and `Data/raw/fraud_test.csv` and start with `python run.py all`.

Each stage also runs separately. For a shorter model demonstration, `python run.py evaluate` trains the frozen model from the training CSV, scores the full holdout and replays the decisions. The optional validation comparison inside that notebook is populated when the thresholds stage has already run.

| Stage | What runs | Output |
| --- | --- | --- |
| `eda` | [01 · EDA and preprocessing](01_EDA_Preprocessing.ipynb) | Training-data audit, exploratory charts and a processed table |
| `models` | [02 · Time-aware model selection](02_Model_Selection.ipynb) | Benchmark and tuned validation comparisons |
| `imbalance` | [03 · Class imbalance](03_Class_Imbalance.ipynb) | XGBoost and LightGBM at three SMOTENC ratios |
| `thresholds` | [Training-only threshold analysis](scripts/analyze_xgboost_thresholds.py) | Candidate cutoffs, score bands and behavior boundaries |
| `evaluate` | [04 · Final test evaluation](04_Final_Test_Evaluation.ipynb) | Frozen model and policy results, first-hit rule counts and history-key audit |

Each modeling notebook builds its inputs from the raw files. The chronological split uses the earliest 70% of the training file for development and the latest 30% for validation. Tuning for XGBoost and LightGBM uses a further split within development. Thresholds are chosen on validation. The final notebook refits the fixed XGBoost configuration on the original development period and applies the frozen cutoffs to the later test file. Raw scores rank relative risk. They have not been calibrated into fraud probabilities.

Reruns write tables under `Data/processed/` and plots under `figures/`. The compact reference outputs for the published baseline are in [results](results/). The public package contains aggregate tables and charts. Local notebook execution can also display transaction-level examples.

## What the behavior rules contribute

Every score at or above the medium cutoff goes to verification or hard decline. Adding merchant familiarity, amount spikes and prior customer activity changes how flagged transactions are handled. Case recall and dollar coverage at this operating point are fully determined by the score cutoff.

The original profile key hashes birth dates as text. The two files encode those dates differently, causing the historical replay to start all customers with an empty history at the train/test boundary. Parsing the dates before building the key recovers 908 previously observed profiles among 924 test profiles. The final notebook keeps the original baseline and measures how the normalization changes hard declines and verification requests while holding every model score and threshold fixed. On the holdout, normalization moved 245 transactions between hard decline and verification. Hard declines changed from 1,665 to 1,424, and legitimate hard declines changed from 214 to 143. Fraudulent transactions hard-declined changed from 1,451 to 1,281. Both versions flagged the same 4,937 transactions. [History routing audit](results/history_routing_audit.csv)

## Next experiment

Select a daily verification budget and a maximum acceptable rate of hard declines on legitimate purchases. On development and validation windows, compare a score-only routing policy with score-plus-history routing at the same total workload. Report legitimate hard declines, fraud dollars routed to verification, and sensitivity to the assumed cost of each action. Reserve a new time window for a prospective comparison after changing policy.
