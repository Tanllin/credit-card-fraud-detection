"""Fit the selected time-aware XGBoost model and derive operating score bands."""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Data" / "raw" / "fraud_train.csv"
THRESHOLD_PATH = ROOT / "Data" / "processed" / "xgboost_threshold_candidates.csv"
BAND_PATH = ROOT / "Data" / "processed" / "xgboost_score_bands.csv"
RULE_PATH = ROOT / "Data" / "processed" / "xgboost_rule_candidates.csv"
BEHAVIOR_PATH = ROOT / "Data" / "processed" / "behavior_flag_thresholds.csv"
RANDOM_STATE = 42


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def prepare_model_features(raw):
    data = raw.drop(
        columns=[column for column in raw.columns if column.startswith("Unnamed:")],
        errors="ignore",
    ).copy()
    transaction_time = pd.to_datetime(
        data["trans_date_trans_time"], format="mixed", dayfirst=True, errors="coerce"
    )
    birth_date = pd.to_datetime(
        data["dob"], format="mixed", dayfirst=True, errors="coerce"
    )

    features = pd.DataFrame(index=data.index)
    features["amt_log1p"] = np.log1p(data["amt"])
    features["city_pop_log1p"] = np.log1p(data["city_pop"])
    features["age_at_transaction"] = (
        transaction_time - birth_date
    ).dt.days / 365.2425
    features["trans_hour"] = transaction_time.dt.hour
    features["trans_day_of_week"] = transaction_time.dt.dayofweek
    features["trans_month"] = transaction_time.dt.month
    features["is_weekend"] = transaction_time.dt.dayofweek.ge(5).astype("int8")
    features["is_night"] = transaction_time.dt.hour.between(0, 5).astype("int8")
    features["hour_sin"] = np.sin(2 * np.pi * transaction_time.dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * transaction_time.dt.hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * transaction_time.dt.dayofweek / 7)
    features["day_cos"] = np.cos(2 * np.pi * transaction_time.dt.dayofweek / 7)
    features["merchant_distance_km"] = haversine_km(
        data["lat"].to_numpy(),
        data["long"].to_numpy(),
        data["merch_lat"].to_numpy(),
        data["merch_long"].to_numpy(),
    )
    features["lat"] = data["lat"]
    features["long"] = data["long"]
    for column in ["category", "gender", "state"]:
        features[column] = data[column].astype("string").str.strip().fillna("missing")

    return features, data["is_fraud"].astype("int8"), transaction_time


def prepare_history_features(raw, transaction_time):
    """Create leakage-safe customer history from earlier transactions only."""
    history = raw.copy()
    history["_transaction_time"] = transaction_time
    history["_row_order"] = np.arange(len(history))
    history["_merchant_normalized"] = (
        history["merchant"].astype("string").str.removeprefix("fraud_").str.strip()
    )
    profile_columns = ["first", "last", "dob", "street"]
    history["_customer_key"] = pd.util.hash_pandas_object(
        history[profile_columns].astype("string"), index=False
    ).astype("uint64")

    history = history.sort_values(
        ["_customer_key", "_transaction_time"], kind="stable"
    ).copy()
    customer_group = history.groupby("_customer_key", sort=False)
    history["customer_prior_txn_count"] = customer_group.cumcount()
    history["minutes_since_previous_txn"] = (
        customer_group["_transaction_time"].diff().dt.total_seconds().div(60)
    )
    history["is_first_observed_txn"] = history[
        "minutes_since_previous_txn"
    ].isna()

    prior_amount_sum = customer_group["amt"].cumsum() - history["amt"]
    prior_amount_mean = prior_amount_sum.div(
        history["customer_prior_txn_count"].replace(0, np.nan)
    )
    history["amount_to_prior_mean"] = (
        history["amt"] / prior_amount_mean.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    history["customer_merchant_prior_count"] = history.groupby(
        ["_customer_key", "_merchant_normalized"], sort=False
    ).cumcount()
    history["is_new_merchant_for_customer"] = history[
        "customer_merchant_prior_count"
    ].eq(0)

    columns = [
        "customer_prior_txn_count", "minutes_since_previous_txn",
        "is_first_observed_txn", "amount_to_prior_mean",
        "customer_merchant_prior_count", "is_new_merchant_for_customer",
    ]
    return history.sort_values("_row_order", kind="stable")[columns].reset_index(drop=True)


def metrics_at_threshold(y_true, score, threshold):
    prediction = score >= threshold
    true_positive = int(((prediction == 1) & (y_true == 1)).sum())
    false_positive = int(((prediction == 1) & (y_true == 0)).sum())
    false_negative = int(((prediction == 0) & (y_true == 1)).sum())
    true_negative = int(((prediction == 0) & (y_true == 0)).sum())
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "threshold": threshold,
        "flagged_transactions": true_positive + false_positive,
        "flag_rate": (true_positive + false_positive) / len(y_true),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


identifier_types = {"cc_num": "string", "zip": "string", "trans_num": "string"}
raw = pd.read_csv(TRAIN_PATH, dtype=identifier_types, low_memory=False)
X_all, y_all, transaction_time = prepare_model_features(raw)
history_all = prepare_history_features(raw, transaction_time)

order = np.argsort(transaction_time.to_numpy(), kind="stable")
X_all = X_all.iloc[order].reset_index(drop=True)
y_all = y_all.iloc[order].reset_index(drop=True)
history_all = history_all.iloc[order].reset_index(drop=True)

split_index = int(len(X_all) * 0.70)
X_train, X_validation = X_all.iloc[:split_index], X_all.iloc[split_index:]
y_train, y_validation = y_all.iloc[:split_index], y_all.iloc[split_index:]
history_train = history_all.iloc[:split_index].reset_index(drop=True)
history_validation = history_all.iloc[split_index:].reset_index(drop=True)

numeric_features = [
    "amt_log1p", "city_pop_log1p", "age_at_transaction", "trans_hour",
    "trans_day_of_week", "trans_month", "is_weekend", "is_night",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "merchant_distance_km", "lat", "long",
]
categorical_features = ["category", "gender", "state"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ]), categorical_features),
])

negative_count = int(y_train.eq(0).sum())
positive_count = int(y_train.eq(1).sum())
scale_pos_weight = negative_count / positive_count

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=117,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        gamma=0.1,
        reg_lambda=5.0,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )),
])
model.fit(X_train, y_train)
score = model.predict_proba(X_validation)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_validation, score)
curve = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1],
})
curve["f1"] = (
    2 * curve["precision"] * curve["recall"]
    / (curve["precision"] + curve["recall"]).replace(0, np.nan)
)

# Hard: highest coverage available while maintaining at least 90% fraud hit rate.
hard_row = curve.loc[curve["precision"].ge(0.90)].sort_values(
    ["recall", "threshold"], ascending=[False, True]
).iloc[0]

# Medium: highest fraud coverage available while maintaining at least 50%
# precision. This creates a useful verification tier below the hard cutoff;
# the maximum-F1 threshold is almost identical to the hard threshold.
medium_row = curve.loc[curve["precision"].ge(0.50)].sort_values(
    ["recall", "threshold"], ascending=[False, True]
).iloc[0]

# Low: highest threshold that still captures at least 99% of fraud.
low_row = curve.loc[curve["recall"].ge(0.99)].sort_values(
    ["threshold", "precision"], ascending=False
).iloc[0]

selected = [
    ("T_hard", "At least 90% precision", float(hard_row["threshold"])),
    ("T_medium", "At least 50% precision", float(medium_row["threshold"])),
    ("T_low", "At least 99% recall", float(low_row["threshold"])),
]

threshold_rows = []
for name, objective, threshold in selected:
    row = metrics_at_threshold(y_validation.to_numpy(), score, threshold)
    row = {"threshold_name": name, "selection_objective": objective, **row}
    threshold_rows.append(row)

threshold_table = pd.DataFrame(threshold_rows)
threshold_table.to_csv(THRESHOLD_PATH, index=False)

t_hard = selected[0][2]
t_medium = selected[1][2]
t_low = selected[2][2]

bands = pd.cut(
    score,
    bins=[-np.inf, t_low, t_medium, t_hard, np.inf],
    labels=[
        "Approve candidate: score < T_low",
        "Monitor/review: T_low to T_medium",
        "Soft decline: T_medium to T_hard",
        "Hard-decline candidate: score >= T_hard",
    ],
    right=False,
)
band_frame = pd.DataFrame({"score_band": bands, "is_fraud": y_validation})
band_table = (
    band_frame.groupby("score_band", observed=False)["is_fraud"]
    .agg(transactions="size", fraud_cases="sum", observed_fraud_rate="mean")
    .reset_index()
)
band_table["nonfraud_transactions"] = (
    band_table["transactions"] - band_table["fraud_cases"]
)
band_table["share_of_validation"] = band_table["transactions"] / len(y_validation)
band_table["share_of_all_fraud"] = band_table["fraud_cases"] / int(y_validation.sum())
band_table.to_csv(BAND_PATH, index=False)

# Customer-history boundaries are learned from the development period only. The
# 10th and 90th percentiles correspond to the extreme deciles shown in notebook 01.
nonfirst_train = history_train.loc[
    ~history_train["is_first_observed_txn"]
]
prior_count_q10 = float(history_train["customer_prior_txn_count"].quantile(0.10))
rapid_interval_q10 = float(
    nonfirst_train["minutes_since_previous_txn"].quantile(0.10)
)
long_interval_q90 = float(
    nonfirst_train["minutes_since_previous_txn"].quantile(0.90)
)
amount_spike_q90 = float(
    nonfirst_train["amount_to_prior_mean"].dropna().quantile(0.90)
)

behavior_thresholds = pd.DataFrame([
    {
        "behavior": "limited_history_nonfirst",
        "development_boundary": prior_count_q10,
        "definition": "Not first observed and prior customer count <= boundary",
    },
    {
        "behavior": "rapid_repeat",
        "development_boundary": rapid_interval_q10,
        "definition": "Minutes since previous transaction <= boundary",
    },
    {
        "behavior": "long_inactivity",
        "development_boundary": long_interval_q90,
        "definition": "Minutes since previous transaction >= boundary",
    },
    {
        "behavior": "amount_spike",
        "development_boundary": amount_spike_q90,
        "definition": "Current amount / prior customer mean >= boundary",
    },
])
behavior_thresholds.to_csv(BEHAVIOR_PATH, index=False)

validation_rules = X_validation.copy()
validation_rules = pd.concat(
    [validation_rules.reset_index(drop=True), history_validation], axis=1
)
validation_rules["score"] = score
validation_rules["is_fraud"] = y_validation.to_numpy()
validation_rules["limited_history_nonfirst"] = (
    ~validation_rules["is_first_observed_txn"]
    & validation_rules["customer_prior_txn_count"].le(prior_count_q10)
)
validation_rules["rapid_repeat"] = (
    ~validation_rules["is_first_observed_txn"]
    & validation_rules["minutes_since_previous_txn"].le(rapid_interval_q10)
)
validation_rules["long_inactivity"] = (
    ~validation_rules["is_first_observed_txn"]
    & validation_rules["minutes_since_previous_txn"].ge(long_interval_q90)
)
validation_rules["amount_spike"] = (
    ~validation_rules["is_first_observed_txn"]
    & validation_rules["amount_to_prior_mean"].ge(amount_spike_q90)
)
validation_rules["high_risk_category"] = validation_rules["category"].isin(
    ["shopping_net", "misc_net", "grocery_pos"]
)
validation_rules["high_risk_time"] = (
    validation_rules["trans_hour"].ge(22)
    | validation_rules["trans_hour"].le(3)
)
validation_rules["behavior_flag_count"] = (
    validation_rules["is_first_observed_txn"].astype("int8")
    + validation_rules["limited_history_nonfirst"].astype("int8")
    + validation_rules["rapid_repeat"].astype("int8")
    + validation_rules["long_inactivity"].astype("int8")
    + validation_rules["amount_spike"].astype("int8")
    + validation_rules["is_new_merchant_for_customer"].astype("int8")
    + validation_rules["high_risk_category"].astype("int8")
    + validation_rules["high_risk_time"].astype("int8")
)
validation_rules["history_uncertainty"] = (
    validation_rules["is_first_observed_txn"]
    | validation_rules["limited_history_nonfirst"]
)
validation_rules["interval_extreme"] = (
    validation_rules["rapid_repeat"]
    | validation_rules["long_inactivity"]
)
validation_rules["risky_category_or_time"] = (
    validation_rules["high_risk_category"]
    | validation_rules["high_risk_time"]
)


def summarize_rule(name, mask):
    subset = validation_rules.loc[mask]
    fraud_cases = int(subset["is_fraud"].sum())
    transactions = len(subset)
    return {
        "rule_candidate": name,
        "transactions": transactions,
        "fraud_cases": fraud_cases,
        "nonfraud_transactions": transactions - fraud_cases,
        "fraud_hit_rate": fraud_cases / transactions if transactions else np.nan,
        "share_of_validation": transactions / len(validation_rules),
        "share_of_all_fraud": fraud_cases / int(y_validation.sum()),
    }


rule_rows = [
    summarize_rule(
        "Score >= T_hard",
        validation_rules["score"].ge(t_hard),
    ),
    summarize_rule(
        "Score >= T_hard AND >=1 behavior flag",
        validation_rules["score"].ge(t_hard)
        & validation_rules["behavior_flag_count"].ge(1),
    ),
    summarize_rule(
        "Score >= T_hard AND >=2 behavior flags",
        validation_rules["score"].ge(t_hard)
        & validation_rules["behavior_flag_count"].ge(2),
    ),
    summarize_rule(
        "Score >= T_hard AND >=3 behavior flags",
        validation_rules["score"].ge(t_hard)
        & validation_rules["behavior_flag_count"].ge(3),
    ),
    summarize_rule(
        "T_medium <= score < T_hard",
        validation_rules["score"].ge(t_medium)
        & validation_rules["score"].lt(t_hard),
    ),
    summarize_rule(
        "T_medium <= score < T_hard AND >=2 behavior flags",
        validation_rules["score"].ge(t_medium)
        & validation_rules["score"].lt(t_hard)
        & validation_rules["behavior_flag_count"].ge(2),
    ),
    summarize_rule(
        "Medium band: new merchant AND amount spike",
        validation_rules["score"].ge(t_medium)
        & validation_rules["score"].lt(t_hard)
        & validation_rules["is_new_merchant_for_customer"]
        & validation_rules["amount_spike"],
    ),
    summarize_rule(
        "Medium band: history uncertainty AND risky category/time",
        validation_rules["score"].ge(t_medium)
        & validation_rules["score"].lt(t_hard)
        & validation_rules["history_uncertainty"]
        & validation_rules["risky_category_or_time"],
    ),
    summarize_rule(
        "Medium band: interval extreme AND risky category/time",
        validation_rules["score"].ge(t_medium)
        & validation_rules["score"].lt(t_hard)
        & validation_rules["interval_extreme"]
        & validation_rules["risky_category_or_time"],
    ),
    summarize_rule(
        "T_low <= score < T_medium AND >=2 behavior flags",
        validation_rules["score"].ge(t_low)
        & validation_rules["score"].lt(t_medium)
        & validation_rules["behavior_flag_count"].ge(2),
    ),
    summarize_rule(
        "T_low <= score < T_medium AND >=3 behavior flags",
        validation_rules["score"].ge(t_low)
        & validation_rules["score"].lt(t_medium)
        & validation_rules["behavior_flag_count"].ge(3),
    ),
    summarize_rule(
        "Score < T_low",
        validation_rules["score"].lt(t_low),
    ),
]
rule_table = pd.DataFrame(rule_rows)
rule_table.to_csv(RULE_PATH, index=False)

print("THRESHOLDS")
print(threshold_table.to_string(index=False))
print("\nEXCLUSIVE SCORE BANDS")
print(band_table.to_string(index=False))
print("\nRULE CANDIDATES")
print(rule_table.to_string(index=False))
print("\nBEHAVIOR BOUNDARIES")
print(behavior_thresholds.to_string(index=False))
print(f"\nValidation rows: {len(y_validation):,}; fraud cases: {int(y_validation.sum()):,}")
