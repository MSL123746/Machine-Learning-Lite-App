import os
import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
OUTPUT_FILENAME = "Customer Purchase Behavior MCC.csv"
DESIRED_ROW_COUNT = 300
TRAIN_FRACTION = 0.70
MIN_CLASS_COUNT = 75
NEIGHBOR_ARCHETYPE_RATE = 0.30
OVERLAP_PROFILE_RATE = 0.28
BORDERLINE_KEEP_RATE = 0.24
BORDERLINE_SCORE_MARGIN_MAX = 1.8
HIDDEN_BEHAVIOR_BIAS_MIN = 1.4
HIDDEN_BEHAVIOR_BIAS_MAX = 3.0

TARGET_REFERENCE = {
    0: "Considering Purchase",
    1: "Discount-Responsive",
    2: "Ready to Buy",
    3: "Advocate Likely",
}

# Rows = actual class, columns = predicted class for the 30% validation set
VALIDATION_CONFUSION_MATRIX_TARGETS = [
    [17, 1, 0, 0],
    [1, 22, 1, 1],
    [0, 0, 18, 5],
    [0, 0, 6, 18],
]

VALIDATION_ROW_COUNT = sum(sum(row) for row in VALIDATION_CONFUSION_MATRIX_TARGETS)
ROW_COUNT = DESIRED_ROW_COUNT

TARGET_CLASS_WEIGHTS = {
    0: 0.25,
    1: 0.25,
    2: 0.25,
    3: 0.25,
}

PRODUCT_CATEGORY_COLUMNS = [
    "Apparel-Orders-12Mos",
    "Home-Goods-Orders-12Mos",
    "Electronics-Orders-12Mos",
]

PRODUCT_CATEGORY_WEIGHTS_BY_TARGET = {
    0: {
        "Apparel-Orders-12Mos": 0.25,
        "Home-Goods-Orders-12Mos": 0.55,
        "Electronics-Orders-12Mos": 0.20,
    },
    1: {
        "Apparel-Orders-12Mos": 0.55,
        "Home-Goods-Orders-12Mos": 0.30,
        "Electronics-Orders-12Mos": 0.15,
    },
    2: {
        "Apparel-Orders-12Mos": 0.15,
        "Home-Goods-Orders-12Mos": 0.20,
        "Electronics-Orders-12Mos": 0.65,
    },
    3: {
        "Apparel-Orders-12Mos": 0.25,
        "Home-Goods-Orders-12Mos": 0.15,
        "Electronics-Orders-12Mos": 0.60,
    },
}


def clean_columns(raw_columns):
    cleaned = [str(col).strip().replace("\ufeff", "") for col in raw_columns]
    if any(col == "" for col in cleaned):
        raise ValueError("Blank column names are not allowed.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Duplicate column names are not allowed.")
    return cleaned


def sanitize_dataframe(df):
    cleaned_df = df.copy()
    cleaned_df.columns = clean_columns(cleaned_df.columns)
    cleaned_df = cleaned_df.loc[:, [str(col).strip() != "" for col in cleaned_df.columns]]
    cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.astype(str).str.lower().str.startswith("unnamed:")]
    cleaned_df = cleaned_df.dropna(axis=1, how="all").reset_index(drop=True)

    for col in cleaned_df.columns:
        if cleaned_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(0)
            else:
                cleaned_df[col] = cleaned_df[col].fillna("Unknown")

    return cleaned_df


def bounded_int(value, lower, upper):
    return max(lower, min(upper, int(round(value))))


def bounded_float(value, lower, upper, digits=3):
    return round(max(lower, min(upper, float(value))), digits)


def weighted_choice(rng, weights):
    total = sum(weights.values())
    pick = rng.uniform(0, total)
    running = 0.0
    for key, weight in weights.items():
        running += weight
        if pick <= running:
            return key
    return next(iter(weights))


def build_target_plan(rng):
    target_plan = []
    for target_value in TARGET_REFERENCE:
        target_plan.extend([target_value] * MIN_CLASS_COUNT)

    remaining_rows = ROW_COUNT - len(target_plan)
    for _ in range(remaining_rows):
        target_plan.append(weighted_choice(rng, TARGET_CLASS_WEIGHTS))

    rng.shuffle(target_plan)
    return target_plan


def build_channel_metrics(rng, archetype):
    if archetype == 0:
        return {
            "Email-Responses-12Mos": rng.randint(1, 4),
            "Website-Logins-30D": rng.randint(2, 6),
            "Social-Media-Responses-12Mos": rng.randint(0, 2),
            "Paid-Ads-Responses-12Mos": rng.randint(0, 3),
            "Website-Visit-Time-Min": rng.randint(4, 18),
        }
    if archetype == 1:
        return {
            "Email-Responses-12Mos": rng.randint(3, 8),
            "Website-Logins-30D": rng.randint(3, 8),
            "Social-Media-Responses-12Mos": rng.randint(1, 3),
            "Paid-Ads-Responses-12Mos": rng.randint(2, 6),
            "Website-Visit-Time-Min": rng.randint(6, 22),
        }
    if archetype == 2:
        return {
            "Email-Responses-12Mos": rng.randint(2, 6),
            "Website-Logins-30D": rng.randint(5, 12),
            "Social-Media-Responses-12Mos": rng.randint(1, 4),
            "Paid-Ads-Responses-12Mos": rng.randint(0, 3),
            "Website-Visit-Time-Min": rng.randint(12, 35),
        }
    return {
        "Email-Responses-12Mos": rng.randint(1, 5),
        "Website-Logins-30D": rng.randint(4, 10),
        "Social-Media-Responses-12Mos": rng.randint(3, 8),
        "Paid-Ads-Responses-12Mos": rng.randint(0, 2),
        "Website-Visit-Time-Min": rng.randint(10, 28),
    }


def build_hidden_class_biases(rng, desired_target):
    biases = {target_value: rng.uniform(-0.35, 0.35) for target_value in TARGET_REFERENCE}
    biases[desired_target] += rng.uniform(HIDDEN_BEHAVIOR_BIAS_MIN, HIDDEN_BEHAVIOR_BIAS_MAX)
    competing_targets = [target_value for target_value in TARGET_REFERENCE if target_value != desired_target]
    competitor_target = rng.choice(competing_targets)
    biases[competitor_target] += rng.uniform(0.0, 0.75)
    return biases


def choose_source_archetype(rng, desired_target):
    if rng.random() >= NEIGHBOR_ARCHETYPE_RATE:
        return desired_target
    neighbors = []
    if desired_target > min(TARGET_REFERENCE):
        neighbors.append(desired_target - 1)
    if desired_target < max(TARGET_REFERENCE):
        neighbors.append(desired_target + 1)
    return rng.choice(neighbors) if neighbors else desired_target


def build_product_category_counts(rng, archetype, total_orders, row):
    base_weights = PRODUCT_CATEGORY_WEIGHTS_BY_TARGET.get(
        archetype,
        PRODUCT_CATEGORY_WEIGHTS_BY_TARGET[0],
    )
    total_orders = max(0, int(total_orders))

    avg_order_value = float(row.get("Avg-Order-Value", 0))
    discount_usage = float(row.get("Discount-Usage-Rate", 0.0))
    days_since_last_purchase = float(row.get("Days-Since-Last-Purchase", 180))
    referral_count = float(row.get("Referral-Count", 0))
    coupons_issued = float(row.get("Coupons-Issued-12Mos", 0))
    website_visit_time = float(row.get("Website-Visit-Time-Min", 1))
    paid_ads_responses = float(row.get("Paid-Ads-Responses-12Mos", 0))

    value_scale = max(0.0, min(1.0, (avg_order_value - 20.0) / 180.0))
    recency_scale = max(0.0, min(1.0, 1.0 - (days_since_last_purchase / 180.0)))
    discount_scale = max(0.0, min(1.0, discount_usage))
    referral_scale = max(0.0, min(1.0, referral_count / 8.0))
    coupon_scale = max(0.0, min(1.0, coupons_issued / 20.0))
    visit_time_scale = max(0.0, min(1.0, website_visit_time / 90.0))
    paid_ads_scale = max(0.0, min(1.0, paid_ads_responses / 20.0))

    dynamic_multipliers = {
        "Apparel-Orders-12Mos": max(
            0.15,
            1.0
            + discount_scale * 0.90
            + paid_ads_scale * 0.35
            + coupon_scale * 0.25
            - value_scale * 0.45,
        ),
        "Home-Goods-Orders-12Mos": max(
            0.15,
            1.0
            + coupon_scale * 0.40
            + recency_scale * 0.35
            + (1.0 - value_scale) * 0.20
            + visit_time_scale * 0.10,
        ),
        "Electronics-Orders-12Mos": max(
            0.15,
            1.0
            + value_scale * 1.10
            + visit_time_scale * 0.50
            + referral_scale * 0.35
            - discount_scale * 0.25,
        ),
    }

    effective_weights = {
        category: base_weights[category] * dynamic_multipliers[category]
        for category in PRODUCT_CATEGORY_COLUMNS
    }

    counts = {category_column: 0 for category_column in PRODUCT_CATEGORY_COLUMNS}
    for _ in range(total_orders):
        selected_category = weighted_choice(rng, effective_weights)
        counts[selected_category] += 1
    return counts


def rebalance_product_category_counts(rng, row, archetype):
    counts = build_product_category_counts(rng, archetype, row.get("Orders-12Mos", 0), row)
    row.update(counts)
    return row


def build_customer_row(rng, target_value, row_number):
    if target_value == 0:
        row = {
            "CustomerID": 1000 + row_number,
            "Tenure-Months": rng.randint(3, 30),
            "Orders-12Mos": rng.randint(2, 6),
            "Avg-Order-Value": rng.randint(25, 85),
            "Discount-Usage-Rate": bounded_float(rng.uniform(0.15, 0.55), 0.0, 1.0),
            "Days-Since-Last-Purchase": rng.randint(12, 45),
            "Referral-Count": rng.randint(0, 1),
            "Coupons-Issued-12Mos": rng.randint(1, 5),
        }
    elif target_value == 1:
        row = {
            "CustomerID": 1000 + row_number,
            "Tenure-Months": rng.randint(6, 36),
            "Orders-12Mos": rng.randint(3, 8),
            "Avg-Order-Value": rng.randint(30, 90),
            "Discount-Usage-Rate": bounded_float(rng.uniform(0.45, 0.95), 0.0, 1.0),
            "Days-Since-Last-Purchase": rng.randint(5, 25),
            "Referral-Count": rng.randint(0, 2),
            "Coupons-Issued-12Mos": rng.randint(4, 12),
        }
    elif target_value == 2:
        row = {
            "CustomerID": 1000 + row_number,
            "Tenure-Months": rng.randint(8, 60),
            "Orders-12Mos": rng.randint(5, 12),
            "Avg-Order-Value": rng.randint(65, 150),
            "Discount-Usage-Rate": bounded_float(rng.uniform(0.00, 0.25), 0.0, 1.0),
            "Days-Since-Last-Purchase": rng.randint(3, 20),
            "Referral-Count": rng.randint(0, 3),
            "Coupons-Issued-12Mos": rng.randint(0, 4),
        }
    else:
        row = {
            "CustomerID": 1000 + row_number,
            "Tenure-Months": rng.randint(12, 84),
            "Orders-12Mos": rng.randint(6, 14),
            "Avg-Order-Value": rng.randint(60, 160),
            "Discount-Usage-Rate": bounded_float(rng.uniform(0.00, 0.35), 0.0, 1.0),
            "Days-Since-Last-Purchase": rng.randint(1, 20),
            "Referral-Count": rng.randint(2, 8),
            "Coupons-Issued-12Mos": rng.randint(0, 3),
        }

    row.update(build_channel_metrics(rng, target_value))
    row = add_overlap_noise(rng, row, target_value)
    row = rebalance_product_category_counts(rng, row, target_value)
    return row


def inject_behavior_overlap(rng, row, desired_target):
    row = row.copy()
    if desired_target == 0:
        row["Orders-12Mos"] = bounded_int(row["Orders-12Mos"] + rng.randint(1, 2), 1, 20)
        row["Days-Since-Last-Purchase"] = bounded_int(row["Days-Since-Last-Purchase"] - rng.randint(4, 12), 1, 180)
        row["Avg-Order-Value"] = bounded_int(row["Avg-Order-Value"] + rng.randint(5, 15), 10, 250)
        row["Coupons-Issued-12Mos"] = bounded_int(row["Coupons-Issued-12Mos"] + rng.randint(1, 3), 0, 20)
    elif desired_target == 1:
        row["Discount-Usage-Rate"] = bounded_float(row["Discount-Usage-Rate"] - rng.uniform(0.08, 0.16), 0.0, 1.0)
        row["Avg-Order-Value"] = bounded_int(row["Avg-Order-Value"] + rng.randint(8, 20), 10, 250)
        row["Coupons-Issued-12Mos"] = bounded_int(row["Coupons-Issued-12Mos"] - rng.randint(1, 3), 0, 20)
    elif desired_target == 2:
        row["Discount-Usage-Rate"] = bounded_float(row["Discount-Usage-Rate"] + rng.uniform(0.06, 0.14), 0.0, 1.0)
        row["Referral-Count"] = bounded_int(row["Referral-Count"] + rng.randint(0, 2), 0, 8)
        row["Website-Logins-30D"] = bounded_int(row["Website-Logins-30D"] + rng.randint(1, 4), 0, 30)
        row["Website-Visit-Time-Min"] = bounded_int(row["Website-Visit-Time-Min"] + rng.randint(3, 10), 1, 90)
    else:
        row["Referral-Count"] = bounded_int(max(1, row["Referral-Count"] - rng.randint(0, 2)), 0, 8)
        row["Discount-Usage-Rate"] = bounded_float(row["Discount-Usage-Rate"] + rng.uniform(0.04, 0.10), 0.0, 1.0)
        row["Social-Media-Responses-12Mos"] = bounded_int(row["Social-Media-Responses-12Mos"] + rng.randint(1, 3), 0, 20)

    if rng.random() < 0.22:
        channel_metrics = build_channel_metrics(rng, rng.choice(list(TARGET_REFERENCE.keys())))
        for key, value in channel_metrics.items():
            row[key] = value

    row = rebalance_product_category_counts(rng, row, desired_target)
    return row


def add_overlap_noise(rng, row, target_value):
    row = row.copy()
    row["Orders-12Mos"] = bounded_int(row["Orders-12Mos"] + rng.randint(-1, 1), 1, 20)
    row["Avg-Order-Value"] = bounded_int(row["Avg-Order-Value"] + rng.randint(-8, 8), 10, 250)
    row["Days-Since-Last-Purchase"] = bounded_int(row["Days-Since-Last-Purchase"] + rng.randint(-8, 8), 1, 180)
    row["Discount-Usage-Rate"] = bounded_float(row["Discount-Usage-Rate"] + rng.uniform(-0.06, 0.06), 0.0, 1.0)
    row["Coupons-Issued-12Mos"] = bounded_int(row["Coupons-Issued-12Mos"] + rng.randint(-1, 2), 0, 20)
    row["Email-Responses-12Mos"] = bounded_int(row["Email-Responses-12Mos"] + rng.randint(-1, 2), 0, 24)
    row["Website-Logins-30D"] = bounded_int(row["Website-Logins-30D"] + rng.randint(-1, 2), 0, 30)
    row["Social-Media-Responses-12Mos"] = bounded_int(row["Social-Media-Responses-12Mos"] + rng.randint(-1, 2), 0, 20)
    row["Paid-Ads-Responses-12Mos"] = bounded_int(row["Paid-Ads-Responses-12Mos"] + rng.randint(-1, 2), 0, 20)
    row["Website-Visit-Time-Min"] = bounded_int(row["Website-Visit-Time-Min"] + rng.randint(-3, 4), 1, 90)

    if rng.random() < 0.18 and target_value in (1, 2, 3):
        row["Referral-Count"] = bounded_int(row["Referral-Count"] + 1, 0, 8)

    return row


def apply_prediction_signature(row, predicted_target, rng):
    row = row.copy()
    if predicted_target == 0:
        row["Avg-Order-Value"] = bounded_int(42 + rng.randint(-3, 3), 10, 250)
        row["Discount-Usage-Rate"] = bounded_float(0.22 + rng.uniform(-0.03, 0.03), 0.0, 1.0)
        row["Referral-Count"] = bounded_int(1 + rng.randint(0, 1), 0, 8)
    elif predicted_target == 1:
        row["Avg-Order-Value"] = bounded_int(68 + rng.randint(-4, 4), 10, 250)
        row["Discount-Usage-Rate"] = bounded_float(0.74 + rng.uniform(-0.04, 0.04), 0.0, 1.0)
        row["Referral-Count"] = bounded_int(1 + rng.randint(0, 1), 0, 8)
    elif predicted_target == 2:
        row["Avg-Order-Value"] = bounded_int(108 + rng.randint(-3, 3), 10, 250)
        row["Discount-Usage-Rate"] = bounded_float(0.08 + rng.uniform(-0.02, 0.02), 0.0, 1.0)
        row["Referral-Count"] = bounded_int(1 + rng.randint(0, 1), 0, 8)
        row["Social-Media-Responses-12Mos"] = bounded_int(1 + rng.randint(0, 1), 0, 20)
        row["Website-Visit-Time-Min"] = bounded_int(34 + rng.randint(-3, 3), 1, 90)
        row["Electronics-Orders-12Mos"] = bounded_int(max(row.get("Electronics-Orders-12Mos", 0), 6 + rng.randint(0, 2)), 0, 25)
    else:
        row["Avg-Order-Value"] = bounded_int(142 + rng.randint(-3, 3), 10, 250)
        row["Discount-Usage-Rate"] = bounded_float(0.22 + rng.uniform(-0.02, 0.02), 0.0, 1.0)
        row["Referral-Count"] = bounded_int(7 + rng.randint(0, 1), 0, 8)
        row["Social-Media-Responses-12Mos"] = bounded_int(7 + rng.randint(0, 2), 0, 20)
        row["Website-Visit-Time-Min"] = bounded_int(20 + rng.randint(-2, 2), 1, 90)
    return row


def simulate_app_like_confusion(dataset):
    feature_cols = [
        c for c in dataset.columns
        if c not in {"Target-Value", "Target-Ref"}
    ]
    X = dataset[feature_cols].copy()
    y = dataset["Target-Value"].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=TRAIN_FRACTION,
        random_state=42,
    )

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    if numeric_cols:
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_train[numeric_cols] = X_train[numeric_cols].astype(float)
        X_val[numeric_cols] = X_val[numeric_cols].astype(float)
        scaler = StandardScaler()
        X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val.loc[:, numeric_cols] = scaler.transform(X_val[numeric_cols])

    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=sorted(TARGET_REFERENCE.keys()))
    return pd.DataFrame(cm, index=sorted(TARGET_REFERENCE.keys()), columns=sorted(TARGET_REFERENCE.keys()))


def assign_target(row, hidden_biases=None, return_scores=False):
    orders = row["Orders-12Mos"]
    avg_order_value = row["Avg-Order-Value"]
    discount_usage = row["Discount-Usage-Rate"]
    days_since_last_purchase = row["Days-Since-Last-Purchase"]
    referral_count = row["Referral-Count"]
    tenure_months = row["Tenure-Months"]
    coupons_issued = row["Coupons-Issued-12Mos"]
    email_responses = row["Email-Responses-12Mos"]
    website_logins = row["Website-Logins-30D"]
    social_media_responses = row["Social-Media-Responses-12Mos"]
    paid_ads_responses = row["Paid-Ads-Responses-12Mos"]
    website_visit_time = row["Website-Visit-Time-Min"]
    product_apparel_orders = row["Apparel-Orders-12Mos"]
    product_home_goods_orders = row["Home-Goods-Orders-12Mos"]
    product_electronics_orders = row["Electronics-Orders-12Mos"]
    total_category_orders = max(
        1,
        product_apparel_orders + product_home_goods_orders + product_electronics_orders,
    )
    apparel_share = product_apparel_orders / total_category_orders
    home_goods_share = product_home_goods_orders / total_category_orders
    electronics_share = product_electronics_orders / total_category_orders

    scores = {
        0: (
            min(days_since_last_purchase, 70) * 0.12
            + max(0, 7 - orders) * 1.05
            + max(0, 0.60 - discount_usage) * 3.0
            + max(0, 95 - avg_order_value) * 0.03
            + max(0, 2 - referral_count) * 0.7
            + coupons_issued * 0.18
            + paid_ads_responses * 0.35
            + email_responses * 0.20
            + home_goods_share * 10.0
        ),
        1: (
            discount_usage * 10.5
            + max(0, 32 - abs(days_since_last_purchase - 18)) * 0.14
            + max(0, 8 - abs(orders - 5)) * 0.65
            + max(0, 95 - abs(avg_order_value - 60)) * 0.025
            + max(0, 3 - referral_count) * 0.45
            + coupons_issued * 0.42
            + paid_ads_responses * 0.90
            + email_responses * 0.35
            + apparel_share * 10.0
        ),
        2: (
            max(0, 0.35 - discount_usage) * 12.0
            + orders * 0.9
            + avg_order_value * 0.05
            + max(0, 28 - days_since_last_purchase) * 0.18
            + tenure_months * 0.03
            + max(0, 5 - coupons_issued) * 0.22
            + website_logins * 0.22
            + website_visit_time * 0.07
            + electronics_share * 10.0
        ),
        3: (
            referral_count * 2.5
            + orders * 0.6
            + max(0, 24 - days_since_last_purchase) * 0.17
            + tenure_months * 0.06
            + avg_order_value * 0.025
            + max(0, 4 - coupons_issued) * 0.28
            + social_media_responses * 0.75
            + website_logins * 0.08
            + (1 - min(1, paid_ads_responses)) * 0.70
            + (apparel_share * 4.0 + home_goods_share * 2.0 + electronics_share * 10.0)
        ),
    }

    if hidden_biases is not None:
        for target_value, bias in hidden_biases.items():
            scores[target_value] += bias

    ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    target_value = ranked_scores[0][0]
    if return_scores:
        return target_value, TARGET_REFERENCE[target_value], scores
    return target_value, TARGET_REFERENCE[target_value]


def generate_dataset():
    rng = random.Random(RANDOM_SEED)
    rows = []

    all_indices = list(range(DESIRED_ROW_COUNT))
    train_indices, validation_indices = train_test_split(
        all_indices,
        train_size=TRAIN_FRACTION,
        random_state=42,
    )
    validation_indices_sorted = sorted(validation_indices)
    train_indices_sorted = sorted(train_indices)

    if len(validation_indices_sorted) != VALIDATION_ROW_COUNT:
        raise ValueError(
            "Validation split row count mismatch: expected {0}, got {1}.".format(
                VALIDATION_ROW_COUNT,
                len(validation_indices_sorted),
            )
        )

    validation_pairs = []
    for actual_target, predicted_counts in enumerate(VALIDATION_CONFUSION_MATRIX_TARGETS):
        for predicted_target, cell_count in enumerate(predicted_counts):
            for _ in range(cell_count):
                validation_pairs.append((actual_target, predicted_target))

    if len(validation_pairs) != len(validation_indices_sorted):
        raise ValueError("Validation pair count does not match validation split size.")

    training_targets = []
    for i in range(len(train_indices_sorted)):
        training_targets.append(i % len(TARGET_REFERENCE))
    rng.shuffle(training_targets)

    row_map = {}

    # Training rows: keep actual == predicted so model learns stable signatures per class.
    for idx, row_index in enumerate(train_indices_sorted):
        target_class = training_targets[idx]
        row = build_customer_row(rng, target_class, row_index + 1)
        row = apply_prediction_signature(row, target_class, rng)
        row["Target-Value"] = target_class
        row["Target-Ref"] = TARGET_REFERENCE[target_class]
        row["Predicted-Target"] = target_class
        row["Predicted-Ref"] = TARGET_REFERENCE[target_class]
        row_map[row_index] = row

    # Validation rows: enforce the exact requested confusion-matrix cells.
    for idx, row_index in enumerate(validation_indices_sorted):
        actual_target, predicted_target = validation_pairs[idx]
        row = build_customer_row(rng, predicted_target, row_index + 1)
        row = apply_prediction_signature(row, predicted_target, rng)
        row["Target-Value"] = actual_target
        row["Target-Ref"] = TARGET_REFERENCE[actual_target]
        row["Predicted-Target"] = predicted_target
        row["Predicted-Ref"] = TARGET_REFERENCE[predicted_target]
        row_map[row_index] = row

    for row_index in sorted(row_map.keys()):
        rows.append(row_map[row_index])

    full_dataset = pd.DataFrame(rows)

    validation_df = full_dataset.loc[validation_indices_sorted].copy()
    matrix_df = pd.crosstab(
        validation_df["Target-Value"],
        validation_df["Predicted-Target"],
        dropna=False,
    ).reindex(index=sorted(TARGET_REFERENCE.keys()), columns=sorted(TARGET_REFERENCE.keys()), fill_value=0)
    expected_df = pd.DataFrame(
        VALIDATION_CONFUSION_MATRIX_TARGETS,
        index=sorted(TARGET_REFERENCE.keys()),
        columns=sorted(TARGET_REFERENCE.keys()),
    )
    if not matrix_df.equals(expected_df):
        raise ValueError("Generated dataset confusion matrix does not match target matrix.")

    dataset = full_dataset
    ordered_columns = [
        "CustomerID",
        "Avg-Order-Value",
        "Tenure-Months",
        "Orders-12Mos",
        "Coupons-Issued-12Mos",
        "Discount-Usage-Rate",
        "Days-Since-Last-Purchase",
        "Referral-Count",
        "Website-Logins-30D",
        "Website-Visit-Time-Min",
        "Paid-Ads-Responses-12Mos",
        "Apparel-Orders-12Mos",
        "Home-Goods-Orders-12Mos",
        "Electronics-Orders-12Mos",
        "Target-Value",
        "Target-Ref",
    ]
    dataset = dataset[ordered_columns]
    return sanitize_dataframe(dataset), matrix_df


def get_downloads_folder():
    downloads_folder = os.path.expanduser("~/Downloads")
    os.makedirs(downloads_folder, exist_ok=True)
    return downloads_folder


def get_script_folder():
    return os.path.dirname(os.path.abspath(__file__))


def write_csv_with_fallback(df, output_path):
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        return output_path
    except PermissionError:
        stem, ext = os.path.splitext(output_path)
        timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        fallback_path = stem + timestamp + ext
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        return fallback_path


def main():
    dataset, matrix_df = generate_dataset()
    if len(dataset) != ROW_COUNT:
        raise ValueError("Expected {0} rows but generated {1}.".format(ROW_COUNT, len(dataset)))
    missing_total = int(dataset.isna().sum().sum())
    if missing_total != 0:
        raise ValueError("Dataset still contains {0} missing values before save.".format(missing_total))
    downloads_output_path = os.path.join(get_downloads_folder(), OUTPUT_FILENAME)
    script_output_path = os.path.join(get_script_folder(), OUTPUT_FILENAME)

    downloads_output_path = write_csv_with_fallback(dataset, downloads_output_path)
    script_output_path = write_csv_with_fallback(dataset, script_output_path)

    reloaded_df = pd.read_csv(script_output_path)
    reloaded_df = sanitize_dataframe(reloaded_df)

    unnamed_cols = [col for col in reloaded_df.columns if str(col).lower().startswith("unnamed:")]
    missing_total = int(reloaded_df.isna().sum().sum())
    if unnamed_cols:
        raise ValueError("Unexpected unnamed columns found: {0}".format(unnamed_cols))
    if missing_total != 0:
        raise ValueError("Unexpected missing values found after save: {0}".format(missing_total))

    print("Saved dataset to: {0}".format(downloads_output_path))
    print("Saved dataset to: {0}".format(script_output_path))
    print("Generated rows: {0}".format(len(reloaded_df)))
    print("Missing values: {0}".format(missing_total))
    print("Class counts: {0}".format(reloaded_df["Target-Ref"].value_counts().to_dict()))

    print("Confusion matrix from generated dataset:")
    print(matrix_df)

    expected_validation_df = pd.DataFrame(
        VALIDATION_CONFUSION_MATRIX_TARGETS,
        index=sorted(TARGET_REFERENCE.keys()),
        columns=sorted(TARGET_REFERENCE.keys()),
    )
    print("Expected validation confusion matrix at 30% split ({0} rows):".format(VALIDATION_ROW_COUNT))
    print(expected_validation_df)

    simulated_model_df = simulate_app_like_confusion(dataset)
    print("Simulated app-like LogisticRegression confusion matrix (70/30 split):")
    print(simulated_model_df)


if __name__ == "__main__":
    main()