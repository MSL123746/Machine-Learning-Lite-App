import os
import random
from datetime import datetime
import pandas as pd

RANDOM_SEED = 42
ROW_COUNT = 300
OUTPUT_FILENAME = "Customer Purchase Behavior.csv"
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
    target_plan = build_target_plan(rng)

    for row_number, desired_target in enumerate(target_plan, start=1):
        for _ in range(120):
            source_archetype = choose_source_archetype(rng, desired_target)
            row = build_customer_row(rng, source_archetype, row_number)
            if rng.random() < OVERLAP_PROFILE_RATE:
                row = inject_behavior_overlap(rng, row, desired_target)
            hidden_biases = build_hidden_class_biases(rng, desired_target)
            target_value, _ = assign_target(row, hidden_biases=hidden_biases)
            visible_target, _, visible_scores = assign_target(row, return_scores=True)
            ranked_scores = sorted(visible_scores.items(), key=lambda item: item[1], reverse=True)
            score_margin = ranked_scores[0][1] - ranked_scores[1][1]
            keep_borderline_label = (
                visible_target != desired_target
                and desired_target in visible_scores
                and score_margin <= BORDERLINE_SCORE_MARGIN_MAX
                and rng.random() < BORDERLINE_KEEP_RATE
            )
            if target_value == desired_target or keep_borderline_label:
                row["Target-Value"] = desired_target
                row["Target-Ref"] = TARGET_REFERENCE[desired_target]
                rows.append(row)
                break
        else:
            row["Target-Value"] = desired_target
            row["Target-Ref"] = TARGET_REFERENCE[desired_target]
            rows.append(row)

    dataset = pd.DataFrame(rows)
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
    return sanitize_dataframe(dataset)


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
    dataset = generate_dataset()
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


if __name__ == "__main__":
    main()