import csv
import random
import datetime
import os
import pandas as pd

random.seed(42)

# Updated column set: product-category and CTA columns removed.
columns = [
    "Launched_Date",
    "Launch_Month",
    "Posts_Count",
    "Posting_Type_Video",
    "Posting_Type_Image",
    "Posting_Type_Announcement",
    "Posting_Type_Podcast",
    "Platform_Instagram",
    "Platform_Facebook",
    "Platform_SnapChat",
    "Platform_TikTok",
    "Platform_Twitter",
    "Promotion_Type_Organic",
    "Promotion_Type_Paid",
    "Age_Min",
    "Age_Max",
    "Gender_F",
    "Gender_M",
    "Product_Category_Apparel",
    "Product_Category_Shoes",
    "Product_Category_Cosmetics",
    "Product_Category_Home_Goods",
    "Product_Category_Accessories",
    "Campaign_Duration_Days",
    "Campaign Spend",
    "Revenue",
]

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

product_categories = [
    "Apparel",
    "Shoes",
    "Cosmetics",
    "Home_Goods",
    "Accessories",
]


def random_date(start, end):
    return start + datetime.timedelta(days=random.randint(0, (end - start).days))

def make_multi_hot(size, count_weights):
    values = [0] * size
    counts = [count for count in sorted(count_weights) if count <= size]
    weights = [count_weights[count] for count in counts]
    selected_count = random.choices(counts, weights=weights, k=1)[0]
    for idx in random.sample(range(size), selected_count):
        values[idx] = 1
    return values

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

start_date = datetime.date(2022, 8, 1)
end_date = datetime.date(2024, 4, 30)

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
now = datetime.datetime.now()
date_str = now.strftime("_%m-%d-%Y_%H-%M-%S")
csv_path = os.path.join(
    downloads_folder,
    f"LinearRegressionDataset_for_Campaign_Marketing_Optimization{date_str}.csv",
)

safe_columns = clean_columns(columns)
rows = []

for _ in range(300):
    date = random_date(start_date, end_date)
    launch_month = months[date.month - 1]

    campaign_duration_days = random.randint(3, 45)
    min_posts = max(1, campaign_duration_days // 4)
    max_posts = max(3, min(20, campaign_duration_days // 2 + 2))
    posts_count = random.randint(min_posts, max_posts)

    posting_types = make_multi_hot(
        size=4,
        count_weights={1: 0.25, 2: 0.30, 3: 0.25, 4: 0.20},
    )

    platforms = make_multi_hot(
        size=5,
        count_weights={1: 0.15, 2: 0.30, 3: 0.25, 4: 0.20, 5: 0.10},
    )

    if random.random() < 0.20:
        promo_types = [1, 1]
    elif random.random() < 0.60:
        promo_types = [1, 0]
    else:
        promo_types = [0, 1]

    age_min = random.randint(18, 55)
    age_max = random.randint(age_min + 5, min(age_min + 35, 75))

    if random.random() < 0.20:
        gender_f, gender_m = 1, 1
    elif random.random() < 0.50:
        gender_f, gender_m = 1, 0
    else:
        gender_f, gender_m = 0, 1

    product_category_flags = make_multi_hot(
        size=len(product_categories),
        count_weights={1: 0.70, 2: 0.20, 3: 0.10},
    )

    spend = random.randint(250, 4000)
    posting_bonus = (
        posting_types[0] * 320
        + posting_types[1] * 180
        + posting_types[2] * 140
        + posting_types[3] * 260
    )
    platform_bonus = (
        platforms[0] * 210
        + platforms[1] * 170
        + platforms[2] * 130
        + platforms[3] * 240
        + platforms[4] * 160
    )
    product_bonus = (
        product_category_flags[0] * 180
        + product_category_flags[1] * 240
        + product_category_flags[2] * 220
        + product_category_flags[3] * 160
        + product_category_flags[4] * 140
    )
    audience_bonus = (age_max - age_min) * 20 + (250 if gender_f and gender_m else 0)
    noise = random.randint(-250, 250)
    duration_multiplier = (
        0.18 if campaign_duration_days <= 2
        else 0.40 if campaign_duration_days <= 5
        else 0.70 if campaign_duration_days <= 10
        else 1.0
    )
    post_multiplier = min(1.0, 0.35 + posts_count * 0.08)
    campaign_efficiency = min(1.0, duration_multiplier * post_multiplier + 0.10)

    revenue = int(max(
        0,
        spend * 4.2 * campaign_efficiency
        + posts_count * 135 * post_multiplier
        + campaign_duration_days * 130 * duration_multiplier
        + posting_bonus * campaign_efficiency
        + platform_bonus * campaign_efficiency
        + product_bonus
        + promo_types[1] * 900 * campaign_efficiency
        + audience_bonus
        + noise,
    ))

    row = [
        date.isoformat(),
        launch_month,
        posts_count,
        *posting_types,
        *platforms,
        *promo_types,
        age_min,
        age_max,
        gender_f,
        gender_m,
        *product_category_flags,
        campaign_duration_days,
        spend,
        revenue,
    ]
    if len(row) != len(safe_columns):
        raise ValueError(f"Row has {len(row)} values but expected {len(safe_columns)}.")
    rows.append(row)

df = pd.DataFrame(rows, columns=safe_columns)
df = sanitize_dataframe(df)
df.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

reloaded_df = pd.read_csv(csv_path)
reloaded_df = sanitize_dataframe(reloaded_df)
reloaded_df.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

unnamed_cols = [col for col in reloaded_df.columns if str(col).lower().startswith("unnamed:")]
missing_total = int(reloaded_df.isna().sum().sum())
if unnamed_cols:
    raise ValueError(f"Unexpected unnamed columns found: {unnamed_cols}")
if missing_total != 0:
    raise ValueError(f"Unexpected missing values found: {missing_total}")

print(f"File '{csv_path}' created.")