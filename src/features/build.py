import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    df = df.copy()

    # --- 1. Convert date column to account_age_days ---
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        ref_date = df[date_column].max()
        df["account_age_days"] = (ref_date - df[date_column]).dt.days
        df = df.drop(columns=[date_column])
        print(f"[features] Converted '{date_column}' -> account_age_days "
              f"(ref: {ref_date.date()})")

    # --- 2. Total daily engagement ---
    eng_cols = [c for c in [
        "likes_given_per_day", "comments_per_day",
        "shares_per_day", "dms_sent_per_day"
    ] if c in df.columns]
    if eng_cols:
        df["total_engagement_per_day"] = df[eng_cols].sum(axis=1)
        df["engagement_per_hour"] = (
            df["total_engagement_per_day"] / (df["daily_usage_hours"] + 1e-6)
        )

    # --- 3. Session intensity ---
    if {"daily_usage_hours", "sessions_per_day"} <= set(df.columns):
        df["avg_session_hours"] = (
            df["daily_usage_hours"] / (df["sessions_per_day"] + 1e-6)
        )
        df["sessions_per_hour"] = (
            df["sessions_per_day"] / (df["daily_usage_hours"] + 1e-6)
        )

    # --- 4. Follower/following ratios ---
    if {"followers_count", "following_count"} <= set(df.columns):
        df["follower_following_ratio"] = (
            df["followers_count"] / (df["following_count"] + 1)
        )
        df["follower_per_post"] = (
            df["followers_count"] / (df["posts_per_week"] + 1)
        )

    # --- 5. Engagement-to-follower ratio ---
    if {"total_engagement_per_day", "followers_count"} <= set(df.columns):
        df["engagement_rate"] = (
            df["total_engagement_per_day"] / (df["followers_count"] + 1)
        )

    print(f"[features] After engineering: {df.shape[1]} columns")
    return df
