import json

import joblib
import pandas as pd

from src.config import load_config, PROJECT_ROOT
from src.features.build import add_features


def load_model():
    config = load_config()
    model_dir = PROJECT_ROOT / config["paths"]["model_dir"]
    pipe = joblib.load(model_dir / "model.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    with open(model_dir / "schema.json") as f:
        schema = json.load(f)
    return pipe, le, schema


def predict_one(user_input: dict) -> dict:
    """Predict from a partial feature dict; missing fields use defaults."""
    pipe, le, schema = load_model()
    defaults = dict(schema.get("defaults", {}))

    # Merge user input over defaults
    row = {**defaults, **user_input}

    # Handle date column
    date_col = schema.get("date_column")
    if date_col and date_col not in user_input:
        row[date_col] = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = pd.DataFrame([row])
    df = add_features(df, date_column=date_col)

    pred_idx = int(pipe.predict(df)[0])
    pred_label = le.inverse_transform([pred_idx])[0]

    result = {schema["target"]: str(pred_label)}

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(df)[0]
        result["probabilities"] = {
            str(cls): round(float(p), 4)
            for cls, p in zip(le.classes_, proba)
        }

    return result


if __name__ == "__main__":
    # ---------- EDIT THIS WITH YOUR OWN INPUT ----------
    new_user = {
        "age": 28,
        "gender": "Female",
        "country": "USA",
        "profession": "Marketer",
        "platforms_used_count": 4,
        "daily_usage_hours": 5.5,
        "sessions_per_day": 12,
        "avg_session_duration_min": 28.0,
        "preferred_device": "Smartphone",
        "peak_usage_time": "Night (9pm-12am)",
        "followers_count": 85000,
        "following_count": 300,
        "posts_per_week": 15,
        "likes_given_per_day": 60,
        "comments_per_day": 25,
        "shares_per_day": 10,
        "dms_sent_per_day": 20,
        "preferred_content_type": "Videos",
        "scroll_speed": "Medium",
        "ad_click_rate": 0.22,
        "purchased_via_social_media": "Yes",
        "monthly_spend_via_social_usd": 120.0,
        "primary_purpose": "Business/Marketing",
        "sleep_disruption": "Mild impact",
        "self_reported_mental_health_score": 6.5,
        "screen_time_concern": "Somewhat",
        "notification_frequency": "Always On",
        "privacy_setting": "Public",
        "account_join_date": "2021-03-10",
        "mood_while_scrolling": "Inspired",
        "takes_social_media_breaks": "Occasionally",
        "primary_platform": "Instagram",
    }
    # --------------------------------------------------

    print("Input:")
    print(json.dumps(new_user, indent=2))
    print()
    print("Prediction:")
    print(json.dumps(predict_one(new_user), indent=2))
