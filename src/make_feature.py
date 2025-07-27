import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def create_features(train_df, test_df, data_dir="/kaggle/input/atma20"):
    """
    公募応募予測のための特徴量を作成する関数

    Args:
        train_df: 学習用データ
        test_df: テスト用データ
        data_dir: データディレクトリのパス

    Returns:
        train_features, test_features: 特徴量が追加されたDataFrame
    """

    # 全社員IDを取得
    all_employee_ids = list(
        set(train_df["社員番号"].unique()) | set(test_df["社員番号"].unique())
    )

    # 各カテゴリを取得
    categories = train_df["category"].unique()

    # 各データを読み込み
    try:
        udemy_df = pd.read_csv(f"{data_dir}/udemy_activity.csv")
        print(f"Udemy data loaded: {len(udemy_df)} records")
    except Exception:
        print("Warning: Could not load udemy_activity.csv")
        udemy_df = pd.DataFrame()

    career_df = pd.read_csv(f"{data_dir}/career.csv")
    dx_df = pd.read_csv(f"{data_dir}/dx.csv")
    hr_df = pd.read_csv(f"{data_dir}/hr.csv")
    overtime_df = pd.read_csv(f"{data_dir}/overtime_work_by_month.csv")
    position_df = pd.read_csv(f"{data_dir}/position_history.csv")

    print(f"Career data: {len(career_df)} records")
    print(f"DX data: {len(dx_df)} records")
    print(f"HR data: {len(hr_df)} records")
    print(f"Overtime data: {len(overtime_df)} records")
    print(f"Position data: {len(position_df)} records")

    # 特徴量を格納するDataFrame
    features_list = []

    for emp_id in all_employee_ids:
        for category in categories:
            features = {"社員番号": emp_id, "category": category}

            # 1. Udemy関連特徴量
            if not udemy_df.empty:
                emp_udemy = udemy_df[udemy_df["社員番号"] == emp_id]
                features.update(create_udemy_features(emp_udemy))
            else:
                features.update(get_default_udemy_features())

            # 2. キャリア感特徴量
            emp_career = career_df[career_df["社員番号"] == emp_id]
            features.update(create_career_features(emp_career))

            # 3. DX研修特徴量
            emp_dx = dx_df[dx_df["社員番号"] == emp_id]
            features.update(create_dx_features(emp_dx))

            # 4. 人事施策特徴量
            emp_hr = hr_df[hr_df["社員番号"] == emp_id]
            features.update(create_hr_features(emp_hr))

            # 5. 残業時間特徴量
            emp_overtime = overtime_df[overtime_df["社員番号"] == emp_id]
            features.update(create_overtime_features(emp_overtime))

            # 6. 職位履歴特徴量
            emp_position = position_df[position_df["社員番号"] == emp_id]
            features.update(create_position_features(emp_position))

            # 7. カテゴリ別特徴量
            features.update(create_category_features(category, emp_dx, emp_hr))

            features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # 学習用とテスト用に分割
    train_features = train_df.merge(
        features_df, on=["社員番号", "category"], how="left"
    )
    test_features = test_df.merge(features_df, on=["社員番号", "category"], how="left")

    return train_features, test_features


def create_udemy_features(emp_udemy):
    """Udemy受講記録から特徴量を作成"""
    if len(emp_udemy) == 0:
        return get_default_udemy_features()

    features = {}

    # 基本統計
    features["udemy_total_records"] = len(emp_udemy)
    features["udemy_unique_courses"] = emp_udemy["コースID"].nunique()

    # 完了率関連
    completion_rates = emp_udemy["推定完了率%"].fillna(0)
    features["udemy_avg_completion"] = completion_rates.mean()
    features["udemy_max_completion"] = completion_rates.max()
    features["udemy_min_completion"] = completion_rates.min()
    features["udemy_std_completion"] = (
        completion_rates.std() if len(completion_rates) > 1 else 0
    )

    # クイズ成績
    quiz_data = emp_udemy[emp_udemy["レクチャーもしくはクイズ"] == "Quiz"]
    if len(quiz_data) > 0:
        quiz_scores = quiz_data["最終結果（クイズの場合）"].fillna(0)
        features["udemy_avg_quiz_score"] = quiz_scores.mean()
        features["udemy_quiz_count"] = len(quiz_data)
    else:
        features["udemy_avg_quiz_score"] = 0
        features["udemy_quiz_count"] = 0

    # マーク済み修了率
    marked_completed = emp_udemy["マーク済み修了"].fillna(False)
    features["udemy_completion_rate"] = marked_completed.sum() / len(emp_udemy)

    # コースカテゴリの多様性
    features["udemy_unique_categories"] = emp_udemy["コースカテゴリー"].nunique()

    # 学習期間
    try:
        emp_udemy["開始日"] = pd.to_datetime(emp_udemy["開始日"])
        emp_udemy["終了日"] = pd.to_datetime(emp_udemy["終了日"])

        # 学習期間の統計
        durations = (
            emp_udemy["終了日"] - emp_udemy["開始日"]
        ).dt.total_seconds() / 3600  # 時間単位
        features["udemy_avg_duration_hours"] = durations.mean()
        features["udemy_total_study_days"] = (
            emp_udemy["終了日"].max() - emp_udemy["開始日"].min()
        ).days
    except Exception:
        features["udemy_avg_duration_hours"] = 0
        features["udemy_total_study_days"] = 0

    return features


def get_default_udemy_features():
    """Udemyデータがない場合のデフォルト特徴量"""
    return {
        "udemy_total_records": 0,
        "udemy_unique_courses": 0,
        "udemy_avg_completion": 0,
        "udemy_max_completion": 0,
        "udemy_min_completion": 0,
        "udemy_std_completion": 0,
        "udemy_avg_quiz_score": 0,
        "udemy_quiz_count": 0,
        "udemy_completion_rate": 0,
        "udemy_unique_categories": 0,
        "udemy_avg_duration_hours": 0,
        "udemy_total_study_days": 0,
    }


def create_career_features(emp_career):
    """キャリア感アンケートから特徴量を作成"""
    features = {}

    if len(emp_career) == 0:
        # デフォルト値
        features["career_score_avg"] = 0
        features["career_has_data"] = 0
        return features

    # キャリア感スコアの平均（数値列のみ）
    numeric_cols = emp_career.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        features["career_score_avg"] = emp_career[numeric_cols].mean().mean()
    else:
        features["career_score_avg"] = 0

    features["career_has_data"] = 1

    return features


def create_dx_features(emp_dx):
    """DX研修記録から特徴量を作成"""
    features = {}

    # 基本統計
    features["dx_training_count"] = len(emp_dx)
    features["dx_unique_categories"] = (
        emp_dx["研修カテゴリ"].nunique() if len(emp_dx) > 0 else 0
    )

    # カテゴリ別受講数
    if len(emp_dx) > 0:
        category_counts = emp_dx["研修カテゴリ"].value_counts()
        features["dx_most_common_category_count"] = (
            category_counts.iloc[0] if len(category_counts) > 0 else 0
        )

        # 特定カテゴリの受講有無
        for cat in ["リテラシー_DX基礎", "リテラシー_データ活用", "スキルアップ"]:
            features[f"dx_has_{cat.replace('_', '_').lower()}"] = int(
                cat in emp_dx["研修カテゴリ"].values
            )
    else:
        features["dx_most_common_category_count"] = 0
        for cat in ["リテラシー_DX基礎", "リテラシー_データ活用", "スキルアップ"]:
            features[f"dx_has_{cat.replace('_', '_').lower()}"] = 0

    # 受講期間
    try:
        if len(emp_dx) > 0:
            emp_dx["研修実施日"] = pd.to_datetime(emp_dx["研修実施日"])
            features["dx_training_span_days"] = (
                emp_dx["研修実施日"].max() - emp_dx["研修実施日"].min()
            ).days
        else:
            features["dx_training_span_days"] = 0
    except Exception:
        features["dx_training_span_days"] = 0

    return features


def create_hr_features(emp_hr):
    """人事施策利用記録から特徴量を作成"""
    features = {}

    # 基本統計
    features["hr_training_count"] = len(emp_hr)
    features["hr_unique_categories"] = (
        emp_hr["カテゴリ"].nunique() if len(emp_hr) > 0 else 0
    )

    # カテゴリ別特徴量
    if len(emp_hr) > 0:
        category_counts = emp_hr["カテゴリ"].value_counts()
        features["hr_most_common_category_count"] = (
            category_counts.iloc[0] if len(category_counts) > 0 else 0
        )

        # 特定カテゴリの受講有無
        important_categories = [
            "ビジネススキルアップ研修",
            "昇格者研修",
            "マネジメント研修",
        ]
        for cat in important_categories:
            cat_key = (
                cat.replace("研修", "")
                .replace("スキルアップ", "skill")
                .replace("昇格者", "promotion")
                .replace("マネジメント", "management")
            )
            features[f"hr_has_{cat_key}"] = int(
                any(cat in str(c) for c in emp_hr["カテゴリ"].values)
            )
    else:
        features["hr_most_common_category_count"] = 0
        important_categories = [
            "ビジネススキルアップ研修",
            "昇格者研修",
            "マネジメント研修",
        ]
        for cat in important_categories:
            cat_key = (
                cat.replace("研修", "")
                .replace("スキルアップ", "skill")
                .replace("昇格者", "promotion")
                .replace("マネジメント", "management")
            )
            features[f"hr_has_{cat_key}"] = 0

    return features


def create_overtime_features(emp_overtime):
    """残業時間記録から特徴量を作成"""
    features = {}

    if len(emp_overtime) == 0:
        features.update(
            {
                "overtime_avg_hours": 0,
                "overtime_max_hours": 0,
                "overtime_min_hours": 0,
                "overtime_std_hours": 0,
                "overtime_total_months": 0,
                "overtime_trend": 0,
            }
        )
        return features

    hours = emp_overtime["hours"].fillna(0)

    # 基本統計
    features["overtime_avg_hours"] = hours.mean()
    features["overtime_max_hours"] = hours.max()
    features["overtime_min_hours"] = hours.min()
    features["overtime_std_hours"] = hours.std() if len(hours) > 1 else 0
    features["overtime_total_months"] = len(emp_overtime)

    # トレンド分析（最近の傾向）
    try:
        emp_overtime["date"] = pd.to_datetime(emp_overtime["date"])
        emp_overtime_sorted = emp_overtime.sort_values("date")

        if len(emp_overtime_sorted) >= 2:
            # 最近半年と過去半年の平均を比較
            mid_point = len(emp_overtime_sorted) // 2
            recent_avg = emp_overtime_sorted.iloc[mid_point:]["hours"].mean()
            past_avg = emp_overtime_sorted.iloc[:mid_point]["hours"].mean()
            features["overtime_trend"] = recent_avg - past_avg
        else:
            features["overtime_trend"] = 0
    except Exception:
        features["overtime_trend"] = 0

    return features


def create_position_features(emp_position):
    """職位履歴から特徴量を作成"""
    features = {}

    if len(emp_position) == 0:
        features.update(
            {
                "position_years_count": 0,
                "position_is_leader": 0,
                "position_is_manager": 0,
                "position_promotion_count": 0,
                "position_latest_year": 0,
            }
        )
        return features

    # 基本統計
    features["position_years_count"] = len(emp_position)
    features["position_latest_year"] = emp_position["year"].max()

    # 役職レベル
    positions = emp_position["役職"].values
    features["position_is_leader"] = int(
        any("リーダー" in str(pos) for pos in positions)
    )
    features["position_is_manager"] = int(
        any("マネ" in str(pos) or "マネージャー" in str(pos) for pos in positions)
    )

    # 昇進回数（役職の変化回数）
    if len(emp_position) > 1:
        position_changes = (
            emp_position.sort_values("year")["役職"].shift()
            != emp_position.sort_values("year")["役職"]
        )
        features["position_promotion_count"] = position_changes.sum()
    else:
        features["position_promotion_count"] = 0

    return features


def create_category_features(category, emp_dx, emp_hr):
    """カテゴリ別の特徴量を作成"""
    features = {}

    # カテゴリとDX研修の関連性
    if "データ" in category or "DX" in category or "BPR" in category:
        features["category_dx_relevance"] = 1
        # データ関連カテゴリでのDX研修受講数
        features["category_relevant_dx_count"] = len(
            emp_dx[emp_dx["研修カテゴリ"].str.contains("データ", na=False)]
        )
    else:
        features["category_dx_relevance"] = 0
        features["category_relevant_dx_count"] = 0

    # カテゴリと人事施策の関連性
    if "マネジメント" in category or "企画" in category:
        features["category_management_relevance"] = 1
        # マネジメント関連研修の受講数
        features["category_relevant_hr_count"] = len(
            emp_hr[emp_hr["カテゴリ"].str.contains("マネジメント|昇格", na=False)]
        )
    else:
        features["category_management_relevance"] = 0
        features["category_relevant_hr_count"] = 0

    return features


if __name__ == "__main__":
    # データ読み込み
    train_df = pd.read_csv("/kaggle/input/atma20/train.csv")
    test_df = pd.read_csv("/kaggle/input/atma20/test.csv")

    print("Creating features...")
    train_features, test_features = create_features(train_df, test_df)

    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(
        f"Feature columns: {len([col for col in train_features.columns if col not in ['社員番号', 'category', 'target']])}"
    )

    # 特徴量の保存
    train_features.to_csv("/kaggle/working/train_features.csv", index=False)
    test_features.to_csv("/kaggle/working/test_features.csv", index=False)

    print(
        "Features saved to /kaggle/working/train_features.csv and /kaggle/working/test_features.csv"
    )

    # 特徴量の統計情報を表示
    feature_cols = [
        col
        for col in train_features.columns
        if col not in ["社員番号", "category", "target"]
    ]
    print("\nFeature statistics:")
    print(train_features[feature_cols].describe())
