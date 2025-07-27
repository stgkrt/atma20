# Ensemble Tool

複数のモデルのsubmissionファイルを重み付きアンサンブルするツールです。

## 機能

- 複数モデルのsubmissionファイルの重み付き平均アンサンブル
- OOF予測のアンサンブル
- アンサンブル結果の自動保存（フォルダ作成）
- メタデータの保存（重み、タイムスタンプなど）

## 使用方法

### 1. Pythonスクリプトから使用

```python
from ensemble import ensemble_submissions

# 基本的な使用法（均等重み）
model_paths = [
    "/kaggle/working/experiment1_cat",
    "/kaggle/working/experiment1_lgb", 
    "/kaggle/working/experiment1_xgb"
]

ensemble_dir = ensemble_submissions(
    model_paths=model_paths,
    ensemble_name="my_ensemble"
)

# 重み付きアンサンブル
weights = [0.2, 0.6, 0.2]  # cat: 20%, lgb: 60%, xgb: 20%
ensemble_dir = ensemble_submissions(
    model_paths=model_paths,
    weights=weights,
    ensemble_name="lgb_focused_ensemble"
)
```

### 2. コマンドラインから使用

```bash
# 基本的な使用法（均等重み）
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb

# 重み付きアンサンブル
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb --weights 0.2 0.6 0.2

# カスタム名前を指定
python run_ensemble.py --models experiment1_cat experiment1_lgb --name "simple_ensemble"

# OOFもアンサンブル
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb --oof

# 異なる基準ディレクトリを使用
python run_ensemble.py --models model1 model2 --base-dir /path/to/models --output-dir /path/to/output
```

## 出力

各アンサンブル実行では、`/kaggle/working`に新しいフォルダが作成されます：

```
/kaggle/working/ensemble_[name]_[timestamp]/
├── submission.csv           # アンサンブル後のsubmission
├── ensemble_metadata.txt   # アンサンブル情報（重み、モデルパスなど）
└── oof_ensemble.csv        # OOFアンサンブル結果（--oofオプション使用時）
```

## パラメータ

### ensemble_submissions 関数

- `model_paths`: モデル結果フォルダのパスのリスト
- `weights`: 各モデルの重み（省略時は均等重み）
- `output_dir`: 出力先ディレクトリ（デフォルト: `/kaggle/working`）
- `ensemble_name`: アンサンブル名（省略時は自動生成）
- `submission_filename`: submissionファイル名（デフォルト: `submission.csv`）

### コマンドライン引数

- `--models`: アンサンブルするモデルフォルダ名のリスト（必須）
- `--weights`: 各モデルの重み（省略時は均等重み）
- `--base-dir`: モデルフォルダの基準ディレクトリ（デフォルト: `/kaggle/working`）
- `--output-dir`: 出力先ディレクトリ（デフォルト: `/kaggle/working`）
- `--name`: アンサンブル名（省略時は自動生成）
- `--oof`: OOFもアンサンブルする

## 注意事項

- すべてのモデルのsubmissionファイルは同じ形式である必要があります
- 重みは自動的に正規化されます（合計が1になるように調整）
- 指定したモデルフォルダが存在しない場合はエラーになります

## 例

現在のワークスペースでは以下のようなアンサンブルが可能です：

```bash
# 3つのモデルを均等重みでアンサンブル
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb

# LightGBMを重視したアンサンブル
python run_ensemble.py --models experiment1_cat experiment1_lgb experiment1_xgb --weights 0.2 0.6 0.2

# CatBoostとLightGBMのみをアンサンブル
python run_ensemble.py --models experiment1_cat experiment1_lgb --weights 0.3 0.7 --name "cat_lgb_ensemble"
```
