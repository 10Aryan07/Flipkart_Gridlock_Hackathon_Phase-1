# GRIDLOCK: Phase 1 Offline ML Pipeline
**ASTraM Event Optimizer**

This directory contains our Phase 1 data science pipeline. Our approach focused on heavy offline batch-processing, feature engineering, and training advanced gradient boosting ensembles(LightGBM, CatBoost, XGBoost) to establish robust baseline predictions.

For a detailed breakdown of our methodology, please refer to `approach.txt`.

## Folder Structure

GRIDLOCK/
│
├── approach.txt             # Detailed methodology and project overview
├── README.md                # Phase 1 documentation
├── requirements.txt         # Offline ML dependencies
│
├── data/                    # Raw competition datasets
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── experiments/             # Baseline model training and tuning scripts
│   ├── 01_lightgbm.py
│   ├── 02_catboost.py
│   └── 03_xgboost.py
│
├── scripts/                 # Out-Of-Fold (OOF) generation and ensembling
│   ├── 01_oof_lightgbm.py
│   ├── 02_oof_catboost.py
│   └── 03_ensemble.py
│
├── logs/                    # Training logs
│   └── catboost_info/       
│
├── predictions/             # Individual model outputs for blending
│   ├── baseline_submission.csv
│   ├── catboost_preds.csv
│   └── xgboost_preds.csv
│
└── submissions/             # Final competition-ready output
    └── submission_ensemble.csv

## Installation & Execution

**1. Navigate to the Directory**

cd GRIDLOCK

**2. Install Phase 1 Dependencies**

pip install -r requirements.txt

**3. Execute the Pipeline**
To recreate our final submission_ensemble.csv, run the scripts in sequential order to generate the out-of-fold predictions and build the final blended ensemble:

python scripts/01_oof_lightgbm.py
python scripts/02_oof_catboost.py
python scripts/03_ensemble.py