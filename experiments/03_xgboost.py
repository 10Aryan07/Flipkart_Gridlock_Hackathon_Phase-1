import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("1. Loading raw data for XGBoost...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def preprocess_data(df):
    df = df.copy()
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
    df.drop('timestamp', axis=1, inplace=True)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
    df['is_peak_morning'] = df['hour'].isin([8, 9, 10, 11]).astype(int)
    df['is_peak_evening'] = df['hour'].isin([17, 18, 19, 20, 21]).astype(int)

    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

    cat_cols_with_na = ['RoadType', 'Weather']
    for col in cat_cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df

train_clean = preprocess_data(train)
test_clean = preprocess_data(test)

categorical_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']

print("2. Label Encoding for XGBoost...")
for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([train_clean[col].astype(str), test_clean[col].astype(str)], axis=0)
    le.fit(combined)
    train_clean[col] = le.transform(train_clean[col].astype(str))
    test_clean[col] = le.transform(test_clean[col].astype(str))

X = train_clean.drop(['Index', 'demand'], axis=1)
y = np.log1p(train_clean['demand'])
X_test = test_clean.drop(['Index'], axis=1, errors='ignore')

print("3. Starting XGBoost 5-Fold Training...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=1500,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False
    )
    
    test_predictions += np.expm1(model.predict(X_test)) / kf.n_splits
    print(f"Fold {fold+1} Finished!")

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_predictions
})
submission.to_csv('predictions/xgboost_preds.csv', index=False)
print("Saved as 'xgboost_preds.csv'")