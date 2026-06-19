import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print("1. Loading raw data for CatBoost...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def preprocess_for_catboost(df):
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
    
    cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            
    return df, cat_cols

print("2. Applying preprocessing...")
train_clean, cat_features = preprocess_for_catboost(train)
test_clean, _ = preprocess_for_catboost(test)

X = train_clean.drop(['Index', 'demand'], axis=1)
y = np.log1p(train_clean['demand']) 
X_test = test_clean.drop(['Index'], axis=1)

print("3. Starting CatBoost 5-Fold Training...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cb_test_predictions = np.zeros(len(X_test))

cb_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'iterations': 1500,
    'depth': 6,
    'random_seed': 42,
    'verbose': 100
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train_f, y_train_f, cat_features=cat_features)
    val_pool = Pool(X_val_f, y_val_f, cat_features=cat_features)
    
    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    cb_test_predictions += np.expm1(model.predict(X_test)) / kf.n_splits

print("\nSaving CatBoost predictions...")
submission = pd.DataFrame({
    'Index': test_clean['Index'], 
    'demand': cb_test_predictions
})
submission.to_csv('predictions/catboost_preds.csv', index=False)
print("Saved as 'catboost_preds.csv'")