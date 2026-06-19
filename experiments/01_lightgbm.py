import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("1. Loading raw data...")
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

print("2. Applying pure baseline preprocessing...")
train_clean = preprocess_data(train)
test_clean = preprocess_data(test)

print("3. Label Encoding...")
categorical_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']

for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([train_clean[col].astype(str), test_clean[col].astype(str)], axis=0)
    le.fit(combined)
    train_clean[col] = le.transform(train_clean[col].astype(str))
    test_clean[col] = le.transform(test_clean[col].astype(str))

X = train_clean.drop(['Index', 'demand'], axis=1)
y = np.log1p(train_clean['demand'])
X_test = test_clean.drop(['Index'], axis=1, errors='ignore')

print("4. Training Pure Baseline LightGBM...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))
cv_scores = []

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    
    'learning_rate': 0.01,        
    'n_estimators': 3000,         
    
    'num_leaves': 127,            
    'max_depth': 8,               
    
    'feature_fraction': 0.8,      
    'bagging_fraction': 0.8,      
    'bagging_freq': 5,           
    
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train_f, label=y_train_f, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val_f, label=y_val_f, categorical_feature=categorical_features)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_preds = np.expm1(model.predict(X_val_f, num_iteration=model.best_iteration))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_val_f), val_preds))
    cv_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.5f}")
    
    test_predictions += np.expm1(model.predict(X_test, num_iteration=model.best_iteration)) / kf.n_splits

print(f"\n RESTORED AVERAGE CV RMSE: {np.mean(cv_scores):.5f}")

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_predictions
})
submission.to_csv('predictions/baseline_submission.csv', index=False)
print("Saved as 'baseline_submission.csv'")