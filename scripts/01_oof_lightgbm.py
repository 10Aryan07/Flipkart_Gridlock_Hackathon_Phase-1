import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
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
    
    cat_cols_with_na = ['RoadType', 'Weather']
    for col in cat_cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    return df

print("2. Applying preprocessing...")
train_clean = preprocess_data(train)
test_clean = preprocess_data(test)

train_clean['demand'] = np.log1p(train_clean['demand'])

print("3. Executing OOF Target Encoding...")
def create_oof_target_encoding(train_df, test_df, cat_col, target_col, n_splits=5):
    train_df['oof_te_' + cat_col] = 0.0
    test_df['oof_te_' + cat_col] = 0.0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        mapping = X_tr.groupby(cat_col)[target_col].mean()
        train_df.loc[val_idx, 'oof_te_' + cat_col] = X_val[cat_col].map(mapping)
        
    full_mapping = train_df.groupby(cat_col)[target_col].mean()
    test_df['oof_te_' + cat_col] = test_df[cat_col].map(full_mapping)
    
    global_mean = train_df[target_col].mean()
    train_df['oof_te_' + cat_col].fillna(global_mean, inplace=True)
    test_df['oof_te_' + cat_col].fillna(global_mean, inplace=True)
    
    return train_df, test_df

train_clean, test_clean = create_oof_target_encoding(train_clean, test_clean, 'geohash', 'demand')

print("4. Label Encoding standard categoricals...")

categorical_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']

for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([train_clean[col].astype(str), test_clean[col].astype(str)], axis=0)
    le.fit(combined)
    train_clean[col] = le.transform(train_clean[col].astype(str))
    test_clean[col] = le.transform(test_clean[col].astype(str))

X = train_clean.drop(['Index', 'demand'], axis=1)
y = train_clean['demand']
X_test = test_clean.drop(['Index'], axis=1, errors='ignore')

print("5. Training OOF LightGBM Model...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05, 
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train_f, label=y_train_f, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val_f, label=y_val_f, categorical_feature=categorical_features, reference=train_data)
    
    model = lgb.train(lgb_params, train_data, valid_sets=[train_data, val_data], num_boost_round=1500, callbacks=[lgb.early_stopping(50, verbose=False)])
    test_predictions += np.expm1(model.predict(X_test, num_iteration=model.best_iteration)) / kf.n_splits

submission = pd.DataFrame({'Index': test['Index'], 'demand': test_predictions})
submission.to_csv('predictions/baseline_submission.csv', index=False)
print("SUCCESS: OOF LightGBM saved!")