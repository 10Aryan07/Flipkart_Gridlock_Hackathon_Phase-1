import pandas as pd

lgb = pd.read_csv('predictions/baseline_submission.csv') 
cat = pd.read_csv('predictions/catboost_preds.csv')

ensemble = lgb.copy()
ensemble['demand'] = ((lgb['demand'] * 0.60) + (cat['demand'] * 0.40)) * 0.99

ensemble.to_csv('submissions/submission_ensemble.csv', index=False)
print("successful")