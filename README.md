# CTR Prediction Models

This repository contains example code for click-through rate (CTR) prediction on the Criteo dataset.
The `ctr_prediction.py` module implements preprocessing and three models:
logistic regression, XGBoost and DeepFM.

## Usage
1. Install Python packages:
   `pandas`, `numpy`, `scikit-learn`, `xgboost` and `torch`.
2. Place the Criteo data CSV so that the first column is the label.
3. Example training snippet:

```python
from ctr_prediction import *

X, y = load_criteo('train.csv')
X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(X, y)
X_train_h = hash_features(X_train)
model = train_logistic_regression(X_train_h, y_train)
auc, loss = evaluate_sklearn_model(model, hash_features(X_test), y_test)
print('AUC', auc, 'LogLoss', loss)
```

The `train_deepfm` function trains the DeepFM model using PyTorch.
