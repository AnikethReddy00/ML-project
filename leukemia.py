import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, HuberRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score

# Step 1: Verify files
train_file = './AMLALL_train.data'
test_file = './AMLALL_test.data'

if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
    raise FileNotFoundError(f"Missing data files. Available: {os.listdir('.')}")
print("✅ Data files located")

# Step 2: Load and process data
try:
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    X_train, y_train = train_data.iloc[:, :-1], pd.factorize(train_data.iloc[:, -1])[0]
    X_test, y_test = test_data.iloc[:, :-1], pd.factorize(test_data.iloc[:, -1])[0]
    print("✅ Data loaded and processed")
except Exception as e:
    raise ValueError(f"Error loading data: {str(e)}")

# Step 3: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✅ Features standardized")

# Step 4: Feature selection with tuned ElasticNet
feature_corrs = np.abs(np.corrcoef(X_train.T)).mean()
alpha_boost = 0.1 * (1 + feature_corrs * 2)
elastic_net = ElasticNet(alpha=alpha_boost, l1_ratio=0.98, max_iter=5000, tol=1e-6)
elastic_net.fit(X_train, y_train)
selected_features = np.where(np.abs(elastic_net.coef_) > 0.015)[0]

if len(selected_features) < 4:
    selected_features = np.argsort(np.abs(elastic_net.coef_))[-4:][::-1]
print(f"🔎 Selected {len(selected_features)} features")

# Apply feature selection
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Step 5: Define and tune models with adjusted regularization
models = {
    "Logistic Regression": LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000, random_state=42),
    "Ridge Regression": Ridge(),
    "Huber Regression": HuberRegressor(max_iter=5000)
}

param_grids = {
    "Logistic Regression": {'C': [0.0001, 0.001, 0.01, 0.1]},
    "Ridge Regression": {'alpha': [0.01, 0.1, 1.0, 10.0]},
    "Huber Regression": {'alpha': [0.0001, 0.001, 0.01, 0.1]}
}

results = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params for {name}: {grid_search.best_params_}")
    
    # Predictions
    y_train_pred = best_model.predict(X_train_selected)
    y_test_pred = best_model.predict(X_test_selected)
    
    # Round predictions for classification accuracy
    y_train_pred_rounded = np.round(y_train_pred).astype(int)
    y_test_pred_rounded = np.round(y_test_pred).astype(int)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
    
    results[name] = {
        "Train MSE": mse_train,
        "Test MSE": mse_test,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Cross-Validation MSE": -cv_scores.mean(),
    }

# Step 6: Display Results
for model, metrics in results.items():
    print(f"\n📊 {model} Results:")
    print(f"   Train MSE: {metrics['Train MSE']:.4f}")
    print(f"   Test MSE: {metrics['Test MSE']:.4f}")
    print(f"   CV MSE: {metrics['Cross-Validation MSE']:.4f}")
    print(f"   Train Accuracy: {metrics['Train Accuracy']:.2%}")
    print(f"   Test Accuracy: {metrics['Test Accuracy']:.2%}")
    print("   ⚠️ Overfitting" if metrics['Train Accuracy'] - metrics['Test Accuracy'] > 0.1 else 
          "   ⚠️ Underfitting" if max(metrics['Train Accuracy'], metrics['Test Accuracy']) < 0.70 else 
          "   ✅ Model balanced")