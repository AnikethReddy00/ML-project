# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

# Define the file path
file_path = "C:\\Users\\AMAN REDDY\\Downloads\\maternal_health_risk_data.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Drop any unnecessary columns (none in this case, but keeping for consistency)
df_cleaned = df.copy()

# Encode the target variable ("RiskLevel") - Convert to numeric labels
label_encoder = LabelEncoder()
df_cleaned["RiskLevel"] = label_encoder.fit_transform(df_cleaned["RiskLevel"])
# Note: This will map 'high risk' -> 0, 'low risk' -> 1, 'mid risk' -> 2 (or similar, based on alphabetical order)

# Separate features (X) and target variable (y)
X = df_cleaned.drop(columns=["RiskLevel"])
y = df_cleaned["RiskLevel"]

# Feature Engineering: Add polynomial features (degree=2) to capture interactions
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Feature Selection: Select top 10 features using SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_poly, y)

# Standardize the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Split data into training (75%) and testing (25%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, 
                                                    random_state=42, stratify=y)

# 1. Sparse Logistic Regression with L1 penalty (Lasso-like sparsity)
sparse_logreg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000, tol=1e-5, multi_class='auto')
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced']
}
grid_search = GridSearchCV(sparse_logreg, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_sparse = grid_search.best_estimator_
y_pred_sparse = best_sparse.predict(X_test)
sparse_accuracy = accuracy_score(y_test, y_pred_sparse)
print(f"Best parameters (Sparse L1): {grid_search.best_params_}")

# 2. Sparse Elastic Net Logistic Regression
elastic_logreg = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, tol=1e-5, multi_class='auto')
param_grid_elastic = {
    'C': [0.1, 1, 10],
    'l1_ratio': [0.3, 0.5, 0.7]
}
grid_search_elastic = GridSearchCV(elastic_logreg, param_grid_elastic, cv=10, scoring='accuracy', n_jobs=-1)
grid_search_elastic.fit(X_train, y_train)
best_elastic = grid_search_elastic.best_estimator_
y_pred_elastic = best_elastic.predict(X_test)
elastic_accuracy = accuracy_score(y_test, y_pred_elastic)
print(f"Best elastic parameters: {grid_search_elastic.best_params_}")

# 3. Linear Regression (Non-Sparse) - Adapted for multiclass by thresholding
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
# Convert continuous predictions to class labels (0, 1, 2) based on thresholds
y_pred_linear_binned = np.digitize(y_pred_linear, bins=[0.5, 1.5])  # Adjust bins based on label encoding
linear_accuracy = accuracy_score(y_test, y_pred_linear_binned)

# 4. Huber Regression (Non-Sparse) - Adapted for multiclass
huber = HuberRegressor(max_iter=100, tol=1e-4)
huber.fit(X_train, y_train)
y_pred_huber = huber.predict(X_test)
y_pred_huber_binned = np.digitize(y_pred_huber, bins=[0.5, 1.5])  # Adjust bins based on label encoding
huber_accuracy = accuracy_score(y_test, y_pred_huber_binned)

# Formatted output
print("\nAccuracy Comparison:")
print(f"Elastic Net (Sparse): {elastic_accuracy * 100:.2f}%")
print(f"Lasso (Sparse): {sparse_accuracy * 100:.2f}%")
print(f"Linear Regression (Non-Sparse): {linear_accuracy * 100:.2f}%")
print(f"Huber Regression (Non-Sparse): {huber_accuracy * 100:.2f}%")

# Function to predict risk level on new patient data using the best sparse model (Elastic Net)
def predict_risk_level(new_data):
    new_data_df = pd.DataFrame([new_data], columns=X.columns)
    new_data_poly = poly.transform(new_data_df)
    new_data_selected = selector.transform(new_data_poly)
    new_data_scaled = scaler.transform(new_data_selected)
    prediction = best_elastic.predict(new_data_scaled)
    return label_encoder.inverse_transform(prediction)[0]  # Convert numeric prediction back to original label

# Example: Predict using a sample row from test data
sample_data = X.iloc[0]
print("\nPrediction for sample:", predict_risk_level(sample_data))