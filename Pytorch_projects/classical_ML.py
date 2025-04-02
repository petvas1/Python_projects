import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.inspection import permutation_importance

# Load dataset
df = pd.read_csv("data.csv").dropna()
X = df.drop(columns=['t', 'time'], axis=1)
y = df['t']
features = X.columns

# Normalize features (important for SVM, not for Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train Random Forest model (for comparison)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate both models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)
    print(f"\n🔹 {model_name} Performance:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Explained Variance Score: {explained_var:.3f}")

evaluate_model(y_test, y_pred_svm, "SVM")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Permutation Importance (for feature importance analysis)
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
feature_importances = perm_importance.importances_mean


def predict_temperature(input_data):
    # Load the trained model and scaler
    model = joblib.load("svm_temperature_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Convert input to a NumPy array and reshape for scaling
    X_new = np.array([input_data]).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)

    # Predict temperature
    predicted_temp = rf_model.predict(X_new_scaled)[0]
    return predicted_temp


# Example Prediction
input_values = [600, 16.25, 48, 40000, 16.896063, 0.004694853, -1.3294983, 1.3652191, 0.07224846, -1.24E-06]
predicted_temp = predict_temperature(input_values)
print(f"\nPredicted Temperature: {predicted_temp:.2f}°C")

# Plot Feature Importance
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.show()
