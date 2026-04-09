import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([3, 6, 9, 13, 15, 20, 22, 30], dtype=float)

def create_polynomial_features(x, degree=2):
    features = [np.ones(len(x))]
    for d in range(1, degree + 1):
        features.append(x ** d)
    return np.column_stack(features)

X_poly = create_polynomial_features(x, degree=2)

kf = KFold(n_splits=4, shuffle=True, random_state=42)
model = LinearRegression(fit_intercept=False)

mae_list, mse_list, rmse_list, r2_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_poly), start=1):
    X_train, X_test = X_poly[train_idx], X_poly[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    
    print(f"Fold {fold}")
    print(f"MAE  = {mae:.3f}")
    print(f"MSE  = {mse:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"R^2  = {r2:.3f}\n")

print("Polynomial Regression (Degree = 2) - Cross Validation Results")
print(f"Average MAE  = {np.mean(mae_list):.3f}")
print(f"Average MSE  = {np.mean(mse_list):.3f}")
print(f"Average RMSE = {np.mean(rmse_list):.3f}")
print(f"Average R^2  = {np.mean(r2_list):.3f}")

model.fit(X_poly, y)

x_pred = np.linspace(min(x), max(x), 200)
X_pred = create_polynomial_features(x_pred, degree=2)
y_pred = model.predict(X_pred)

plt.figure()
plt.scatter(x, y, label="Actual Data")
plt.plot(x_pred, y_pred, label="Polynomial Regression Curve", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression (Degree = 2)")
plt.legend()
plt.show()
