import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

x = np.array([5, 10, 15, 20, 25, 30])
y = np.array([3, 4, 6, 7, 9, 11])
print("X:",x)
print("Y:",y)

x_mean = np.mean(x)
y_mean = np.mean(y)
w1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
w0 = y_mean - w1 * x_mean

print("\nRegression Coefficients")
print("intecept (w0):", round(w0,2))
print("slope (w1):", round(w1, 2))

y_pred_manual = w0 + w1 * x
print("\nManual Predictions:", np.round(y_pred_manual, 2))

mae = mean_absolute_error(y, y_pred_manual)
mse = mean_squared_error(y, y_pred_manual)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_manual)

print("\nManual Error Metrics")
print("MAE:", round(mae,2))
print("MSE:", round(mse,2))
print("RMSE:", round(rmse,2))
print("R2:", round(r2,2))

X = x.reshape(-1,1)
model = LinearRegression()
model.fit(X,y)

y_pred_sklearn = model.predict(X)

print("\nsklearn Regression Coefficients")
print("intercept:",round(model.intercept_,2))
print("slope:",round(model.coef_[0],2))

print("\nsklearn error metrics")
print("MAE:", round(mean_absolute_error(y, y_pred_sklearn), 2))
print("MSE:", round(mean_squared_error(y, y_pred_sklearn), 2))
print("RMSE:", round(np.sqrt(mean_absolute_error(y, y_pred_sklearn)), 2))
print("R2:", round(r2_score(y, y_pred_sklearn), 2))

plt.figure()
plt.scatter(x,y)
plt.plot(x, y_pred_manual)
plt.xlabel("open ports")
plt.ylabel("incidents")
plt.title("simple linear regression")
plt.show()
