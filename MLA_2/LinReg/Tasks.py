import math
import numpy as np
import matplotlib.pyplot as plt

file_path = "PCB.dt"
points = np.loadtxt(file_path)

print(points.shape)

y = points[:, 1]
x = points[:, 0].reshape(-1, 1)
x = np.hstack((np.ones((x.shape[0], 1)), x))  # add a column of 1's

print(y)

def pseudo_inverse(X: np.ndarray):
    X_cross = np.linalg.inv(X.T.dot(X)).dot(X.T)
    return X_cross

def linear_regression_params(X: np.ndarray, Y: np.ndarray):
    """Computes the params for linear regression using input data"""
    X_cross = pseudo_inverse(X) 
    params = np.matmul(X_cross, Y)  
    return params

def predict(params, new_x):
    """Predicts a single point"""
    return params[0] + params[1] * new_x

def exponential_predict(params, new_x): # h(x) = exp^(ax+b)
    return np.exp(params[0] + params[1] * new_x)

def exp_sqrt_predict(params, new_x): # h(x) = exp^(ax+b)
    return np.exp(params[0] + params[1] * np.sqrt(new_x))

y_transformed = np.log(y) # take the log of y
x_transformed = np.sqrt(x) # take sqrt of x
params_transformed = linear_regression_params(x, y_transformed) # transformed params
params_nonlin =linear_regression_params(x_transformed, y_transformed)

# MSE for nonlinear fucn
nonlin_preds = exp_sqrt_predict(params_nonlin, x[:, 1])
nonlin_mse = np.mean((y - nonlin_preds))
# Calculate the MSE for the exponetial model
exp_predictions = exponential_predict (params_transformed, x[:, 1])
exp_mse = np.mean((y - exp_predictions)**2)
# The MSE for the transformed linear function
ln_mse = np.mean((y_transformed - np.log(exp_predictions))**2)
# The MSE of the linear model
params = linear_regression_params(x, y)
predictions = predict(params, x[:, 1])
mse = np.mean((y - predictions)**2)

# Report the parameters and MSE
print("exp Parameters a and b:", params_transformed[1], params_transformed[0])
print("lin Parameters a and b:", params[1], params[0]) 
print("MSE: Linear model:", mse)
print("MSE: Expoenential model:", exp_mse)
print("MSE: Expoenential model on ln(y) data:", ln_mse)

print("MSE: nonlin model", nonlin_mse )

# Calculate R-squared for the linear model
SSE_linear = np.sum((y - predictions)**2)
SST = np.sum((y - np.mean(y))**2)
r_squared_linear = round(1 - (SSE_linear / SST),2)

# Calculate R-squared for the exponential model
SSE_exp = np.sum((y - exp_predictions)**2)
r_squared_exp = round(1 - (SSE_exp / SST),2)

# Calculate R-squared for the exponential model on ln(y) data
SSE_ln = np.sum((y_transformed - np.log(exp_predictions))**2)
r_squared_ln = round(1 - (SSE_ln / SST), 2)

# Calculate R-squared for the nonlinear model
SSE_nlin = np.sum((y - nonlin_preds)**2)
r_squared_nlin = round(1 - (SSE_nlin / SST),2)

# Report the R-squared values
print("R-squared (Linear model):", r_squared_linear)
print("R-squared (Exponential model):", r_squared_exp)
print("R-squared (Exponential model on ln(y) data):", r_squared_ln)
print("R-squared (Exponential model on ln(y) data):", r_squared_nlin)

# Make comparison plot
ages = range(0, 13)
pred = [predict(params, age) for age in ages]
exp_pred = [exponential_predict(params_transformed, age) for age in ages]
ln_pred = [predict(params_transformed, age) for age in ages]
nonlin_pred = [exp_sqrt_predict(params_nonlin, age) for age in ages]
print(nonlin_preds)
plt.figure(figsize=(10, 6))
plt.plot(ages, pred, label=f"Best Linear fit: Params {np.round(params,2)}: MSE {round(mse,2)}: R^2 {r_squared_linear}:")
plt.plot(ages, exp_pred, label = f"Exponential fit: Params {np.round(params_transformed,2)}: MSE {round(exp_mse,2)}: R^2 {r_squared_exp}:")
plt.plot(ages, nonlin_pred, label=f"Other nonlinear fit: Params {np.round(params_nonlin,2)}: MSE {round(nonlin_mse,2)}: R^2 {r_squared_nlin}:", color = "Red") 
plt.xlabel("ages")
plt.ylabel("PCB Concentration")
plt.scatter(x[:, 1], y, marker='o', label="Data", color = "Green")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.xlabel("ages")
plt.ylabel("PCB Concentration")
plt.scatter(x[:, 1], y, marker='o', label="Raw data", color = "Red")
plt.scatter(x[:, 1], np.log(y), marker='o', label="log(y) data", color = "Blue")
plt.plot(ages, pred, label=f"Best Linear fit")
plt.plot(ages, ln_pred, label = " Best fit to ln(y)" )
plt.grid(True)
plt.legend()
plt.show()