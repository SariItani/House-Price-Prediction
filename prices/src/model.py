import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump

train = pd.read_csv('../data/processed/train_processed.csv')
test = pd.read_csv('../data/processed/test_processed.csv')

X_train = train.drop('SalePrice', axis=1)
y_train = train['SalePrice'].reset_index()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

for df in [X_train, X_val, y_train, y_val]:
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.describe())
    
selector = SelectKBest(score_func=f_regression, k=10)
X_train_top_10 = selector.fit_transform(X_train, y_train)
top_10_feature_indices = selector.get_support(indices=True)
top_10_features = X_train.columns[top_10_feature_indices]

with open('../data/results/headers.txt', 'w') as file:
    file.write(str(top_10_features))
    
with open('../data/results/temp.txt', 'w') as file:
    file.write(str(pd.DataFrame(X_train_top_10, columns=top_10_features).head()))

# Predict on the validation set
X_val_top_10 = X_val[top_10_features]

print(pd.DataFrame(X_train_top_10, columns=top_10_features).columns)
print(X_val_top_10.columns)
print(pd.DataFrame(X_train_top_10, columns=top_10_features).shape)
print(X_val_top_10.shape)

# Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train_top_10, y_train)
y_pred_linear_regression = linear_regression.predict(X_val_top_10)

mse_linear_regression = mean_squared_error(y_val, y_pred_linear_regression)
mae_linear_regression = mean_absolute_error(y_val, y_pred_linear_regression)
r2_linear_regression = r2_score(y_val, y_pred_linear_regression)

# Random Forest Regression
random_forest_regression = RandomForestRegressor()
random_forest_regression.fit(X_train_top_10, y_train)
y_pred_random_forest_regression = random_forest_regression.predict(X_val_top_10)

mse_random_forest_regression = mean_squared_error(y_val, y_pred_random_forest_regression)
mae_random_forest_regression = mean_absolute_error(y_val, y_pred_random_forest_regression)
r2_random_forest_regression = r2_score(y_val, y_pred_random_forest_regression)

# Gradient Boosting Regression
gradient_boosting_regression = GradientBoostingRegressor()
gradient_boosting_regression.fit(X_train_top_10, y_train)
y_pred_gradient_boosting_regression = gradient_boosting_regression.predict(X_val_top_10)

mse_gradient_boosting_regression = mean_squared_error(y_val, y_pred_gradient_boosting_regression)
mae_gradient_boosting_regression = mean_absolute_error(y_val, y_pred_gradient_boosting_regression)
r2_gradient_boosting_regression = r2_score(y_val, y_pred_gradient_boosting_regression)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_top_10)
X_val_poly = poly_features.transform(X_val_top_10)

polynomial_regression = LinearRegression()
polynomial_regression.fit(X_train_poly, y_train)
y_pred_polynomial_regression = polynomial_regression.predict(X_val_poly)

mse_polynomial_regression = mean_squared_error(y_val, y_pred_polynomial_regression)
mae_polynomial_regression = mean_absolute_error(y_val, y_pred_polynomial_regression)
r2_polynomial_regression = r2_score(y_val, y_pred_polynomial_regression)

# Print error metrics
print("Linear Regression:")
print("MSE:", mse_linear_regression)
print("MAE:", mae_linear_regression)
print("R-squared:", r2_linear_regression)
print()
print("Random Forest Regression:")
print("MSE:", mse_random_forest_regression)
print("MAE:", mae_random_forest_regression)
print("R-squared:", r2_random_forest_regression)
print()
print("Gradient Boosting Regression:")
print("MSE:", mse_gradient_boosting_regression)
print("MAE:", mae_gradient_boosting_regression)
print("R-squared:", r2_gradient_boosting_regression)
print()
print("Polynomial Regression (Degree 2):")
print("MSE:", mse_polynomial_regression)
print("MAE:", mae_polynomial_regression)
print("R-squared:", r2_polynomial_regression)

# Plot the error graph
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_linear_regression, color='blue', label='Linear Regression', s=5)
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/linear.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_random_forest_regression, color='red', label='Random Forest Regression', s=5)
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/rf.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_gradient_boosting_regression, color='green', label='Gradient Boosting Regression', s=5)
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/gb.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_polynomial_regression, color='orange', label='Polynomial Regression (Deg 2)', s=5)
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/polynomial.png')

# Plot error trend vs. datapoint index
linear_regression_errors = y_val.values.flatten() - y_pred_linear_regression.flatten()
random_forest_errors = y_val.values.flatten() - y_pred_random_forest_regression.flatten()
gradient_boosting_errors = y_val.values.flatten() - y_pred_gradient_boosting_regression.flatten()
polynomial_errors = y_val.values.flatten() - y_pred_polynomial_regression.flatten()
plt.figure(figsize=(10, 8))
plt.plot(linear_regression_errors, color='blue', label='Linear Regression')
plt.xlabel('Datapoint Index')
plt.ylabel('Error')
plt.title('Error Trend vs. Datapoint Index')
plt.legend()
plt.savefig('../results/model_training/error_vs_datapoint_linear.png')

plt.figure(figsize=(10, 8))
plt.plot(random_forest_errors, color='red', label='Random Forest Regression')
plt.xlabel('Datapoint Index')
plt.ylabel('Error')
plt.title('Error Trend vs. Datapoint Index')
plt.legend()
plt.savefig('../results/model_training/error_vs_datapoint_rf.png')

plt.figure(figsize=(10, 8))
plt.plot(gradient_boosting_errors, color='green', label='Gradient Boosting Regression')
plt.xlabel('Datapoint Index')
plt.ylabel('Error')
plt.title('Error Trend vs. Datapoint Index')
plt.legend()
plt.savefig('../results/model_training/error_vs_datapoint_gb.png')

plt.figure(figsize=(10, 8))
plt.plot(polynomial_errors, color='orange', label='Polynomial Regression (Deg 2)')
plt.xlabel('Datapoint Index')
plt.ylabel('Error')
plt.title('Error Trend vs. Datapoint Index')
plt.legend()
plt.savefig('../results/model_training/error_vs_datapoint_polynomial.png')

# Initialize empty lists to store error values
mse_linear_regression_list = []
mae_linear_regression_list = []
r2_linear_regression_list = []
mse_random_forest_regression_list = []
mae_random_forest_regression_list = []
r2_random_forest_regression_list= []
mse_gradient_boosting_regression_list = []
mae_gradient_boosting_regression_list = []
r2_gradient_boosting_regression_list = []
mse_polynomial_regression_list = []
mae_polynomial_regression_list = []
r2_polynomial_regression_list = []

# Train the models and track error values
for i in range(1, len(X_train_top_10) + 1, 50):
    X_train_current = X_train_top_10[:i]
    y_train_current = y_train[:i]
    
    # Reinitialize models to clear fitting cache
    linear_regression = LinearRegression()
    random_forest_regression = RandomForestRegressor()
    gradient_boosting_regression = GradientBoostingRegressor()
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_current)
    X_val_poly = poly_features.transform(X_val_top_10[:i])
    polynomial_regression = LinearRegression()

    # Linear Regression
    linear_regression.fit(X_train_current, y_train_current)
    y_pred_linear_regression = linear_regression.predict(X_val_top_10[:i])
    
    mse_linear_regression_list.append(mean_squared_error(y_val[:i], y_pred_linear_regression))
    mae_linear_regression_list.append(mean_absolute_error(y_val[:i], y_pred_linear_regression))
    r2_linear_regression_list.append(r2_score(y_val[:i], y_pred_linear_regression))

    # Random Forest Regression
    random_forest_regression.fit(X_train_current, y_train_current)
    y_pred_random_forest_regression = random_forest_regression.predict(X_val_top_10[:i])
    
    mse_random_forest_regression_list.append(mean_squared_error(y_val[:i], y_pred_random_forest_regression))
    mae_random_forest_regression_list.append(mean_absolute_error(y_val[:i], y_pred_random_forest_regression))
    r2_random_forest_regression_list.append(r2_score(y_val[:i], y_pred_random_forest_regression))

    # Gradient Boosting Regression
    gradient_boosting_regression.fit(X_train_current, y_train_current)
    y_pred_gradient_boosting_regression = gradient_boosting_regression.predict(X_val_top_10[:i])
    
    mse_gradient_boosting_regression_list.append(mean_squared_error(y_val[:i], y_pred_gradient_boosting_regression))
    mae_gradient_boosting_regression_list.append(mean_absolute_error(y_val[:i], y_pred_gradient_boosting_regression))
    r2_gradient_boosting_regression_list.append(r2_score(y_val[:i], y_pred_gradient_boosting_regression))

    # Polynomial Regression
    polynomial_regression.fit(X_train_poly, y_train_current)
    y_pred_polynomial_regression = polynomial_regression.predict(X_val_poly)

    mse_polynomial_regression_list.append(mean_squared_error(y_val[:i], y_pred_polynomial_regression))
    mae_polynomial_regression_list.append(mean_absolute_error(y_val[:i], y_pred_polynomial_regression))
    r2_polynomial_regression_list.append(r2_score(y_val[:i], y_pred_polynomial_regression))

# Print error metrics
print("Linear Regression:")
print("MSE:", mse_linear_regression_list[-1])
print("MAE:", mae_linear_regression_list[-1])
print("R-squared:", r2_linear_regression_list[-1])
print()
print("Random Forest Regression:")
print("MSE:", mse_random_forest_regression_list[-1])
print("MAE:", mae_random_forest_regression_list[-1])
print("R-squared:", r2_random_forest_regression_list[-1])
print()
print("Gradient Boosting Regression:")
print("MSE:", mse_gradient_boosting_regression_list[-1])
print("MAE:", mae_gradient_boosting_regression_list[-1])
print("R-squared:", r2_gradient_boosting_regression_list[-1])
print()
print("Polynomial Regression (Degree 2):")
print("MSE:", mse_polynomial_regression_list[-1])
print("MAE:", mae_polynomial_regression_list[-1])
print("R-squared:", r2_polynomial_regression_list[-1])

# Plot the error graph
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_linear_regression, color='blue', label='Linear Regression', s=5)
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/linear_historical.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_random_forest_regression, color='red', label='Random Forest Regression', s=5)
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/rf_historical.png')

plt.figure(figsize=(8, 6))
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.scatter(y_val, y_pred_gradient_boosting_regression, color='green', label='Gradient Boosting Regression', s=5)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/gb_historical.png')

plt.figure(figsize=(8, 6))
plt.plot(y_val, y_val, color='black', linewidth=0.5, label='y = x')
plt.scatter(y_val, y_pred_polynomial_regression, color='orange', label='Polynomial Regression (Deg 2)', s=5)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Regression Model Predictions')
plt.legend()
plt.savefig('../results/model_training/polynomial_historical.png')

# Plot the error trend
plt.figure(figsize=(8, 6))
plt.plot(range(1, 1+ len(mse_linear_regression_list)), mse_linear_regression_list, color='blue', label='Linear Regression')
plt.plot(range(1, 1+ len(mse_random_forest_regression_list)), mse_random_forest_regression_list, color='red', label='Random Forest Regression')
plt.plot(range(1, 1+ len(mse_gradient_boosting_regression_list)), mse_gradient_boosting_regression_list, color='green', label='Gradient Boosting Regression')
plt.plot(range(1, 1+ len(mse_polynomial_regression_list)), mse_polynomial_regression_list, color='orange', label='Polynomial Regression (Deg 2)')
plt.xlabel('Number of Data Points')
plt.ylabel('Mean Squared Error')
plt.title('Error Trend')
plt.legend()
plt.savefig('../results/model_training/mse_historical.png')

plt.figure(figsize=(8, 6))
plt.plot(range(1, 1+ len(mae_linear_regression_list)), mae_linear_regression_list, color='blue', label='Linear Regression')
plt.plot(range(1, 1+ len(mae_random_forest_regression_list)), mae_random_forest_regression_list, color='red', label='Random Forest Regression')
plt.plot(range(1, 1+ len(mae_gradient_boosting_regression_list)), mae_gradient_boosting_regression_list, color='green', label='Gradient Boosting Regression')
plt.plot(range(1, 1+ len(mae_polynomial_regression_list)), mae_polynomial_regression_list, color='orange', label='Polynomial Regression (Deg 2)')
plt.xlabel('Number of Data Points')
plt.ylabel('Mean Squared Error')
plt.title('Error Trend')
plt.legend()
plt.savefig('../results/model_training/mae_historical.png')

plt.figure(figsize=(8, 6))
plt.plot(range(1, 1+ len(r2_linear_regression_list)), r2_linear_regression_list, color='blue', label='Linear Regression')
plt.plot(range(1, 1+ len(r2_random_forest_regression_list)), r2_random_forest_regression_list, color='red', label='Random Forest Regression')
plt.plot(range(1, 1+ len(r2_gradient_boosting_regression_list)), r2_gradient_boosting_regression_list, color='green', label='Gradient Boosting Regression')
plt.plot(range(1, 1+ len(r2_polynomial_regression_list)), r2_polynomial_regression_list, color='orange', label='Polynomial Regression (Deg 2)')
plt.xlabel('Number of Data Points')
plt.ylabel('Mean Squared Error')
plt.title('Error Trend')
plt.legend()
plt.savefig('../results/model_training/r2_historical.png')

# Saving the Final Models Models
dump(linear_regression, '../models/linear_model.joblib')
dump(random_forest_regression, '../models/rf_model.joblib')
dump(gradient_boosting_regression, '../models/gb_model.joblib')
dump(polynomial_regression, '../models/poly_model.joblib')