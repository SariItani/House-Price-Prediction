from joblib import load
from matplotlib import pyplot as plt
import pandas as pd

# Load the models
linear_model = load('../models/linear_model.joblib')
random_forest_model = load('../models/rf_model.joblib')
gradient_boosting_model = load('../models/gb_model.joblib')
polynomial_model = load('../models/poly_model.joblib')

# Read the test data
test = pd.read_csv('../data/processed/test_processed.csv')

# Read headers from file
with open('../data/results/headers.txt', 'r') as file:
    headers_str = file.read().replace("\n", "").replace("'", "")
    headers = headers_str.split("[")[1].split("]")[0].split(", ")

headers = [header.strip() for header in headers]

# Select columns based on headers
test = test[headers]
print(test.columns)
print(test.shape)

with open('../data/results/temp1.txt', 'w') as file:
    file.write(str(test.head()))

# Dictionary of models
models = {
    # 'LinearRegression': linear_model, # decent but not good
    'RandomForestRegression': random_forest_model, # best model
    'GradientBoostingRegression': gradient_boosting_model, # also best model
    # 'PolynomialRegression': polynomial_model # bad model
}

# Generate predictions for each model
predictions = {}
for model_name, model in models.items():
    predictions[model_name] = model.predict(test)

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv('../data/results/test_predictions.csv', index=True)


predicted_values_rf = predictions_df['RandomForestRegression']
predicted_values_gb = predictions_df['GradientBoostingRegression']

plt.figure(figsize=(10, 6))
plt.scatter(predictions_df.index, predicted_values_rf, color='blue', label='Predicted Sale Price', s=5)
plt.scatter(predictions_df.index, predicted_values_gb, color='red', label='Predicted Sale Price', s=5)
plt.xlabel('Index')
plt.ylabel('Predicted Sale Price')
plt.title('Predicted Sale Prices')
plt.legend()
plt.savefig('../results/making_predictions/predictions.png')

plt.figure(figsize=(8, 6))
plt.hist(x=predicted_values_rf)
title = 'Histogram of Predicted SalePrice - RF'
plt.title(title, fontsize=16)
plt.xlabel('Predicted SalePrice', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig('../results/making_predictions/predictions_rf.png')

plt.figure(figsize=(8, 6))
plt.hist(x=predicted_values_gb)
title = 'Histogram of Predicted SalePrice - GB'
plt.title(title, fontsize=16)
plt.xlabel('Predicted SalePrice', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig('../results/making_predictions/predictions_gb.png')