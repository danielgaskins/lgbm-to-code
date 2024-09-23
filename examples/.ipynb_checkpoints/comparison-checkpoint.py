import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from lgbm_to_code import lgbm_to_code

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train an LGBM model
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Generate Python code from the trained model
python_code = lgbm_to_code.parse_lgbm_model(model, "python")

# Save the Python code to a file
with open("./code/lgbm_model.py", "w") as f:
    f.write(python_code)

# Import the generated function
from code.lgbm_model import lgbminfer  

# Make predictions using both the original model and the generated function
lgbm_predictions = model.predict(X_test)
generated_predictions = np.array([lgbminfer(x) for x in X_test])

# Compare the predictions
print(f"LGBM Predictions: {lgbm_predictions[:5]}")
print(f"Generated Predictions: {generated_predictions[:5]}")