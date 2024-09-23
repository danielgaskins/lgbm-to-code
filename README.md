# lgbm-to-code

This package provides functionality to convert trained LightGBM models into native code for different programming languages. This allows you to deploy your models in environments where Python or LightGBM dependencies might not be readily available.

## Installation

```
pip install lgbm-to-code
```

## Usage

```python
import lightgbm as lgb
from lgbm_to_code import lgbm_to_code

# Train your LightGBM model...
# For example:
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Convert to desired language
languages = ["python", "cpp", "javascript"]
for language in languages:
    code = lgbm_to_code.parse_lgbm_model(model._Booster, language)
    with open(f"lgbm_model_{language}.{'py' if language == 'python' else language}", "w") as f:
        f.write(code)
```

## Supported Languages

- Python
- C++
- JavaScript 

## Example

```python
import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from lgbm_to_code import lgbm_to_code

# Load dataset and train model
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Generate Python code
python_code = lgbm_to_code.parse_lgbm_model(model._Booster, "python")

# Save the code to a file
with open("lgbm_model.py", "w") as f:
    f.write(python_code)

# Now you can use this code in a separate Python environment
```

## Limitations

- Currently, the code generation only supports numerical features. 
- The generated code is not optimized for performance.

## License

[MIT](https://choosealicense.com/licenses/mit/)
