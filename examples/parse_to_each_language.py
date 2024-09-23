import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
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

# Generate and save code for each supported language
languages = ["python", "cpp", "javascript"]
file_extensions = {"python": ".py", "cpp": ".cpp", "javascript": ".js"}

for language in languages:
    code = lgbm_to_code.parse_lgbm_model(model._Booster, language)
    file_name = f"./code/lgbm_model_{language}{file_extensions[language]}"
    with open(file_name, "w") as f:
        f.write(code)

    print(f"Model code in {language.upper()} saved to {file_name}")

# You can then separately import and test each generated file
