import pytest
from lgbm_to_code import lgbm_to_code
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


@pytest.fixture
def trained_lgbm_model():
    X, y = make_classification(random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def test_parse_lgbm_model_python(trained_lgbm_model):
    code = lgbm_to_code.parse_lgbm_model(trained_lgbm_model, "python")
    assert "def lgbminfer(x):" in code
    assert "def func_" in code
    assert "if x[" in code
    assert "return " in code

def test_parse_lgbm_model_cpp(trained_lgbm_model):
    code = lgbm_to_code.parse_lgbm_model(trained_lgbm_model, "cpp")
    assert "float lgbminfer(const std::vector<float>& x)" in code
    assert "float func_" in code
    assert "if (x[" in code
    assert "return " in code

def test_parse_lgbm_model_javascript(trained_lgbm_model):
    code = lgbm_to_code.parse_lgbm_model(trained_lgbm_model, "javascript")
    assert "export const lgbminfer = (x) =>" in code
    assert "const func_" in code
    assert "if (x[" in code
    assert "return " in code

def test_parse_lgbm_model_unsupported_language(trained_lgbm_model):
    with pytest.raises(ValueError) as excinfo:
        lgbm_to_code.parse_lgbm_model(trained_lgbm_model, "unsupported")
    assert "Unsupported language: unsupported" in str(excinfo.value)

def test_parse_lgbm_model_inference_python(trained_lgbm_model):
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    code = lgbm_to_code.parse_lgbm_model(trained_lgbm_model, "python")
    exec(code, globals())
    predictions = np.array([lgbminfer(x) for x in X_test])
    assert len(predictions) == len(X_test)
    # Add assertions to compare predictions with actual values if needed