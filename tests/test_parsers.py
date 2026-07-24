from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from lgbm_to_code import parse_lgbm_model


class DumpedModel:
    def __init__(self, dumped: dict[str, object]):
        self.dumped = dumped

    def dump_model(self) -> dict[str, object]:
        return self.dumped


@pytest.fixture(scope="module")
def regression_case():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(160, 6))
    features[::11, 2] = np.nan
    target = 2.5 * np.nan_to_num(features[:, 0]) - features[:, 1] ** 2
    model = lgb.LGBMRegressor(
        n_estimators=17,
        num_leaves=9,
        learning_rate=0.08,
        verbosity=-1,
        random_state=42,
    )
    model.fit(features[:120], target[:120])
    return model, features[120:]


def python_predictions(code: str, rows: np.ndarray) -> np.ndarray:
    namespace: dict[str, object] = {}
    exec(code, namespace)
    infer = namespace["lgbminfer"]
    return np.asarray([infer(row) for row in rows])


def test_python_matches_lightgbm_raw_scores(regression_case) -> None:
    model, rows = regression_case
    generated = python_predictions(parse_lgbm_model(model, "python"), rows)
    expected = model.predict(rows, raw_score=True)
    np.testing.assert_allclose(generated, expected, rtol=1e-12, atol=1e-12)


def test_accepts_native_booster(regression_case) -> None:
    model, rows = regression_case
    code = parse_lgbm_model(model.booster_, "python", function_name="score")
    namespace: dict[str, object] = {}
    exec(code, namespace)
    generated = np.asarray([namespace["score"](row) for row in rows])
    np.testing.assert_allclose(
        generated, model.predict(rows, raw_score=True), rtol=1e-12, atol=1e-12
    )


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")
def test_cpp_compiles_and_matches_lightgbm(regression_case, tmp_path: Path) -> None:
    model, rows = regression_case
    code = parse_lgbm_model(model, "cpp")
    row_literals = ",\n".join(
        "{" + ",".join("NAN" if np.isnan(value) else repr(float(value)) for value in row) + "}"
        for row in rows
    )
    source = tmp_path / "model.cpp"
    source.write_text(
        code
        + "\n#include <iomanip>\n#include <iostream>\n"
        + "int main() {\n"
        + f"  std::vector<std::vector<double>> rows = {{{row_literals}}};\n"
        + '  std::cout << std::setprecision(17);\n'
        + '  for (const auto& row : rows) std::cout << lgbminfer(row) << "\\n";\n'
        + "}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "model"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    generated = np.asarray([float(value) for value in completed.stdout.splitlines()])
    np.testing.assert_allclose(
        generated, model.predict(rows, raw_score=True), rtol=1e-12, atol=1e-12
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_runs_and_matches_lightgbm(regression_case, tmp_path: Path) -> None:
    model, rows = regression_case
    module = tmp_path / "model.mjs"
    serializable = [
        [None if np.isnan(value) else float(value) for value in row] for row in rows
    ]
    # JSON has no NaN. Convert null back to NaN inside the JavaScript fixture.
    module.write_text(
        parse_lgbm_model(model, "javascript")
        + f"\nconst encoded = {json.dumps(serializable)};\n"
        + "for (const row of encoded) {\n"
        + "  const values = row.map((value) => value === null ? Number.NaN : value);\n"
        + "  console.log(lgbminfer(values).toPrecision(17));\n"
        + "}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(module)], check=True, capture_output=True, text=True
    )
    generated = np.asarray([float(value) for value in completed.stdout.splitlines()])
    np.testing.assert_allclose(
        generated, model.predict(rows, raw_score=True), rtol=1e-12, atol=1e-12
    )


def test_rejects_unsupported_language(regression_case) -> None:
    model, _ = regression_case
    with pytest.raises(ValueError, match="unsupported language"):
        parse_lgbm_model(model, "rust")


def test_rejects_invalid_function_name(regression_case) -> None:
    model, _ = regression_case
    with pytest.raises(ValueError, match="invalid generated function name"):
        parse_lgbm_model(model, "python", function_name="bad-name")


def test_rejects_unfitted_sklearn_model() -> None:
    with pytest.raises(TypeError, match="fitted"):
        parse_lgbm_model(lgb.LGBMRegressor(), "python")


def test_rejects_multiclass_model() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(60, 4))
    target = np.arange(60) % 3
    model = lgb.LGBMClassifier(n_estimators=3, verbosity=-1, random_state=8)
    model.fit(features, target)
    with pytest.raises(ValueError, match="multiclass"):
        parse_lgbm_model(model, "python")


def test_zero_missing_routing_follows_default_branch() -> None:
    model = DumpedModel(
        {
            "num_class": 1,
            "tree_info": [
                {
                    "tree_structure": {
                        "split_feature": 0,
                        "threshold": 1.5,
                        "decision_type": "<=",
                        "missing_type": "Zero",
                        "default_left": True,
                        "left_child": {"leaf_value": 2.0},
                        "right_child": {"leaf_value": -1.0},
                    }
                }
            ],
        }
    )
    generated = python_predictions(
        parse_lgbm_model(model, "python"), np.asarray([[0.0], [np.nan], [2.0]])
    )
    np.testing.assert_array_equal(generated, np.asarray([2.0, 2.0, -1.0]))
