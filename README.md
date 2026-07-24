# lgbm-to-code

Generate dependency-free raw-score inference code from a trained, one-output
LightGBM model.

The generated function is plain source code and does not need Python or LightGBM
at inference time:

- Python
- C++17
- JavaScript (ES module)

Version 0.3 focuses on a narrow promise that can be tested rigorously: generated
code follows the same numerical tree paths and returns the same raw score as
LightGBM for supported models.

## Why

Tree models are often trained in Python and then need to run in a browser, a small
service, a compiled application, or another environment where shipping the full
LightGBM runtime is undesirable. `lgbm-to-code` turns the learned trees into
readable conditionals that can be reviewed, compiled, and embedded directly.

## Installation

```bash
pip install lgbm-to-code
```

The unreleased development version can be installed from a clone:

```bash
pip install -e ".[test]"
```

## Usage

```python
import lightgbm as lgb
from lgbm_to_code import parse_lgbm_model

model = lgb.LGBMRegressor(n_estimators=25, random_state=42)
model.fit(X_train, y_train)

python_source = parse_lgbm_model(model, "python")
cpp_source = parse_lgbm_model(model, "cpp")
javascript_source = parse_lgbm_model(model, "javascript")
```

The generated function is named `lgbminfer` by default. A safe custom identifier
can be supplied:

```python
source = parse_lgbm_model(model.booster_, "python", function_name="score_row")
```

## Output contract

The generated function returns `predict(..., raw_score=True)` for supported
one-output models.

- For ordinary regression objectives, the raw score is normally the prediction.
- For binary classification, the raw score is the logit. Apply the model's
  objective transform when a probability is required.

The package does not silently guess output semantics. This is deliberate: exact
tree traversal and objective-specific post-processing are separate concerns.

## Numerical verification

The automated suite trains a LightGBM regression model with missing values, emits
all three target languages, executes the Python and JavaScript, compiles and runs
the C++17, and compares every result to LightGBM raw scores with `rtol=1e-12` and
`atol=1e-12`.

Run it with:

```bash
python -m pytest
```

CI runs the suite on Python 3.10 and 3.12. JavaScript execution uses Node 22 and
C++ is compiled with `g++ -std=c++17`.

## Supported behavior

- One-output LightGBM `Booster` objects.
- Fitted LightGBM sklearn estimators through `booster_`.
- Numerical `<=` splits.
- LightGBM `None`, `NaN`, and `Zero` missing-value routing.
- Configurable generated function names with identifier validation.
- Full double-precision literals in generated source.

## Explicit limitations

- Multiclass ensembles are rejected.
- Categorical splits are rejected.
- Probability and other objective transforms are not generated.
- Generated code favors auditability and portability; it is not yet optimized for
  code size or latency.
- Callers remain responsible for feature ordering and schema validation.

Rejecting unsupported models is safer than emitting plausible-looking code with
different behavior.

## License

MIT
