# Changelog

## 0.3.0 — unreleased

- Verify raw-score parity by executing Python and JavaScript and compiling C++17.
- Generate helper functions before their callers in all target languages.
- Preserve double-precision thresholds and leaf values.
- Match LightGBM routing for NaN and zero-as-missing numerical splits.
- Accept native `Booster` and fitted sklearn-style LightGBM estimators.
- Reject multiclass, categorical, malformed, and unfitted inputs explicitly.
- Validate generated function identifiers.
- Add CI across Python 3.10 and 3.12.

