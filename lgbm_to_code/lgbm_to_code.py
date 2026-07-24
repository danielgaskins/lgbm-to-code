"""Generate dependency-free raw-score inference code from LightGBM trees."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


SUPPORTED_LANGUAGES = ("python", "cpp", "javascript")
_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"tree contains non-finite numeric value: {value!r}")
    return format(number, ".17g")


def _validate_function_name(function_name: str) -> None:
    if not _FUNCTION_NAME.fullmatch(function_name):
        raise ValueError(f"invalid generated function name: {function_name!r}")


def _model_dump(model: Any) -> Mapping[str, Any]:
    if hasattr(model, "dump_model"):
        dumped = model.dump_model()
    elif hasattr(model, "booster_") and hasattr(model.booster_, "dump_model"):
        dumped = model.booster_.dump_model()
    else:
        raise TypeError(
            "model must be a lightgbm.Booster or a fitted LightGBM sklearn model"
        )
    if not isinstance(dumped, Mapping):
        raise TypeError("LightGBM dump_model() must return a mapping")
    return dumped


def _ensemble(model_dump: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    num_class = int(model_dump.get("num_class", 1))
    if num_class != 1:
        raise ValueError("multiclass models are not supported; use a one-output model")
    trees = model_dump.get("tree_info")
    if not isinstance(trees, Sequence) or isinstance(trees, (str, bytes)) or not trees:
        raise ValueError("model contains no tree_info entries")
    for tree in trees:
        if not isinstance(tree, Mapping) or "tree_structure" not in tree:
            raise ValueError("each tree_info entry must contain tree_structure")
    return trees


def _node(node: Any) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ValueError("tree node must be a mapping")
    return node


def _split(node: Mapping[str, Any]) -> tuple[int, str, str, bool]:
    decision_type = node.get("decision_type")
    if decision_type != "<=":
        raise ValueError(
            f"unsupported decision_type {decision_type!r}; only numerical '<=' splits "
            "are supported"
        )
    try:
        feature = int(node["split_feature"])
        threshold = _number(node["threshold"])
    except KeyError as error:
        raise ValueError(f"split node is missing {error.args[0]!r}") from error
    missing_type = str(node.get("missing_type", "None"))
    if missing_type not in {"None", "NaN", "Zero"}:
        raise ValueError(f"unsupported missing_type: {missing_type!r}")
    return feature, threshold, missing_type, bool(node.get("default_left", False))


def _python_condition(feature: int, threshold: str, missing: str, default_left: bool) -> str:
    value = f"x[{feature}]"
    missing_test = "False"
    if missing == "NaN":
        missing_test = f"math.isnan({value})"
    elif missing == "Zero":
        missing_test = f"({value} == 0 or math.isnan({value}))"
    comparison = f"{value} <= {threshold}"
    if default_left:
        return f"({missing_test}) or ({comparison})"
    return f"not ({missing_test}) and ({comparison})"


def _cpp_condition(feature: int, threshold: str, missing: str, default_left: bool) -> str:
    value = f"x[{feature}]"
    missing_test = "false"
    if missing == "NaN":
        missing_test = f"std::isnan({value})"
    elif missing == "Zero":
        missing_test = f"({value} == 0.0 || std::isnan({value}))"
    comparison = f"{value} <= {threshold}"
    if default_left:
        return f"({missing_test}) || ({comparison})"
    return f"!({missing_test}) && ({comparison})"


def _javascript_condition(
    feature: int, threshold: str, missing: str, default_left: bool
) -> str:
    value = f"x[{feature}]"
    missing_test = "false"
    if missing == "NaN":
        missing_test = f"Number.isNaN({value})"
    elif missing == "Zero":
        missing_test = f"({value} === 0 || Number.isNaN({value}))"
    comparison = f"{value} <= {threshold}"
    if default_left:
        return f"({missing_test}) || ({comparison})"
    return f"!({missing_test}) && ({comparison})"


def _python_tree(node_value: Any, indent: int) -> list[str]:
    node = _node(node_value)
    prefix = "    " * indent
    if "split_feature" not in node:
        if "leaf_value" not in node:
            raise ValueError("leaf node is missing 'leaf_value'")
        return [f"{prefix}return {_number(node['leaf_value'])}"]
    feature, threshold, missing, default_left = _split(node)
    condition = _python_condition(feature, threshold, missing, default_left)
    lines = [f"{prefix}if {condition}:"]
    lines.extend(_python_tree(node.get("left_child"), indent + 1))
    lines.append(f"{prefix}else:")
    lines.extend(_python_tree(node.get("right_child"), indent + 1))
    return lines


def _braced_tree(node_value: Any, indent: int, language: str) -> list[str]:
    node = _node(node_value)
    prefix = "    " * indent
    if "split_feature" not in node:
        if "leaf_value" not in node:
            raise ValueError("leaf node is missing 'leaf_value'")
        return [f"{prefix}return {_number(node['leaf_value'])};"]
    feature, threshold, missing, default_left = _split(node)
    if language == "cpp":
        condition = _cpp_condition(feature, threshold, missing, default_left)
    else:
        condition = _javascript_condition(feature, threshold, missing, default_left)
    lines = [f"{prefix}if ({condition}) {{"]
    lines.extend(_braced_tree(node.get("left_child"), indent + 1, language))
    lines.append(f"{prefix}}} else {{")
    lines.extend(_braced_tree(node.get("right_child"), indent + 1, language))
    lines.append(f"{prefix}}}")
    return lines


def parse_ensemble_to_python(
    ensemble: Sequence[Mapping[str, Any]], function_name: str = "lgbminfer"
) -> str:
    """Generate Python code returning the ensemble's raw score."""
    _validate_function_name(function_name)
    blocks = ["import math"]
    for index, tree in enumerate(ensemble):
        lines = [f"def _tree_{index}(x):"]
        lines.extend(_python_tree(tree["tree_structure"], 1))
        blocks.append("\n".join(lines))
    total = " + ".join(f"_tree_{index}(x)" for index in range(len(ensemble)))
    blocks.append(f"def {function_name}(x):\n    return {total}")
    return "\n\n".join(blocks) + "\n"


def parse_ensemble_to_cpp(
    ensemble: Sequence[Mapping[str, Any]], function_name: str = "lgbminfer"
) -> str:
    """Generate C++17 code returning the ensemble's raw score."""
    _validate_function_name(function_name)
    blocks = ["#include <cmath>\n#include <vector>"]
    for index, tree in enumerate(ensemble):
        lines = [f"double _tree_{index}(const std::vector<double>& x) {{"]
        lines.extend(_braced_tree(tree["tree_structure"], 1, "cpp"))
        lines.append("}")
        blocks.append("\n".join(lines))
    total = " + ".join(f"_tree_{index}(x)" for index in range(len(ensemble)))
    blocks.append(
        f"double {function_name}(const std::vector<double>& x) {{\n"
        f"    return {total};\n"
        "}"
    )
    return "\n\n".join(blocks) + "\n"


def parse_ensemble_to_javascript(
    ensemble: Sequence[Mapping[str, Any]], function_name: str = "lgbminfer"
) -> str:
    """Generate JavaScript code returning the ensemble's raw score."""
    _validate_function_name(function_name)
    blocks: list[str] = []
    for index, tree in enumerate(ensemble):
        lines = [f"const _tree_{index} = (x) => {{"]
        lines.extend(_braced_tree(tree["tree_structure"], 1, "javascript"))
        lines.append("};")
        blocks.append("\n".join(lines))
    total = " + ".join(f"_tree_{index}(x)" for index in range(len(ensemble)))
    blocks.append(
        f"export const {function_name} = (x) => {{\n"
        f"    return {total};\n"
        "};"
    )
    return "\n\n".join(blocks) + "\n"


def parse_lgbm_model(
    model: Any, language: str, function_name: str = "lgbminfer"
) -> str:
    """Generate dependency-free code for a fitted one-output LightGBM model.

    The generated function returns LightGBM's raw score. For regression objectives
    this is normally the prediction. For binary classification it is the logit;
    apply the appropriate objective transform if probabilities are required.

    Numerical splits, including LightGBM's NaN and zero missing-value routing, are
    supported. Categorical and multiclass models are rejected explicitly.
    """
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"unsupported language {language!r}; choose one of: {supported}")
    ensemble = _ensemble(_model_dump(model))
    if language == "python":
        return parse_ensemble_to_python(ensemble, function_name)
    if language == "cpp":
        return parse_ensemble_to_cpp(ensemble, function_name)
    return parse_ensemble_to_javascript(ensemble, function_name)
