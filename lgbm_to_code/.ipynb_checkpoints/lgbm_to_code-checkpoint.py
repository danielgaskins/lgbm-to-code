import lightgbm as lgb

def parse_ensemble_to_cpp(ensemble):
    # Function to parse tree structure
    def parse_tree(tree, increment=0):
        prefix_tabs = "\t" * (increment + 1)
        linetab = "\n\t" + prefix_tabs

        if 'split_feature' in tree:
            threshold = tree["threshold"]
            split_feature = tree["split_feature"]
            left_child = tree.get("left_child")
            right_child = tree.get("right_child")
            decision_type = tree["decision_type"]
            if decision_type == "<=":
                decision = f"x[{split_feature}] <= {threshold:.9f}"
            else:
                decision = f"x[{split_feature}] > {threshold:.9f}"
            statement = f"if ({decision}) {{ " + linetab + parse_tree(left_child, increment + 1) + f" \n{prefix_tabs}}} else {{ " + linetab + parse_tree(right_child, increment + 1) + " \n" + prefix_tabs + "}"
            return statement
        else:
            leaf_value = tree["leaf_value"]
            return f"return {leaf_value:.9f};"

    def parse_tree_as_function(tree, increment=0):
        content = parse_tree(tree)
        return f"float func_{increment}(const std::vector<float>& x) {{\n{content}\n}}"

    def apply_function_header_final(code, function_name, nb_trees):
        inference_code = "  return " + " + ".join([f"func_{i}(x)" for i in range(nb_trees)]) + ";"
        return f"float {function_name}(const std::vector<float>& x) {{\n{inference_code}\n{code}\n}}"

    # Replace {final_code} with the aggregated if-else statements
    nb_trees = len(ensemble)
    code = "\n".join(
        parse_tree_as_function(ensemble[i]["tree_structure"], i) for i in range(nb_trees))
    code = apply_function_header_final(code, "lgbminfer", nb_trees)
    return "#include <vector>\n" + code

def parse_ensemble_to_javascript(ensemble):
    # Function to parse tree structure
    def parse_tree(tree, increment=0):
        prefix_tabs = "\t" * (increment + 1)
        linetab = "\n\t" + prefix_tabs

        if 'split_feature' in tree:
            threshold = tree["threshold"]
            split_feature = tree["split_feature"]
            left_child = tree.get("left_child")
            right_child = tree.get("right_child")
            decision_type = tree["decision_type"]
            if decision_type == "<=":
                decision = f"x[{split_feature}] <= {threshold:.9f}"
            else:
                decision = f"x[{split_feature}] > {threshold:.9f}"
            statement = f"if ({decision}) {{ " + linetab + parse_tree(left_child, increment + 1) + f" \n{prefix_tabs}}} else {{ " + linetab + parse_tree(right_child, increment + 1) + " \n" + prefix_tabs + "}"
            return statement
        else:
            leaf_value = tree["leaf_value"]
            return f"return {leaf_value:.9f};"

    def parse_tree_as_function(tree, increment=0):
        content = parse_tree(tree)
        return f"const func_{increment} = (x) => {{\n{content}\n}};"

    def apply_function_header_final(code, function_name, nb_trees):
        inference_code = "  return " + " + ".join([f"func_{i}(x)" for i in range(nb_trees)]) + ";"
        return f"export const {function_name} = (x) => {{\n{inference_code}\n{code}\n}};"

    # Replace {final_code} with the aggregated if-else statements
    nb_trees = len(ensemble)
    code = "\n".join(
        parse_tree_as_function(ensemble[i]["tree_structure"], i) for i in range(nb_trees))
    code = apply_function_header_final(code, "lgbminfer", nb_trees)
    return code

def parse_ensemble_to_python(ensemble):
    # Function to parse tree structure
    def parse_tree(tree, increment=0):
        prefix_tabs = "\t"*(increment + 1)
        linetab = "\n\t" + prefix_tabs
    
        if 'split_feature' in tree:
            threshold = tree["threshold"]
            split_feature = tree["split_feature"]
            left_child = tree.get("left_child")
            right_child = tree.get("right_child")
            decision_type = tree["decision_type"]
            if decision_type == "<=":
                decision = f"x[{split_feature}] <= {threshold:.9f}"
            else:
                decision = f"x[{split_feature}] > {threshold:.9f}"
            statement = f"if {decision}: " + linetab + parse_tree(left_child, increment+1) + f" \n{prefix_tabs}else: " + linetab + parse_tree(right_child, increment + 1)
            if increment == 0:
                statement = "\t" + statement
            return statement
        else:
            leaf_value = tree["leaf_value"]
            return f"return {leaf_value:.9f}"
    
    def parse_tree_as_function(tree, increment=0):
        content = parse_tree(tree)
        return "\n" + apply_function_header(content, f"func_{increment}")
    
        
    def apply_function_header(code, function_name):
        return f"def {function_name}(x): \n{code}"
        
    def apply_function_header_final(code, function_name, nb_trees):
        inference_code = "\treturn " + "+".join([f"func_{i}(x)" for i in range(nb_trees)])
        return f"def {function_name}(x):\n{inference_code}\n{code}"
    
    # Replace {final_code} with the aggregated if-else statements
    nb_trees = len(ensemble)
    code = "\n".join(parse_tree_as_function(ensemble[i]["tree_structure"], i) for i in range(nb_trees))
    code = apply_function_header_final(code, "lgbminfer", nb_trees)
    return code

def parse_lgbm_model(booster: lgb.Booster, language: str):
    """
    Parses an LGBM booster object and returns the model code in the specified language.

    Args:
        booster (lgb.Booster): The LGBM booster object.
        language (str): The desired output language ("python", "cpp", or "javascript").

    Returns:
        str: The model code in the specified language.

    Raises:
        ValueError: If an unsupported language is provided.
    """
    ensemble = booster.dump_model()["tree_info"]

    if language == "python":
        return parse_ensemble_to_python(ensemble)
    elif language == "cpp":
        return parse_ensemble_to_cpp(ensemble)
    elif language == "javascript":
        return parse_ensemble_to_javascript(ensemble)
    else:
        raise ValueError(f"Unsupported language: {language}")