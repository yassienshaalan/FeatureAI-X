# Code snippet part 43
import pandas as pd
import numpy as np
import os

def validate_feature_set(feature_set):
    """
    Validates a pandas DataFrame 'feature_set' based on given rules.

    Args:
        feature_set (pd.DataFrame): The DataFrame to validate.

    Returns:
        dict: A dictionary of validation outcomes.
    """
    validation_results = {
        "total_features": feature_set.shape[1],
        "missing_values_per_column": feature_set.isnull().sum(),
        "count_of_outliers": 0,
    }

    # Check for outliers
    for col in feature_set.columns:
        try:
            q1 = feature_set[col].quantile(0.25)
            q3 = feature_set[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            validation_results["count_of_outliers"] += ((feature_set[col] < lower_bound) | (feature_set[col] > upper_bound)).sum()
        except Exception as e:
            validation_results["error"] = str(e)

    # Save the validation results to a CSV file
    results_file = "feature_set_validation_outcomes_{}.csv".format(datetime.now().strftime("%Y%m%d%H%M%S"))
    feature_set.to_csv(results_file, index=False)
    validation_results["validation_file"] = results_file

    return validation_results

def main():
    if os.path.exists("C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv"):
        try:
            feature_set = pd.read_csv("C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv")
            validation_results = validate_feature_set(feature_set)
            print(validation_results)
        except Exception as e:
            print("Error: {}".format(str(e)))
    else:
        print("Error: 'C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv' not found.")


if __name__ == "__main__":
    main()
