# Code snippet part 4
import pandas as pd
import os
import numpy as np

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates a Pandas DataFrame containing feature data.

    Args:
        feature_set (pd.DataFrame): The DataFrame containing feature data.

    Returns:
        dict: A dictionary of validation outcomes.
    """
    validation_results = {
        "total_features": feature_set.shape[1],
        "missing_values_per_column": feature_set.isnull().sum().to_dict(),
        "outliers_detected": 0,  # Placeholder for outlier detection logic
        "other_statistics": {},  # Placeholder for additional statistics
    }

    try:
        # Check for outliers using IQR method
        iqr = feature_set.quantile(0.75) - feature_set.quantile(0.25)
        lower_bound = feature_set.quantile(0.25) - (1.5 * iqr)
        upper_bound = feature_set.quantile(0.75) + (1.5 * iqr)
        validation_results["outliers_detected"] = (
            (feature_set < lower_bound) | (feature_set > upper_bound)
        ).sum().sum()

    except Exception as e:
        validation_results["error"] = str(e)

    return validation_results

def main():
    try:
        if os.path.exists('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv'):
            feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
            validation_outcomes = validate_feature_set(feature_set)
            pd.DataFrame(validation_outcomes).to_csv('feature_set_validation_outcomes_2.csv', index=False)
            print(f"Validation outcomes saved to feature_set_validation_outcomes_2.csv")
        else:
            print("File C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv does not exist.")
    except Exception as e:
        print(f"An error occurred during validation: {str(e)}")

if __name__ == "__main__":
    main()
