# Code snippet part 15
import os
import pandas as pd

def validate_feature_set(feature_set_path):
    """
    Validates a feature set based on a predefined set of rules.

    Args:
        feature_set_path (str): The path to the feature set CSV file.

    Returns:
        dict: A dictionary containing the validation results.
    """

    # Ensure the feature set file exists
    if not os.path.isfile(feature_set_path):
        print(f"Error: File {feature_set_path} not found.")
        return

    # Load the feature set into a DataFrame
    try:
        feature_set = pd.read_csv(feature_set_path)
    except Exception as e:
        print(f"Error loading feature set: {e}")
        return

    # Check if the DataFrame has the necessary columns
    necessary_columns = ["feature_name", "data_type", "missing_values"]
    for column in necessary_columns:
        if column not in feature_set.columns:
            print(f"Error: Missing column {column} in feature set.")
            return

    # Validate the feature set
    validation_results = {
        "total_features": len(feature_set),
        "missing_values_per_column": {},
        "outliers_detected": 0,
    }

    for column in feature_set.columns:
        # Count missing values
        missing_values = feature_set[column].isnull().sum()
        validation_results["missing_values_per_column"][column] = missing_values

        # Check for outliers (assuming numerical data)
        if feature_set[column].dtype == "float64" or feature_set[column].dtype == "int64":
            iqr = feature_set[column].quantile(0.75) - feature_set[column].quantile(0.25)
            lower_bound = feature_set[column].quantile(0.25) - (1.5 * iqr)
            upper_bound = feature_set[column].quantile(0.75) + (1.5 * iqr)
            outliers = feature_set[(feature_set[column] < lower_bound) | (feature_set[column] > upper_bound)].shape[0]
            validation_results["outliers_detected"] += outliers

    # Save the validation results to a CSV file
    try:
        validation_results_path = feature_set_path.replace(".csv", "_validation_outcomes_l.csv")
        validation_results_df = pd.DataFrame(validation_results)
        validation_results_df.to_csv(validation_results_path, index=False)
    except Exception as e:
        print(f"Error saving validation results: {e}")

    return validation_results


def main():
    """
    Calls the validation function and prints the outcomes.
    """

    feature_set_path = "C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv"
    validation_results = validate_feature_set(feature_set_path)

    if validation_results is not None:
        print(validation_results)


if __name__ == "__main__":
    main()
