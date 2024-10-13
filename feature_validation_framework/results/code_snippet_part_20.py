# Code snippet part 20
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates the given feature set based on predefined rules.

    Args:
        feature_set (pd.DataFrame): The feature set to validate.

    Returns:
        dict: A dictionary containing the validation outcomes.
    """

    # Initialize validation results
    validation_results = {}

    try:
        # 1. Feature count
        total_features = feature_set.shape[1]
        validation_results["Total Features"] = total_features

        # 2. Missing values count per column
        missing_values_count = feature_set.isna().sum()
        validation_results["Missing Values Count"] = missing_values_count.to_dict()

        # 3. Outlier count per column
        # Here, outliers are defined as values that are more than 3 standard deviations away from the mean
        outlier_count = (abs(feature_set - feature_set.mean()) > (3 * feature_set.std())).sum()
        validation_results["Outlier Count"] = outlier_count.to_dict()

    except Exception as e:
        # Log the error in the validation results
        validation_results["Error"] = str(e)

    return validation_results


def main():
    """
    Calls the validation function and prints the outcomes.
    """

    # File path
    feature_set_file = "C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv"

    try:
        # Load the feature set
        feature_set = pd.read_csv(feature_set_file)

        # Validate the feature set
        validation_results = validate_feature_set(feature_set)

        # Print the validation outcomes
        print("Validation Outcomes:")
        for key, value in validation_results.items():
            print(f"{key}: {value}")

        # Save the validation outcomes to a CSV file
        validation_results_file = "feature_set_validation_outcomes_4.csv"
        pd.DataFrame(validation_results, index=[0]).to_csv(validation_results_file)

    except Exception as e:
        # Log the error
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
