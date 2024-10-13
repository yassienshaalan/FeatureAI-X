# Code snippet part 28
import pandas as pd
import os
import csv

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validate a pandas DataFrame 'feature_set' based on the specified validation rules.

    Args:
        feature_set (pd.DataFrame): The DataFrame to be validated.

    Returns:
        dict: A dictionary containing the validation outcomes.
    """

    # Initialize the validation results dictionary
    validation_results = {}

    try:
        # Check if the necessary columns exist in the DataFrame
        necessary_columns = ["feature1", "feature2", "target"]
        missing_columns = [col for col in necessary_columns if col not in feature_set.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Count the total number of features
        validation_results["Total Features"] = feature_set.shape[1]

        # Count the number of missing values per column
        missing_values = feature_set.isnull().sum()
        validation_results["Missing Values per Column"] = missing_values.to_dict()

        # Detect outliers using the Z-score method
        z_scores = abs(feature_set - feature_set.mean()) / feature_set.std()
        outliers = z_scores[z_scores > 3].count()
        validation_results["Outliers Detected"] = outliers

    except Exception as e:
        validation_results["Error"] = str(e)

    return validation_results


def main():
    try:
        # Load the feature set DataFrame
        feature_set = pd.read_csv("C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv")
    except FileNotFoundError:
        print("Error: File 'C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv' not found.")
        return

    # Validate the feature set
    validation_results = validate_feature_set(feature_set)

    # Save the validation results to a CSV file
    with open("feature_set_validation_outcomes_b.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(validation_results.keys())
        writer.writerow(validation_results.values())

    # Print the validation outcomes
    print("Validation Outcomes:")
    for key, value in validation_results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
