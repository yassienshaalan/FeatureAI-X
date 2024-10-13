# Code snippet part 11
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates the given feature set based on the specified data validation rules.

    Args:
        feature_set (pd.DataFrame): The feature set to validate.

    Returns:
        dict: A dictionary containing the validation outcomes.
    """

    # 1. Ensure the DataFrame contains the necessary columns.
    required_columns = ['feature_name', 'data_type', 'min_value', 'max_value']
    missing_columns = set(required_columns) - set(feature_set.columns)
    if missing_columns:
        validation_results['errors'].append(f"Missing columns: {', '.join(missing_columns)}")

    # 2. Handle missing values per column.
    validation_results['missing_values'] = feature_set.isnull().sum().to_dict()

    # 3. Perform the necessary checks.
    validation_results['total_features'] = len(feature_set)

    # 4. Count outliers detected (assuming data_type is numeric).
    for column in feature_set.select_dtypes(include=[np.number]).columns:
        validation_results[f'outliers_{column}'] = (
            feature_set[column] < feature_set[f'{column}_min']
        ) | (
            feature_set[column] > feature_set[f'{column}_max']
        )

    # 5. Save the validation results to a CSV file.
    try:
        validation_results.to_csv('feature_set_validation_outcomes_1.csv', index=False)
    except Exception as e:
        validation_results['errors'].append(f"Error saving results to CSV: {e}")

    return validation_results

def main():
    """
    Calls the validate_feature_set function and prints the validation outcomes.
    """

    # Read the feature set from a file.
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f"Error reading feature set: {e}")
        return

    # Validate the feature set.
    validation_results = validate_feature_set(feature_set)

    # Print the validation outcomes.
    print(validation_results)

if __name__ == "__main__":
    main()
