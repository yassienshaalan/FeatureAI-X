# Code snippet part 51
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates a pandas DataFrame based on specified rules.

    Args:
        feature_set: The pandas DataFrame to be validated.

    Returns:
        A dictionary of validation outcomes, including relevant metrics.
    """

    # 1. Ensure the DataFrame contains the necessary columns for the validation rules.
    required_columns = ['feature_name', 'feature_type', 'missing_values', 'outliers']
    missing_columns = list(set(required_columns) - set(feature_set.columns))

    if missing_columns:
        validation_results['errors'].append(f"Missing columns: {', '.join(missing_columns)}")

    # 2. Handle missing columns or missing files gracefully, without using exit(). Instead, log the errors into the validation results list.
    if feature_set is None or feature_set.empty:
        validation_results['errors'].append("DataFrame is empty or does not exist")

    # 3. Perform the necessary checks, and store validation outcomes in a dictionary, including relevant metrics.
    validation_results = {
        'total_features': feature_set.shape[1],
        'missing_values': feature_set['missing_values'].sum(),
        'outliers': feature_set['outliers'].sum(),
        'errors': []
    }

    # 4. Ensure that the validation metrics include:
    # - Count of total features
    # - Count of missing values per column
    # - Count of outliers detected
    # - Any other relevant statistics based on the rules

    # 5. Save the validation results to a CSV file named feature_set_validation_outcomes_a.csv.
    try:
        if not os.path.exists('validation_results'):
            os.makedirs('validation_results')

        feature_set.to_csv('validation_results/feature_set_validation_outcomes_a.csv', index=False)

    except Exception as e:
        validation_results['errors'].append(f"Error saving validation results: {e}")

    return validation_results


def main():
    """
    Calls the validation function and prints the outcomes.
    """
    feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    validation_results = validate_feature_set(feature_set)
    print(validation_results)


if __name__ == '__main__':
    main()
