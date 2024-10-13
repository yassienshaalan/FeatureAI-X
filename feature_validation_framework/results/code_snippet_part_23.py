# Code snippet part 23
import os
import pandas as pd

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates a pandas DataFrame named 'feature_set' based on the following rules:
    1. The DataFrame 'feature_set' contains the necessary columns for the validation rules.
    2. Handle missing columns or missing files gracefully, **without using exit(). Instead, log the errors into the validation results list**.
    3. Perform the necessary checks, and store validation outcomes in a dictionary, including relevant metrics.
    4. Ensure that the validation metrics include:
    - Count of total features
    - Count of missing values per column
    - Count of outliers detected
    - Any other relevant statistics based on the rules
    5. Save the validation results to a CSV file named feature_set_validation_outcomes_v.csv.

    Args:
        feature_set (pd.DataFrame): The DataFrame to be validated.

    Returns:
        dict: A dictionary containing the validation outcomes.
    """

    validation_results = {}

    # Check if the necessary columns are present
    try:
        if not set(feature_set.columns).issuperset({'feature_name', 'feature_value'}):
            validation_results['error'] = 'Missing required columns: feature_name, feature_value'
    except Exception as e:
        validation_results['error'] = f'Error checking columns: {e}'

    # Check for missing values
    validation_results['missing_values'] = feature_set.isnull().sum().to_dict()

    # Check for outliers
    # Assuming no specific outlier detection method is provided, here's a simple IQR-based approach
    iqr = feature_set['feature_value'].quantile(0.75) - feature_set['feature_value'].quantile(0.25)
    lower_bound = feature_set['feature_value'].quantile(0.25) - (1.5 * iqr)
    upper_bound = feature_set['feature_value'].quantile(0.75) + (1.5 * iqr)
    validation_results['outliers'] = (feature_set['feature_value'] < lower_bound) | (feature_set['feature_value'] > upper_bound).sum()

    # Save the validation results to a CSV file
    try:
        validation_results.to_csv('feature_set_validation_outcomes_v.csv', index=False)
    except Exception as e:
        validation_results['error'] = f'Error saving results: {e}'

    return validation_results

def main():
    """
    Calls the validation function and prints the outcomes.
    """

    # Load the feature set
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f'Error loading feature set: {e}')
        return

    # Validate the feature set
    validation_results = validate_feature_set(feature_set)

    # Print the validation results
    print(validation_results)

if __name__ == '__main__':
    main()
