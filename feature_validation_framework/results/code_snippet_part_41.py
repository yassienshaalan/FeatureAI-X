# Code snippet part 41
import pandas as pd
import os

def validate_feature_set(feature_set, validation_results):
    """
    Validates a pandas DataFrame 'feature_set' based on the following rules:
    1. Count of total features
    2. Count of missing values per column
    3. Count of outliers detected
    Saves the validation results to a CSV file 'feature_set_validation_outcomes_O.csv'.

    Args:
        feature_set: pandas DataFrame to be validated
        validation_results: list to store validation outcomes
    """
    try:
        # Ensure 'feature_set' contains necessary columns
        if not set(feature_set.columns).issuperset(['feature1', 'feature2', 'feature3']):
            validation_results.append('Missing columns in feature_set')
            return

        # Count of total features
        validation_results.append(f'Total features: {len(feature_set.columns)}')

        # Count of missing values per column
        missing_values = feature_set.isnull().sum()
        validation_results.extend([f'Missing values in {col}: {val}' for col, val in missing_values.items()])

        # Count of outliers detected (assuming any value outside 2 standard deviations is an outlier)
        outliers = (feature_set - feature_set.mean()) / feature_set.std().abs() > 2
        outlier_count = outliers.sum().sum()
        validation_results.append(f'Total outliers: {outlier_count}')

        # Save validation results to CSV file
        with open('feature_set_validation_outcomes_O.csv', 'w') as f:
            f.write('\n'.join(validation_results))

    except Exception as e:
        validation_results.append(f'Error occurred during validation: {e}')

def main():
    # Load 'C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv' into a DataFrame
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f'Error occurred loading C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv: {e}')
        return

    # Validation results list
    validation_results = []

    # Perform validation
    validate_feature_set(feature_set, validation_results)

    # Print validation outcomes
    print('\nValidation Outcomes:')
    for outcome in validation_results:
        print(outcome)

if __name__ == '__main__':
    main()
