# Code snippet part 5
import os
import pandas as pd
import numpy as np

def validate_feature_set(feature_set):
    """
    Validate a pandas DataFrame 'feature_set' based on the following rules:

    Args:
        feature_set (pandas.DataFrame): DataFrame to be validated

    Returns:
        validation_results (dict): Dictionary of validation outcomes
    """
    # Check if the necessary columns exist
    required_columns = ['feature_name', 'feature_values']
    missing_columns = [col for col in required_columns if col not in feature_set.columns]
    if missing_columns:
        validation_results = {'error': f'Missing required columns: {", ".join(missing_columns)}'}
        return validation_results

    # Count of total features
    total_features = len(feature_set)

    # Count of missing values per column
    missing_values_per_column = feature_set.isnull().sum()

    # Count of outliers detected
    iqr = feature_set['feature_values'].quantile(0.75) - feature_set['feature_values'].quantile(0.25)
    lower_bound = feature_set['feature_values'].quantile(0.25) - (iqr * 1.5)
    upper_bound = feature_set['feature_values'].quantile(0.75) + (iqr * 1.5)
    outliers = feature_set[(feature_set['feature_values'] < lower_bound) | (feature_set['feature_values'] > upper_bound)]
    count_outliers = len(outliers)

    # Additional validation rules
    # ...

    # Store validation outcomes in a dictionary
    validation_results = {
        'total_features': total_features,
        'missing_values_per_column': missing_values_per_column.to_dict(),
        'count_outliers': count_outliers,
        # Add any other relevant metrics here
    }

    # Save validation results to a CSV file
    try:
        validation_results_path = 'feature_set_validation_outcomes_N.csv'
        validation_results.to_csv(validation_results_path, index=False)
        print(f'Validation results saved to {validation_results_path}')
    except Exception as e:
        validation_results['error'] = f'Error saving validation results: {str(e)}'

    return validation_results

if __name__ == '__main__':
    # If 'os' or 'pandas' is not defined, import them
    if 'os' not in globals():
        import os
    if 'pandas' not in globals():
        import pandas as pd

    # Load the feature set DataFrame
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f'Error loading feature set: {str(e)}')
        exit(1)

    # Validate the feature set
    validation_results = validate_feature_set(feature_set)

    # Print the validation outcomes
    if 'error' in validation_results:
        print(f'Validation failed: {validation_results["error"]}')
    else:
        print('Validation successful.')
        for metric, value in validation_results.items():
            print(f'{metric}: {value}')
