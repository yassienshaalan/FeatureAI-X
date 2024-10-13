# Code snippet part 10
import pandas as pd
import os
import numpy as np

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates a pandas DataFrame 'feature_set' based on the given data validation rules.

    Args:
        feature_set: The pandas DataFrame to be validated.

    Returns:
        validation_results: A dictionary containing validation outcomes and relevant metrics.
    """

    validation_results = {}
    error_list = []

    try:
        # Ensure necessary columns are present
        required_columns = ['feature_1', 'feature_2', 'feature_3']
        missing_columns = [column for column in required_columns if column not in feature_set.columns]

        if missing_columns:
            error_list.append(f"Missing columns: {missing_columns}")
            validation_results['validation_status'] = False
            return validation_results

        # Count of total features
        validation_results['total_features'] = len(feature_set.columns)

        # Count of missing values per column
        validation_results['missing_values'] = feature_set.isnull().sum()

        # Count of outliers detected
        for feature in feature_set.columns:
            iqr = feature_set[feature].quantile(0.75) - feature_set[feature].quantile(0.25)
            lower_bound = feature_set[feature].quantile(0.25) - (1.5 * iqr)
            upper_bound = feature_set[feature].quantile(0.75) + (1.5 * iqr)
            validation_results[f'outliers_{feature}'] = feature_set[(feature_set[feature] < lower_bound) | (feature_set[feature] > upper_bound)].shape[0]

        # Other relevant statistics
        validation_results['data_type_counts'] = feature_set.dtypes.value_counts().to_dict()
        validation_results['unique_value_counts'] = feature_set.nunique().to_dict()
        validation_results['correlation_matrix'] = feature_set.corr().to_dict()

        validation_results['validation_status'] = True
        return validation_results

    except Exception as e:
        error_list.append(f"Error during validation: {str(e)}")
        validation_results['validation_status'] = False
        validation_results['error_list'] = error_list
        return validation_results

def main():
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
        validation_results = validate_feature_set(feature_set)
        validation_results['error_list'] = [] if 'error_list' not in validation_results else validation_results['error_list']

        # Save validation results to CSV
        pd.DataFrame(validation_results).to_csv('feature_set_validation_outcomes_w.csv', index=False)

        print("Validation completed successfully.")
        print("Validation results:")
        print(validation_results)

    except Exception as e:
        print(f"Error during validation: {str(e)}")

if __name__ == '__main__':
    main()
