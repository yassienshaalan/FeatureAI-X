# Code snippet part 2
import pandas as pd
import os

def validate_feature_set(feature_set):
    """Validate a pandas DataFrame 'feature_set' based on given rules.

    Args:
        feature_set (pandas.DataFrame): DataFrame to validate.

    Returns:
        dict: Dictionary containing validation outcomes and metrics.
    """
    # Initialize validation results
    validation_results = {}

    # 1. Check if DataFrame contains necessary columns
    required_columns = ['feature_1', 'feature_2', 'feature_3']
    missing_columns = [column for column in required_columns if column not in feature_set.columns]
    if missing_columns:
        validation_results['Missing Columns'] = missing_columns

    # 2. Count total features
    validation_results['Total Features'] = len(feature_set.columns)

    # 3. Count missing values per column
    missing_values_per_column = feature_set.isnull().sum()
    validation_results['Missing Values per Column'] = missing_values_per_column.to_dict()

    # 4. Handle outliers (assuming numeric features)
    numeric_features = feature_set.select_dtypes(include=['number']).columns
    for feature in numeric_features:
        try:
            iqr = feature_set[feature].quantile(0.75) - feature_set[feature].quantile(0.25)
            lower_bound = feature_set[feature].quantile(0.25) - 1.5 * iqr
            upper_bound = feature_set[feature].quantile(0.75) + 1.5 * iqr
            outliers = feature_set[feature][(feature_set[feature] < lower_bound) | (feature_set[feature] > upper_bound)]
            validation_results['Count of Outliers in ' + feature] = len(outliers)
        except KeyError:
            validation_results['Outlier Detection Error for ' + feature] = 'Column not found or not numeric'

    # 5. Other relevant statistics (e.g., data types, correlations)
    validation_results['Data Types'] = feature_set.dtypes.to_dict()
    correlation_matrix = feature_set.corr().abs()
    validation_results['Correlation Matrix'] = correlation_matrix.to_dict()

    # Save validation results to CSV file
    try:
        validation_results_path = 'feature_set_validation_outcomes_6.csv'
        pd.DataFrame(validation_results).to_csv(validation_results_path, index=False)
    except Exception as e:
        validation_results['CSV Saving Error'] = e

    return validation_results

def main():
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
        validation_outcomes = validate_feature_set(feature_set)
        print('Feature Set Validation Outcomes:')
        print(validation_outcomes)
    except Exception as e:
        print('Error occurred during validation:', e)

if __name__ == "__main__":
    main()
