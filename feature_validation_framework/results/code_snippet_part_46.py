# Code snippet part 46
import os
import pandas as pd

def validate_feature_set(feature_set):
    validation_results = []

    # Ensure the required columns are present
    required_columns = ['column_1', 'column_2', 'column_3']
    missing_columns = list(set(required_columns) - set(feature_set.columns))
    if missing_columns:
        validation_results.append(f'Missing required columns: {", ".join(missing_columns)}')

    # Count total features
    feature_count = len(feature_set.columns)
    validation_results.append(f'Total number of features: {feature_count}')

    # Count missing values per column
    missing_values_per_column = feature_set.isnull().sum()
    validation_results.append(f'Missing values per column: {missing_values_per_column.to_dict()}')

    # Count outliers
    for column in feature_set.columns:
        try:
            iqr = feature_set[column].quantile(0.75) - feature_set[column].quantile(0.25)
            lower_bound = feature_set[column].quantile(0.25) - (1.5 * iqr)
            upper_bound = feature_set[column].quantile(0.75) + (1.5 * iqr)
            outliers = feature_set[(feature_set[column] < lower_bound) | (feature_set[column] > upper_bound)]
            validation_results.append(f'Outliers detected in column {column}: {len(outliers)}')
        except Exception as e:
            validation_results.append(f'Error while detecting outliers in column {column}: {e}')

    # Save validation results to CSV file
    try:
        with open('feature_set_validation_outcomes_h.csv', 'w') as f:
            for result in validation_results:
                f.write(f'{result}\n')
    except Exception as e:
        validation_results.append(f'Error while saving validation results: {e}')

    return validation_results

def main():
    # Load the feature set
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f'Error while loading feature set: {e}')
        return

    # Validate the feature set
    validation_results = validate_feature_set(feature_set)

    # Print the validation outcomes
    for result in validation_results:
        print(result)

if __name__ == '__main__':
    main()
