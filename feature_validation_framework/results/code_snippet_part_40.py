# Code snippet part 40
import pandas as pd
import os
import csv

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates the given DataFrame 'feature_set' based on the following rules:

    - Ensure the 'feature_set' contains the necessary columns for the validation rules.
    - Handle missing columns or missing files gracefully, without using exit(). Instead, log the errors into the validation results list.
    - Perform the necessary checks, and store validation outcomes in a dictionary, including relevant metrics.
    - Ensure that the validation metrics include:
        - Count of total features
        - Count of missing values per column
        - Count of outliers detected
        - Any other relevant statistics based on the rules
    - Save the validation results to a CSV file named feature_set_validation_outcomes_r.csv.

    Args:
        feature_set (pd.DataFrame): The DataFrame to be validated.

    Returns:
        dict: A dictionary containing the validation outcomes.
    """

    # Validation outcomes dictionary
    validation_results = {}

    try:
        # Ensure the DataFrame has the necessary columns
        required_columns = ['feature1', 'feature2', 'feature3']
        missing_columns = [column for column in required_columns if column not in feature_set.columns]
        if missing_columns:
            validation_results['Missing Columns'] = missing_columns
        else:
            # Count of total features
            validation_results['Total Features'] = len(feature_set.columns)

            # Count of missing values per column
            missing_values_per_column = feature_set.isnull().sum()
            validation_results['Missing Values per Column'] = dict(missing_values_per_column)

            # Count of outliers detected
            # Assuming z-score method for outlier detection with a threshold of 3
            z_scores = np.abs(stats.zscore(feature_set))
            outliers = (z_scores > 3).sum()
            validation_results['Count of Outliers'] = outliers

            # Save validation results to a CSV file
            with open('feature_set_validation_outcomes_r.csv', 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Validation Metric', 'Value'])
                for metric, value in validation_results.items():
                    csv_writer.writerow([metric, value])

    except Exception as e:
        validation_results['Error'] = str(e)

    return validation_results


def main():
    # Load the DataFrame 'feature_set'
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f'Error loading DataFrame: {e}')
        return

    # Validate the DataFrame
    validation_outcomes = validate_feature_set(feature_set)

    # Print the validation outcomes
    for metric, value in validation_outcomes.items():
        print(f'{metric}: {value}')


if __name__ == '__main__':
    main()
