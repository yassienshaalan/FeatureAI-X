# Code snippet part 53
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates a pandas DataFrame based on specified rules.

    Args:
        feature_set (pd.DataFrame): The DataFrame to validate.

    Returns:
        dict: A dictionary containing the validation results.
    """

    validation_results = {}

    # Check for missing columns
    required_columns = ['column1', 'column2', 'column3']  # Replace with actual required columns
    missing_columns = [column for column in required_columns if column not in feature_set.columns]
    if missing_columns:
        validation_results['Missing Columns'] = missing_columns

    # Check for missing values
    missing_values = feature_set.isnull().sum()
    validation_results['Missing Values per Column'] = missing_values.to_dict()

    # Check for outliers using IQR method
    iqr = feature_set.quantile(0.75) - feature_set.quantile(0.25)
    outliers = (feature_set < (feature_set.quantile(0.25) - 1.5 * iqr)) | (feature_set > (feature_set.quantile(0.75) + 1.5 * iqr))
    validation_results['Outliers Detected'] = outliers.sum().to_dict()

    # Other relevant statistics (if needed)
    validation_results['Total Features'] = len(feature_set.columns)

    return validation_results

def main():
    """
    Calls the validation function and prints the outcomes.
    """

    # Define the feature set (replace with actual DataFrame)
    feature_set = pd.DataFrame({
        'column1': [1, 2, 3, 4, 5],
        'column2': [6, 7, 8, 9, 10],
        'column3': [11, 12, 13, 14, 15]
    })

    # Validate the feature set
    validation_results = validate_feature_set(feature_set)

    # Save validation results to CSV file
    try:
        validation_results_df = pd.DataFrame(validation_results)
        validation_results_df.to_csv('feature_set_validation_outcomes_.csv', index=False)
        print('Validation results saved to feature_set_validation_outcomes_.csv')
    except Exception as e:
        print(f'Error saving validation results: {e}')

    # Print validation results
    print(validation_results)

if __name__ == '__main__':
    main()
