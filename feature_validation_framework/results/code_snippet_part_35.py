# Code snippet part 35
import pandas as pd
import os

def validate_feature_set(feature_set):
    """
    Performs data validation on a pandas DataFrame and generates validation outcomes.

    Args:
        feature_set (pandas.DataFrame): The DataFrame to validate.

    Returns:
        dict: A dictionary containing the validation outcomes.
    """

    validation_results = {}

    # Handle missing columns or missing files gracefully
    try:
        # Ensure the DataFrame contains the necessary columns
        required_columns = ['Feature 1', 'Feature 2']
        for column in required_columns:
            if column not in feature_set.columns:
                validation_results[f'Missing column: {column}'] = True
                continue
    except Exception as e:
        validation_results['Error: Could not validate schema'] = str(e)
        return validation_results

    # Perform necessary checks and store validation outcomes

    # Count of total features
    validation_results['Total features'] = feature_set.shape[1]

    # Count of missing values per column
    validation_results['Missing values per column'] = feature_set.isnull().sum()

    # Count of outliers detected: This logic is to be customized based on the actual rules for detecting outliers.
    validation_results['Outliers detected'] = feature_set[(feature_set < -3) | (feature_set > 3)].count()

    return validation_results

def main():
    """
    Calls the validation function and prints the outcomes.
    """

    # Load the DataFrame
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except Exception as e:
        print(f'Error loading feature set: {str(e)}')
        return

    # Perform validation and store outcomes
    validation_outcomes = validate_feature_set(feature_set)

    # Print validation outcomes
    for key, value in validation_outcomes.items():
        print(f'{key}: {value}')

    # Save validation results to CSV file
    try:
        validation_outcomes_df = pd.DataFrame(validation_outcomes, index=[0])
        validation_outcomes_df.to_csv('feature_set_validation_outcomes_P.csv', index=False)
        print('Validation results saved to feature_set_validation_outcomes_P.csv')
    except Exception as e:
        print(f'Error saving validation results: {str(e)}')

if __name__ == '__main__':
    main()
