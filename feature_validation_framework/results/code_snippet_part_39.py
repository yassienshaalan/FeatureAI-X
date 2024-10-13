# Code snippet part 39
import pandas as pd
import os

def validate_feature_set(feature_set):
    """
    Validates the features in a pandas DataFrame.

    Args:
        feature_set (pandas.DataFrame): The DataFrame to validate.

    Returns:
        A dictionary of validation outcomes.
    """

    # Check if the necessary columns are present.
    try:
        for column in ['feature_name', 'min', 'max', 'mean', 'median', 'std']:
            if column not in feature_set.columns:
                raise KeyError(f"Column '{column}' is missing from the feature set.")
    except KeyError as e:
        validation_errors.append(str(e))

    # Count the total number of features.
    num_features = feature_set.shape[0]

    # Count the number of missing values per column.
    num_missing_values = feature_set.isnull().sum()

    # Detect outliers.
    z_scores = (feature_set[['min', 'max']] - feature_set[['mean', 'mean']]) / feature_set[['std', 'std']]
    num_outliers = (z_scores['min'] < -3) | (z_scores['max'] > 3).sum()

    # Store the validation outcomes in a dictionary.
    validation_outcomes = {
        'num_features': num_features,
        'num_missing_values': num_missing_values,
        'num_outliers': num_outliers
    }

    return validation_outcomes


def main():
    """
    Calls the validation function and prints the outcomes.
    """

    # Load the feature set.
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except FileNotFoundError as e:
        validation_errors.append(str(e))

    # Validate the feature set.
    validation_outcomes = validate_feature_set(feature_set)

    # Print the validation outcomes.
    print(validation_outcomes)

    # Save the validation outcomes to a CSV file.
    try:
        validation_outcomes.to_csv('feature_set_validation_outcomes_t.csv', index=False)
    except Exception as e:
        validation_errors.append(str(e))


if __name__ == '__main__':
    main()
