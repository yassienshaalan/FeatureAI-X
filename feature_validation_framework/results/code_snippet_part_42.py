# Code snippet part 42
import pandas as pd
import numpy as np
import os

def validate_feature_set(feature_set):
    """Validate a pandas DataFrame 'feature_set' based on various criteria.

    :param feature_set: Pandas DataFrame to validate
    :return: None
    """

    validation_results = {}

    # Check for necessary columns
    required_columns = ['column1', 'column2', 'column3']
    missing_columns = set(required_columns) - set(feature_set.columns)
    if missing_columns:
        validation_results['Missing Columns'] = list(missing_columns)

    # Count total features
    validation_results['Total Features'] = feature_set.shape[1]

    # Count missing values per column
    validation_results['Missing Values per Column'] = feature_set.isnull().sum()

    # Detect outliers using z-score
    z_scores = np.abs(stats.zscore(feature_set))
    outlier_count = (z_scores > 3).sum().sum()
    validation_results['Outliers Detected'] = outlier_count

    # Other relevant statistics
    validation_results['Mean'] = feature_set.mean()
    validation_results['Standard Deviation'] = feature_set.std()

    return validation_results

def main():
    """Execute the feature validation and save the results to a file."""

    try:
        if os.path.exists('feature_set_validation_outcomes_5.csv'):
            os.remove('feature_set_validation_outcomes_5.csv')
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
        validation_results = validate_feature_set(feature_set)
        validation_results_df = pd.DataFrame(validation_results)
        validation_results_df.to_csv('feature_set_validation_outcomes_5.csv', index=False)
        print("Validation results saved to feature_set_validation_outcomes_5.csv")
    except Exception as e:
        print("An error occurred during validation:", e)

if __name__ == "__main__":
    main()
