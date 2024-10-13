# Code snippet part 21
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:

    validation_results = {}

    try:
        # Check if necessary columns exist
        required_columns = ['column_1', 'column_2', 'column_3']  # Example required columns
        for column in required_columns:
            if column not in feature_set.columns:
                validation_results[f'Missing column: {column}'] = True
            else:
                validation_results[f'Missing column: {column}'] = False

        # Count of data missing
        missing_value_count_per_column = feature_set.isnull().sum()
        validation_results.update(missing_value_count_per_column.to_dict())

        # Check for outliers - Example rule using IQR
        iqr = feature_set.quantile(0.75) - feature_set.quantile(0.25)
        outliers = ((feature_set < (feature_set.quantile(0.25) - 1.5 * iqr)) | (feature_set > (feature_set.quantile(0.75) + 1.5 * iqr))).any()
        validation_results['Outliers detected'] = outliers.sum()

        # Count of total features
        validation_results['Total features'] = feature_set.shape[1]

        # Any other relevant statistics
        validation_results['Other statistic 1'] = 'Example Value 1'
        validation_results['Other statistic 2'] = 'Example Value 2'

    except Exception as e:
        validation_results['Error'] = str(e)

    return validation_results


def main():
    try:
        if not os.path.exists('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv'):
            raise FileNotFoundError("File 'C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv' not found")

        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv', index_col=0)
        validation_outcomes = validate_feature_set(feature_set)
        
        # Save validation results to CSV file
        validation_df = pd.DataFrame(validation_outcomes, index=[0])
        validation_df.to_csv('feature_set_validation_outcomes_3.csv', index=False)

        print("Validation Results:")
        print(validation_df)

    except FileNotFoundError as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
