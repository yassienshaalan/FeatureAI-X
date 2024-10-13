# Code snippet part 44
import pandas as pd
import os

def validate_feature_set(feature_set):
    validation_results = []

    # Ensure the DataFrame contains the necessary columns for the validation rules
    try:
        for column in ['Feature', 'Rule', 'Outcome', 'Metric']:
            if column not in feature_set.columns:
                validation_results.append(f'Missing column: {column}')
                raise KeyError
    except KeyError:
        pass

    # Perform the necessary checks and store validation outcomes in a dictionary, including relevant metrics
    for index, row in feature_set.iterrows():
        feature = row['Feature']
        rule = row['Rule']
        metric = row['Metric']
        outcome = 'Pass'
        try:
            if rule == 'Count of missing values':
                missing_count = feature_set[feature].isnull().sum()
                if missing_count > 0:
                    outcome = 'Fail'
                metric = missing_count
            # Add other validation checks here, based on the validation rules
            validation_results.append({
                'Feature': feature,
                'Rule': rule,
                'Outcome': outcome,
                'Metric': metric
            })
        except Exception as e:
            validation_results.append({
                'Feature': feature,
                'Rule': rule,
                'Outcome': 'Error',
                'Metric': str(e)
            })

    return pd.DataFrame(validation_results)

def main():
    # Read the feature set from a file
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except FileNotFoundError:
        validation_results = [{'Feature': 'File not found', 'Rule': 'File exists', 'Outcome': 'Fail', 'Metric': ''}]
        pd.DataFrame(validation_results).to_csv('feature_set_validation_outcomes_f.csv', index=False)
        print('Error: File not found')
        return

    # Validate the feature set
    validation_outcomes = validate_feature_set(feature_set)

    # Save the validation results to a CSV file
    try:
        validation_outcomes.to_csv('feature_set_validation_outcomes_f.csv', index=False)
    except Exception as e:
        print(f'Error saving validation results: {e}')

    # Print the validation outcomes on console
    print(validation_outcomes)

if __name__ == '__main__':
    main()
