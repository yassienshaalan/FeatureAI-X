# Code snippet part 18
import os
import pandas as pd

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
  """Validates a pandas DataFrame based on specified rules.

  Args:
    feature_set: The pandas DataFrame to validate.

  Returns:
    A dictionary containing the validation outcomes.
  """
  validation_results = {}
  try:
    # Check if necessary columns exist
    required_columns = ['feature_name', 'feature_type', 'missing_values']
    for col in required_columns:
      if col not in feature_set.columns:
        validation_results[f'Missing column: {col}'] = True

    # Count total features
    validation_results['Total features'] = feature_set.shape[1]

    # Count missing values per column
    validation_results['Missing values per column'] = feature_set['missing_values'].value_counts().to_dict()

    # Detect outliers
    outliers = feature_set[feature_set['feature_type'] == 'numerical'].select_dtypes(include='number').apply(lambda x: x[x > x.quantile(0.95) | x < x.quantile(0.05)], axis=0)
    validation_results['Count of outliers detected'] = outliers.shape[1]
  except Exception as e:
    validation_results['Error'] = str(e)

  return validation_results

def main():
  # Assuming C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv exists in the current working directory
  try:
    feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    validation_outcomes = validate_feature_set(feature_set)
    pd.DataFrame(validation_outcomes, index=[0]).to_csv('feature_set_validation_outcomes_C.csv', index=False)
    print(validation_outcomes)
  except Exception as e:
    print(f'Error occurred: {e}')

if __name__ == '__main__':
  main()
