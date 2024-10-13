# Code snippet part 16
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
  """
  Validates a pandas DataFrame 'feature_set' based on given rules.

  Args:
    feature_set: The pandas DataFrame to be validated.

  Returns:
    A dictionary containing the validation outcomes.
  """
  
  # Create an empty dictionary to store validation outcomes
  validation_outcomes = {}

  # Check if all necessary columns are present in the DataFrame
  missing_columns = []
  for expected_column in ['feature_name', 'feature_type', 'missing_values_count']:
    if expected_column not in feature_set.columns:
      missing_columns.append(expected_column)
  
  if missing_columns:
    validation_outcomes['missing_columns'] = missing_columns
    print(f"Missing columns: {missing_columns}. Please ensure all necessary columns are present in the DataFrame.")
  
  try:
    # Count of total features
    validation_outcomes['total_features'] = feature_set.shape[1]
    
    # Count of missing values per column
    validation_outcomes['missing_values_count'] = feature_set['missing_values_count'].to_dict()
    
    # Count of outliers detected
    # Placeholder. Add code to detect outliers based on specific rules.
    validation_outcomes['outliers_detected'] = 0
    
  except Exception as e:
    print(f"Error occurred during validation: {e}")

  return validation_outcomes

def main():
  """
  Calls the 'validate_feature_set' function for a specific file and prints the outcomes.
  """
  try:
    feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
  except FileNotFoundError:
    print("File 'C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv' not found.")
  except Exception as e:
    print(f"Error occurred while reading the file: {e}")
    return
  
  validation_outcomes = validate_feature_set(feature_set)

  # Save the validation outcomes to a CSV file
  try:
    pd.DataFrame(validation_outcomes, index=[0]).to_csv('feature_set_validation_outcomes_U.csv', index=False)
  except Exception as e:
    print(f"Error occurred while saving the validation outcomes to CSV: {e}")

  # Print the validation outcomes
  for key, value in validation_outcomes.items():
    print(f"{key}: {value}")

if __name__ == "__main__":
  main()
