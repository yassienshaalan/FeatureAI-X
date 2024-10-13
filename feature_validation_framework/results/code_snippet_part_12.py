# Code snippet part 12
import pandas as pd
import os

def validate_feature_set(feature_set_path):
    validation_results = []

    # Check if the file exists
    if not os.path.isfile(feature_set_path):
        validation_results.append("Error: File not found.")
        return validation_results

    # Read the feature set into a DataFrame
    try:
        feature_set = pd.read_csv(feature_set_path)
    except Exception as e:
        validation_results.append(f"Error reading file: {e}")
        return validation_results

    # Check for missing columns
    missing_columns = [column for column in ['feature_name', 'data_type', 'missing_values'] if column not in feature_set.columns]
    if len(missing_columns) > 0:
        validation_results.append("Error: Missing columns: {}".format(", ".join(missing_columns)))
        return validation_results

    # Calculate validation metrics
    feature_names = feature_set['feature_name'].tolist()
    data_types = feature_set['data_type'].tolist()
    missing_values = feature_set['missing_values'].tolist()

    validation_metrics = {
        "Total Features": len(feature_names),
        "Missing Values Per Column": {feature_name: missing_value for feature_name, missing_value in zip(feature_names, missing_values)},
        "Outliers Detected": 0  # Placeholder, needs actual implementation
    }

    # Save validation results to CSV file
    try:
        pd.DataFrame(validation_metrics).to_csv("feature_set_validation_outcomes_.csv")
    except Exception as e:
        validation_results.append(f"Error saving validation results: {e}")

    return validation_results

def main():
    validation_results = validate_feature_set("C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv")
    for result in validation_results:
        print(result)

if __name__ == "__main__":
    main()
