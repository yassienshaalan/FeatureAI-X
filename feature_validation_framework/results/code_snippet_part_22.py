# Code snippet part 22
import os
import pandas as pd

def validate_feature_set(feature_set, validation_results):
    # Check if required columns exist
    required_cols = ['Feature', 'Values']
    missing_cols = [col for col in required_cols if col not in feature_set.columns]
    if missing_cols:
        validation_results.append(f"Missing required columns: {missing_cols}")

    # Count total features
    total_features = len(feature_set['Feature'])

    # Count missing values per column
    missing_values = feature_set.isnull().sum()
    missing_values_dict = dict(zip(missing_values.index, missing_values.values))

    # Count outliers
    outliers = 0  # Assuming no outliers for this example
    
    # Calculate statistics
    validation_metrics = {
        "Total Features": total_features,
        "Missing Values": missing_values_dict,
        "Outliers": outliers
    }
    
    return validation_metrics

def main():
    # Read feature set
    try:
        feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
    except FileNotFoundError:
        validation_results.append("Feature set file not found")
        return

    # Perform validation
    validation_results = []
    validation_metrics = validate_feature_set(feature_set, validation_results)

    # Save validation results
    try:
        df_results = pd.DataFrame(validation_results)
        df_results.to_csv('feature_set_validation_outcomes_K.csv', index=False)
    except Exception as e:
        validation_results.append(f"Error saving validation results: {str(e)}")

    # Print outcomes
    print("Validation Outcomes:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
