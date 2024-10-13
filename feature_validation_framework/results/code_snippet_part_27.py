# Code snippet part 27
import os
import pandas as pd

def validate_feature_set(feature_set):
    validation_results = []
    
    # Check column existence
    expected_columns = [] # Define the expected columns here
    missing_columns = list(set(expected_columns) - set(feature_set.columns))
    if len(missing_columns) > 0:
        validation_results.append("Missing columns: {}".format(", ".join(missing_columns)))
    
    # Count of total features
    feature_count = feature_set.shape[1]
    
    # Count of missing values per column
    missing_values_per_column = feature_set.isnull().mean() * 100
    
    # Count of outliers (assuming z-score threshold of 3)
    z_scores = abs((feature_set - feature_set.mean()) / feature_set.std())
    outlier_count = (z_scores > 3).sum().sum()
    
    # Store validation outcomes in a dictionary
    validation_metrics = {
        "Total Features": feature_count,
        "Missing Values Per Column": missing_values_per_column.to_dict(),
        "Outlier Count": outlier_count
    }
    
    # Save validation results to a CSV file
    try:
        validation_df = pd.DataFrame(validation_metrics, index=[0])
        validation_df.to_csv("feature_set_validation_outcomes.csv", index=False)
    except Exception as e:
        validation_results.append("Error saving validation results: {}".format(str(e)))
    
    return validation_results


def main():
    try:
        if os.path.exists('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv'):
            feature_set = pd.read_csv('C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv')
            validation_results = validate_feature_set(feature_set)
            if len(validation_results) > 0:
                print("Validation Errors:")
                for error in validation_results:
                    print(error)
            else:
                print("Validation Successful")
        else:
            print("Error: C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv file not found")
    except Exception as e:
        print("Error during execution: {}".format(str(e)))


if __name__ == '__main__':
    main()
