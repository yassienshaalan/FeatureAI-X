# Code snippet part 17
import pandas as pd
import os

def validate_feature_set(feature_set: pd.DataFrame) -> dict:
    """
    Validates a pandas DataFrame based on specified rules.

    Args:
        feature_set: The pandas DataFrame to validate.

    Returns:
        A dictionary containing the validation outcomes.
    """

    validation_results = {}

    try:
        # Count of total features
        validation_results["Total Features"] = len(feature_set.columns)

        # Count of missing values per column
        validation_results["Missing Values per Column"] = feature_set.isnull().sum()

        # Count outliers detected
        # Assuming that an outlier is defined as a value that is more than 3 standard deviations away from the mean
        validation_results["Outliers Detected"] = (
            (feature_set - feature_set.mean()) / feature_set.std() > 3
        ).sum()

        # Other relevant statistics
        validation_results["Data Types"] = feature_set.dtypes

    except Exception as e:
        validation_results["Error"] = str(e)

    return validation_results


def main():

    try:
        if os.path.exists("C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv"):
            feature_set = pd.read_csv("C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv")
        else:
            raise FileNotFoundError("File C:/Projects/GenAI_DB_Hack/FeatureAI-X/feature_validation_framework/results/insurance_claims_report_dynamic_rules.csv not found")

        validation_outcomes = validate_feature_set(feature_set)

        pd.DataFrame(validation_outcomes, index=[0]).to_csv(
            "feature_set_validation_outcomes.csv", index=False
        )

    except Exception as e:
        validation_outcomes = {"Error": str(e)}
        pd.DataFrame(validation_outcomes, index=[0]).to_csv(
            "feature_set_validation_outcomes.csv", index=False
        )

if __name__ == "__main__":
    main()
