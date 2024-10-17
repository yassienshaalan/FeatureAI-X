from scipy.stats import ks_2samp
import logging
import numpy as np 

class DriftManager:
    """
    This class handles both drift detection and updating validation rules based on drift.
    It uses the DriftDetector to check for drift and RAGValidation to update the rules if drift is detected.
    """
    def __init__(self, baseline_data, rag_validator):
        self.drift_detector = DriftDetector(historical_data=baseline_data)
        self.rag_validator = rag_validator

    def check_drift_and_update_rules(self, current_data, set_name):
        """
        This method checks for drift in the current dataset compared to the baseline.
        If drift is detected, it updates the validation rules using RAGValidation.
        """
        drifted_columns = self.drift_detector.check_drift(current_data)
        if drifted_columns:
            print(f"Drift detected in columns: {drifted_columns}")
            metadata = {
                'data_types': current_data.dtypes.apply(lambda dtype: dtype.name).to_dict(),
                'unique_counts': current_data.nunique().to_dict(),
                'num_rows': current_data.shape[0] 
            }
            updated_rules = self.rag_validator.validate(current_data, set_name,metadata)
            return updated_rules, drifted_columns
        else:
            print("No drift detected.")
            return None, None

class DriftDetector:
    """
    This class detects drift by comparing the new dataset with the baseline dataset (historical data).
    """
    def __init__(self, historical_data):
        self.historical_data = historical_data

    def check_drift(self, new_data, threshold=0.05):
        # Ensure only numeric columns are used for the drift detection
        numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
        p_values = {
            col: ks_2samp(self.historical_data[col], new_data[col])[1]
            for col in new_data.columns
            if col in numeric_cols  # Only check drift for numeric columns
        }
        drifted_columns = [col for col, p in p_values.items() if p < threshold]
        return drifted_columns
