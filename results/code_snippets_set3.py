import pandas as pd
import numpy as np

# Read the data
feature_set = pd.read_csv('data_set3.csv')

# Completeness Checks
feature_set = feature_set.dropna(subset=['vehicle_claim', 'property_damage', 'incident_hour_of_the_day', 'comments'])

# Consistency Checks
feature_set['vehicle_claim'] = feature_set['vehicle_claim'].astype(int)  # Convert to integer for comparison
feature_set['property_damage'] = feature_set['property_damage'].astype(int)  # Convert to integer for comparison
feature_set['incident_hour_of_the_day'] = feature_set['incident_hour_of_the_day'].astype(int)  # Convert to integer for comparison
feature_set = feature_set[(feature_set['vehicle_claim'] == feature_set['insurance_claim']) &
                           (feature_set['property_damage'] == feature_set['reported_property_damage']) &
                           (feature_set['incident_hour_of_the_day'] == feature_set['police_report_time']) &
                           (feature_set['comments'] == feature_set['incident_summary'])]

# Uniqueness Checks
feature_set = feature_set.drop_duplicates()

# Range Checks
feature_set = feature_set[(feature_set['vehicle_claim'] >= 70) & (feature_set['vehicle_claim'] <= 79560)]
feature_set = feature_set[(feature_set['property_damage'] >= 0) & (feature_set['property_damage'] <= 2)]
feature_set = feature_set[(feature_set['incident_hour_of_the_day'] >= 0) & (feature_set['incident_hour_of_the_day'] <= 23)]

# Outlier Checks
feature_set['vehicle_claim_outlier'] = np.where((feature_set['vehicle_claim'] < 70) | (feature_set['vehicle_claim'] > 79560), True, False)
feature_set['property_damage_outlier'] = np.where((feature_set['property_damage'] < 0) | (feature_set['property_damage'] > 2), True, False)
feature_set['incident_hour_of_the_day_outlier'] = np.where((feature_set['incident_hour_of_the_day'] < 0) | (feature_set['incident_hour_of_the_day'] > 23), True, False)

# Comments Validation
# Assume categories are defined in a separate dataset
valid_categories = pd.read_csv('valid_categories.csv')
feature_set['comments_valid'] = np.where(feature_set['comments'].isin(valid_categories['category']), True, False)

# Feature Drift Detection
# Store metrics for future comparison

# Save Validation Results
feature_set.to_csv('validation_results_set3.csv', index=False)