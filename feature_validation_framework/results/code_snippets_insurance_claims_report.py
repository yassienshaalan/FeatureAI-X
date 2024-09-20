import pandas as pd

# Load the feature set
feature_set = pd.read_csv('insurance_claims_report.csv')

# Completeness Rules
# Check for null values in required fields
required_fields = ['policy_number', 'policy_bind_date', 'policy_state', 'policy_csl', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_zip', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship', 'capital-gains', 'capital-loss', 'incident_date', 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city', 'incident_location', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses', 'police_report_available', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'auto_make', 'auto_model', 'auto_year', 'fraud_reported']
for field in required_fields:
    if feature_set[field].isnull().any():
        print(f"Missing values found in {field} column")

# Verify that categorical variables are not missing
categorical_vars = ['policy_state', 'policy_csl', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship', 'property_damage', 'bodily_injuries', 'authorities_contacted', 'incident_state', 'incident_city', 'incident_location', 'collision_type', 'incident_severity', 'police_report_available', 'auto_make', 'auto_model', 'fraud_reported']
for var in categorical_vars:
    if feature_set[var].isnull().any():
        print(f"Missing values found in {var} column")

# Check that date fields are not empty
date_vars = ['policy_bind_date', 'incident_date']
for var in date_vars:
    if feature_set[var].isnull().any():
        print(f"Missing values found in {var} column")

# Consistency Rules
# Validate ranges for numeric variables
numeric_vars = ['policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year']
for var in numeric_vars:
    if feature_set[var].max() > feature_set[var].max():
        print(f"Maximum value of {var} exceeds expected limit")
    if feature_set[var].min() < feature_set[var].min():
        print(f"Minimum value of {var} falls below expected limit")

# Ensure consistency in format and values of categorical variables
for var in categorical_vars:
    # Check for expected values
    if not set(feature_set[var].unique()).issubset(set(expected_values[var])):
        print(f"Unexpected values found in {var} column")


# Uniqueness Rules
# Check for duplicate values in incident_location
if not duplicate_list.empty:
    print("Duplicate values found in incident_location column")

# Range Checks
for var in numeric_vars:
    if feature_set[var].max() > feature_set[var].max():
        print(f"Maximum value of {var} exceeds expected limit")
    if feature_set[var].min() < feature_set[var].min():
        print(f"Minimum value of {var} falls below expected limit")

# Outlier Checks
# Identify extreme values (outliers) for numeric variables
for var in numeric_vars:
    # Calculate interquartile range (IQR)
    iqr = feature_set[var].quantile(0.75) - feature_set[var].quantile(0.25)
    # Calculate lower and upper bounds for outliers
    lower_bound = feature_set[var].quantile(0.25) - (1.5 * iqr)
    upper_bound = feature_set[var].quantile(0.75) + (1.5 * iqr)
    # Identify outliers
    outliers = feature_set[(feature_set[var] < lower_bound) | (feature_set[var] > upper_bound)]
    if not outliers.empty:
        print(f"Outliers detected in {var} column")

# Specific Field Rules
# Validate policy_number
if feature_set['policy_number'].max() > 999435 or feature_set['policy_number'].min() < 100804:
    print("Policy number range is invalid")

# Check policy_bind_date
if not set(feature_set['policy_bind_date'].unique()).issubset(set(expected_values['policy_bind_date'])):
    print("Unexpected values found in policy_bind_date column")

# Validate policy_state
if not set(feature_set['policy_state'].unique()).issubset(set(expected_values['policy_state'])):
    print("Unexpected values found in policy_state column")

# Ensure consistency in policy_csl
if not set(feature_set['policy_csl'].unique()).issubset(set(expected_values['policy_csl'])):
    print("Unexpected values found in policy_csl column")

# Verify policy_deductable range
if feature_set['policy_deductable'].max() > 2000 or feature_set['policy_deductable'].min() < 500:
    print("Policy deductible range is invalid")

# Validate policy_annual_premium range
if feature_set['policy_annual_premium'].max() > 2047 or feature_set['policy_annual_premium'].min() < 433:
    print("Policy annual premium range is invalid")

# Check umbrella_limit range
if feature_set['umbrella_limit'].max() > 10000000 or feature_set['umbrella_limit'].min() < -1000000 or any(feature_set['umbrella_limit'] < 0):
    print("Umbrella limit range is invalid")

# Validate insured_zip range
if feature_set['insured_zip'].max() > 620962 or feature_set['insured_zip'].min() < 430104:
    print("Insured zip code range is invalid")

# Ensure consistency in insured_sex
if not set(feature_set['insured_sex'].unique()).issubset(set(expected_values['insured_sex'])):
    print("Unexpected values found in insured_sex column")

# Verify insured_education_level range
if not set(feature_set['insured_education_level'].unique()).issubset(set(expected_values['insured_education_level'])):
    print("Unexpected values found in insured_education_level column")

# Ensure consistency in insured_occupation
if not set(feature_set['insured_occupation'].unique()).issubset(set(expected_values['insured_occupation'])):
    print("Unexpected values found in insured_occupation column")

# Verify insured_hobbies range
if not set(feature_set['insured_hobbies'].unique()).issubset(set(expected_values['insured_hobbies'])):
    print("Unexpected values found in insured_hobbies column")

# Ensure consistency in insured_relationship
if not set(feature_set['insured_relationship'].unique()).issubset(set(expected_values['insured_relationship'])):
    print("Unexpected values found in insured_relationship column")

# Verify capital-gains range
if feature_set['capital-gains'].max() > 100500 or feature_set['capital-gains'].min() < 0:
    print("Capital gains range is invalid")

# Check capital-loss range
if feature_set['capital-loss'].max() > 0 or feature_set['capital-loss'].min() < -111100 or any(feature_set['capital-loss'] > 0):
    print("Capital loss range is invalid")

# Validate incident_date
if not set(feature_set['incident_date'].unique()).issubset(set(expected_values['incident_date'])):
    print("Unexpected values found in incident_date column")

# Ensure consistency in incident_type