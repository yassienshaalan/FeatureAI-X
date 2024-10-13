# Code snippet part 1
import os
import pandas as pd
import numpy as np

def validate_feature_set(feature_set):
    """
    Validates a pandas DataFrame 'feature_set' based on specified data validation rules.

    Args:
        feature_set (pandas.DataFrame): The DataFrame to be validated.

    Returns:
        dict: A dictionary containing the validation outcomes, including:
            - Count of total features
            - Count of missing values per column
            - Count of outliers detected
            - Any other relevant statistics based on the rules
    """

    # Ensure the necessary columns are present
    expected_columns = ['months_as_customer', 'age', 'policy_number', 'policy_bind_date', 'policy_state', 'policy_csl',
                       'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_zip', 'insured_sex',
                       'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship',
                       'capital-gains', 'capital-loss', 'incident_date', 'incident_type', 'collision_type',
                       'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city',
                       'incident_location', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'property_damage',
                       'bodily_injuries', 'witnesses', 'police_report_available', 'total_claim_amount', 'injury_claim',
                       'property_claim', 'vehicle_claim', 'auto_make', 'auto_model', 'auto_year', 'fraud_reported']
    missing_columns = set(expected_columns) - set(feature_set.columns)
    if missing_columns:
        validation_results['missing_columns'] = list(missing_columns)
        return validation_results

    # Perform completeness checks
    missing_values = feature_set.isnull().sum()
    validation_results['missing_values'] = missing_values.to_dict()

    # Perform consistency checks
    try:
        validation_results['policy_number_unique'] = feature_set['policy_number'].nunique() == len(feature_set)
    except:
        validation_results['policy_number_unique'] = False
        validation_results['error'] = 'Error checking policy_number uniqueness.'

    try:
        validation_results['incident_location_unique'] = feature_set['incident_location'].nunique() == len(feature_set)
    except:
        validation_results['incident_location_unique'] = False
        validation_results['error'] = 'Error checking incident_location uniqueness.'

    try:
        validation_results['auto_make_and_model_consistent'] = (feature_set['auto_make'] == feature_set['auto_model']).all()
    except:
        validation_results['auto_make_and_model_consistent'] = False
        validation_results['error'] = 'Error checking auto_make and auto_model consistency.'

    # Perform uniqueness checks
    try:
        validation_results['policy_number_unique'] = feature_set['policy_number'].nunique() == len(feature_set)
    except:
        validation_results['policy_number_unique'] = False
        validation_results['error'] = 'Error checking policy_number uniqueness.'

    try:
        validation_results['incident_location_unique'] = feature_set['incident_location'].nunique() == len(feature_set)
    except:
        validation_results['incident_location_unique'] = False
        validation_results['error'] = 'Error checking incident_location uniqueness.'

    # Perform range checks
    try:
        validation_results['months_as_customer_range'] = (feature_set['months_as_customer'] >= 1) & (feature_set['months_as_customer'] <= 479)
    except:
        validation_results['months_as_customer_range'] = False
        validation_results['error'] = 'Error checking months_as_customer range.'

    try:
        validation_results['age_range'] = (feature_set['age'] >= 18) & (feature_set['age'] <= 64)
    except:
        validation_results['age_range'] = False
        validation_results['error'] = 'Error checking age range.'

    try:
        validation_results['policy_number_range'] = (feature_set['policy_number'] >= 104594) & (feature_set['policy_number'] <= 999435)
    except:
        validation_results['policy_number_range'] = False
        validation_results['error'] = 'Error checking policy_number range.'

    try:
        validation_results['policy_deductable_range'] = (feature_set['policy_deductable'] >= 500) & (feature_set['policy_deductable'] <= 2000)
    except:
        validation_results['policy_deductable_range'] = False
        validation_results['error'] = 'Error checking policy_deductable range.'

    try:
        validation_results['policy_annual_premium_range'] = (feature_set['policy_annual_premium'] >= 625) & (feature_set['policy_annual_premium'] <= 1935)
    except:
        validation_results['policy_annual_premium_range'] = False
        validation_results['error'] = 'Error checking policy_annual_premium range.'

    try:
        validation_results['umbrella_limit_range'] = (feature_set['umbrella_limit'] >= -1000000) & (feature_set['umbrella_limit'] <= 10000000)
    except:
        validation_results['umbrella_limit_range'] = False
        validation_results['error'] = 'Error checking umbrella_limit range.'

    try:
        validation_results['insured_zip_range'] = (feature_set['insured_zip'] >= 430567) & (feature_set['insured_zip'] <= 620962)
    except:
        validation_results['insured_zip_range'] = False
        validation_results['error'] = 'Error checking insured_zip range.'

    try:
        validation_results['capital-gains_range'] = (feature_set['capital-gains'] >= 0) & (feature_set['capital-gains'] <= 100500)
    except:
        validation_results['capital-gains_range'] = False
        validation_results['error'] = 'Error checking capital-gains range.'

    try:
        validation_results['capital-loss_range'] = (feature_set['capital-loss'] >= -93600) & (feature_set['capital-loss'] <= 0)
    except:
        validation_results['capital-loss_range'] = False
        validation_results['error'] = 'Error checking capital-loss range.'

    try:
        validation_results['incident_hour_of_the_day_range'] = (feature_set['incident_hour_of_the_day'] >= 0) & (feature_set['incident_hour_of_the_day'] <= 23)
    except:
        validation_results['incident_hour_of_the_day_range'] = False
        validation_results['error'] = 'Error checking incident_hour_of_the_day range.'

    try:
        validation_results['number_of_vehicles_involved_range'] = (feature_set['number_of_vehicles_involved'] >= 1) & (feature_set['number_of_vehicles_involved'] <= 4)
    except:
        validation_results['number_of_vehicles_involved_range'] = False
        validation_results['error'] = 'Error checking number_of_vehicles_involved range.'

    try:
        validation_results['bodily_injuries_range'] = (feature_set['bodily_injuries'] >= 0) & (feature_set['bodily_injuries'] <= 2)
    except:
        validation_results['bodily_injuries_range'] = False
        validation_results['error'] = 'Error checking bodily_injuries range.'

    try:
        validation_results['witnesses_range'] = (feature_set['witnesses'] >= 0) & (feature_set['witnesses'] <= 3)
    except:
        validation_results['witnesses_range'] = False
        validation_results['error'] = 'Error checking witnesses range.'

    try:
        validation_results['total_claim_amount_range'] = (feature_set['total_claim_amount'] >= 2250) & (feature_set['total_claim
