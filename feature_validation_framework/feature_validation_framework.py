import os
import pandas as pd
import logging
import google.generativeai as genai
from datetime import datetime
import mlflow
from scipy.stats import ks_2samp
import numpy as np
import json
import csv
import sqlite3
from datetime import datetime
import os
import sqlite3
import logging
from datetime import datetime


logging.basicConfig(filename='knowledge_base.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ValidationRuleGenerator:
    def __init__(self, generative_model, kb_interface):
        self.generative_model = generative_model
        self.kb_interface = kb_interface

    def generate_validation_rules(self, metadata, stats, set_name):
        existing_rules = self.kb_interface.load_rules_from_kb(set_name)
        if existing_rules:
            print(f"Using existing validation rules for '{set_name}'.")
            return existing_rules
        
        prompt = self.generate_prompt_from_metadata(metadata, stats)
        rules = self.generative_model.generate(prompt)
        self.kb_interface.save_rules_to_kb(set_name, rules)
        return rules

    def generate_prompt_from_metadata(self, metadata, stats):
        """
        Generates a validation prompt based on the metadata and statistics of the dataset.
        """
        prompt = "Generate a set of comprehensive data validation rules based on the following dataset statistics:\n"
        for column in stats.index:
            dtype = metadata['data_types'][column]
            if pd.api.types.is_numeric_dtype(dtype):
                prompt += f"- '{column}' should have values between {int(stats.loc[column, 'min'])} and {int(stats.loc[column, 'max'])}.\n"
                if stats.loc[column, 'min'] < 0:
                    prompt += f"- '{column}' should not have negative values.\n"
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                prompt += f"- '{column}' should only contain valid categories based on historical data.\n"
                if metadata['unique_counts'][column] == metadata['num_rows']:
                    prompt += f"- '{column}' should be unique.\n"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                prompt += f"- '{column}' should have valid and sequential dates.\n"
            if pd.api.types.is_string_dtype(dtype):
                prompt += f"- '{column}' should be checked for completeness, readability, and keyword presence.\n"
        prompt += "Ensure that the rules cover completeness, consistency, uniqueness, range checks, and any potential outliers."
        return prompt

    def generate_validation_rules(self, metadata, stats, set_name):
        """
        Generates the validation rules using the selected AI model based on the metadata and statistics.
        Retrieves previously generated rules if they exist, only updating if necessary.
        """
        existing_rules = self.kb_interface.load_rules_from_kb(set_name)
        if existing_rules:
            print(f"Using existing validation rules for '{set_name}'.")
            return existing_rules
        
        prompt = self.generate_prompt_from_metadata(metadata, stats)
        print(f"Generated prompt for feature validation:\n{prompt}")
        rules = self.generate_response(text=prompt)
        
        # Save the newly generated rules in the knowledge base
        self.kb_interface.save_rules_to_kb(set_name, rules)
        return rules

    def generate_response(self, text: str):
        """
        Sends the prompt to the selected AI model and retrieves the response (rules).
        """
        if self.model_choice == "gemini":
            print("Fetching response from Gemini AI, please wait...")
            response = self.model.generate_content(text)
            if response and response.text:
                return response.text
            else:
                logging.warning("Gemini response was empty or blocked. Check safety ratings.")
                return None
        elif self.model_choice in ["llama", "mistral"]:
            print("Fetching response from Llama/Mistral AI, please wait...")
            response = self.model.predict(text)  # Llama/Mistral model's API call
            if response:
                return response
            else:
                logging.warning("Llama/Mistral response was empty or blocked.")
                return None

    def generate_code_snippets_with_genai(self, rules,set_name):
        # Generate a prompt that asks the AI to create Python code based on the rules
        #prompt = f"Generate Python code to validate the following data based on these rules:\n{rules}\n"
        prompt = f"""
        Based on the following data validation rules, generate Python code to implement these rules for a pandas DataFrame named 'feature_set':

        {rules}

        The code should perform the necessary checks and save the validation outcomes, including all relevant metrics, to a CSV file named 'validation_results_{set_name}.csv'.
        """
        code_snippets = self.model.generate_content(prompt).text
        return code_snippets

class FeatureSetValidator:
    """
    Validates feature sets by generating and applying validation rules, 
    and stores validation results for easy integration with a knowledge base.
    """

    '''
    def __init__(self, kb_interface):
        self.rule_generator = ValidationRuleGenerator(api_key=os.getenv('GOOGLE_API_KEY'), kb_interface=kb_interface)
        self.kb_interface = kb_interface

    '''
    def __init__(self, rule_generator, code_generator):
        self.rule_generator = rule_generator
        self.code_generator = code_generator

    def validate_feature_set(self, metadata, stats, set_name, feature_set):
        # Step 1: Generate or retrieve validation rules.
        rules = self.rule_generator.generate_validation_rules(metadata, stats, set_name)
        print(f"Generated Validation Rules for {set_name}:\n{rules}")
        self.save_generated_rules(rules, set_name)
        # Step 2: Generate Python code snippets based on the rules.
        code_snippets = self.code_generator.generate_code_snippets(rules, set_name)

        # Step 3: Execute the generated code snippets within the context of the provided dataset.
        local_scope = {'feature_set': feature_set, 'validation_outcomes': {}}  # Initialize validation_outcomes
        try:
            exec(code_snippets, globals(), local_scope)  # Execute the generated validation code
        except Exception as e:
            print(f"Error executing code snippets: {e}")
            return

        validation_outcomes = local_scope.get('validation_outcomes', 'No validation outcomes defined.')
        print(f"Validation outcomes for '{set_name}': {validation_outcomes}")

        # Step 4: Save validation results and metrics.
        self.save_validation_results(validation_outcomes, set_name)
        metrics = {
            'total_features': len(feature_set.columns),
            'missing_values': feature_set.isnull().sum().sum(),
            'validation_outcomes': validation_outcomes
        }
        self.save_metrics(metrics, set_name)

    def save_validation_results(self, validation_outcomes, set_name):
        # Save the validation outcomes to a CSV file
        results_filename = os.path.join('results', f'validation_results_{set_name}.csv')
        with open(results_filename, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['Feature', 'Validation Result'])  # Add headers
            for feature, result in validation_outcomes.items():
                writer.writerow([feature, result])

        print(f"Validation results saved to {results_filename}")

    def save_metrics(self, metrics, set_name):
        # Save the metrics to a CSV file
        metrics_filename = os.path.join('results', f'validated_metrics_{set_name}.csv')
        with open(metrics_filename, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])  
            for key, value in metrics.items():
                writer.writerow([key, value])

        print(f"Metrics saved to {metrics_filename}")

    def validate_feature_set_old(self, set_name, feature_set, baseline_df=None, save=0):
        """
        Validates a feature set, applies the validation rules, generates Python code, 
        executes the code, and stores the results. Checks with the knowledge base 
        before generating new validation rules.
        """
        print(f"Validating feature set '{set_name}'...")

        # Step 1: Check if validation rules already exist in the knowledge base
        if self.kb_interface.check_existing_rules(set_name):
            print(f"Skipping validation rule generation for '{set_name}' as they already exist in the knowledge base.")
            return

        # Step 2: Define Metadata
        metadata = self.define_metadata(feature_set)
        stats = feature_set.describe(include='all').T

        # Step 3: Generate Validation Rules
        rules = self.rule_generator.generate_validation_rules(metadata=metadata, stats=stats, set_name=set_name)
        print(f"Generated Validation Rules for {set_name}:\n{rules}")

        # Save the validation rules using centralized logic
        self.kb_interface.save_rules_to_kb(set_name, rules)

        # Step 4: Custom Validation Issues
        issues = self.dynamic_custom_rules(feature_set)

        if issues:
            print(f"Data Quality Issues Found in {set_name}:")
            for issue in issues:
                print(f"- {issue}")
            self.write_feature_quality_to_table(set_name, issues)

            # Generate Python code snippets based on the generated rules
            code_snippets = self.rule_generator.generate_code_snippets_with_genai(rules,"feature_set.csv")
            print(f"Generated Python code for validation:\n{code_snippets}")

            # Save and execute the generated code snippets
            validation_outcomes = self.save_and_execute_code_snippets(code_snippets, set_name, feature_set)

            # Merge validation outcomes into metrics
            metrics = {
                'total_features': len(feature_set.columns),
                'missing_values': feature_set.isnull().sum().sum(),
                'validation_outcomes': validation_outcomes
            }
        else:
            print(f"No data quality issues found in {set_name}.")
            self.write_feature_quality_to_table(set_name, ["No issues found"])

            metrics = {
                'total_features': len(feature_set.columns),
                'missing_values': feature_set.isnull().sum().sum(),
                'validation_outcomes': "No issues found"
            }

        # Step 5: Save Validation Metrics and Results
        self.save_metrics(metrics, set_name=set_name)

        # Save the generated rules
        self.save_generated_rules(rules, set_name=set_name)

        print(f"Validation for '{set_name}' completed.")

    def save_and_execute_code_snippets(self, code_snippets, set_name, feature_set):

        # Strip out the markdown code block markers
        if code_snippets.startswith("```python"):
            code_snippets = code_snippets[len("```python"):].strip()
        if code_snippets.endswith("```"):
            code_snippets = code_snippets[:-len("```")].strip()

         # Define the dataset file name
        orig_dataset_filename = f'{set_name}.csv'
        print("orig_dataset_filename",orig_dataset_filename)
        # Save the dataset to a CSV file
        dataset_filename = os.path.join("results", orig_dataset_filename)
        feature_set.to_csv(dataset_filename, index=False)
        print(f"Dataset for '{set_name}' saved as {dataset_filename}")
        
        # Replace the dataset reference in code snippets (if necessary)
        code_snippets = code_snippets.replace('feature_set.csv', f'{orig_dataset_filename}')
        # Save the code snippets to a file
        code_snippet_filename = os.path.join('results', f'code_snippets_{set_name}.py')
        print("code_snippet_filename")
        print(code_snippet_filename)
        with open(code_snippet_filename, 'w', encoding="utf-8") as file:
            file.write(code_snippets)

        print(f"Generated code snippets saved to {code_snippet_filename}")

        # Execute the saved code snippets
        local_vars = {}
        try:
            exec(code_snippets, {}, local_vars)  # Execute the code snippets and capture local variables
        except Exception as e:
            print(f"Error executing code snippets: {e}")
            return {}

        validation_outcomes = local_vars.get('validation_outcomes', {})
        print(f"Validation outcomes for '{set_name}': {validation_outcomes}")

        return validation_outcomes
        
    def save_generated_rules(self, rules, set_name):
        # Define the JSON file path
        rules_filename = os.path.join('results', f'generated_rules_{set_name}.json')

        # Write rules to a JSON file
        with open(rules_filename, 'w', encoding="utf-8") as file:
            json.dump(rules, file, indent=4)  # Pretty print the rules

        print(f"Rules saved to {rules_filename}")

    def define_metadata(self, df):
        """
        Extracts metadata from the dataframe, including column names, unique counts, and data types.
        """
        metadata = {
            "columns": df.columns.tolist(),
            "num_rows": df.shape[0],
            "unique_counts": df.nunique(),
            "data_types": df.dtypes.to_dict()
        }
        return metadata

    def dynamic_custom_rules(self, df):
        """
        Applies dynamic custom rules to validate the dataset.
        These rules include:
        - Outlier detection using z-scores for numeric columns
        - Validation of relationships between numeric columns
        - Detecting anomalies in categorical or text data
        - Logical checks for date columns
        - General statistical validations for completeness and consistency
        """
        custom_issues = []

        # 1. Check for outliers using z-scores for numeric columns
        for column in df.select_dtypes(include=['number']).columns:
            mean_value = df[column].mean()
            std_dev = df[column].std()
            if std_dev > 0:
                z_scores = (df[column] - mean_value) / std_dev
                outliers = df[abs(z_scores) > 3]
                if not outliers.empty:
                    custom_issues.append(f"Column '{column}' contains outliers based on z-score (> 3).")

        # 2. Correlation check between numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            for col1 in numeric_columns:
                for col2 in numeric_columns:
                    if col1 != col2 and correlation_matrix.loc[col1, col2] < 0.1:
                        custom_issues.append(f"Columns '{col1}' and '{col2}' have a low correlation, which may indicate data quality issues.")
        
        # 3. Completeness and uniqueness checks for categorical columns
        for column in df.select_dtypes(include=['category', 'object']).columns:
            if df[column].isnull().sum() > 0:
                custom_issues.append(f"Column '{column}' has missing values.")
            if df[column].nunique() == df.shape[0]:
                custom_issues.append(f"Column '{column}' should not be unique for every row, which may indicate a data entry issue.")

        # 4. Text column validation: Ensure text is not empty or unusually short
        for column in df.select_dtypes(include=['object']).columns:
            for idx, value in df[column].items():
                if isinstance(value, str):  # Check if the value is a string
                    if len(value.strip()) == 0:
                        custom_issues.append(f"Row {idx} in column '{column}' is empty.")
                    elif len(value) < 5:  # Threshold can be adjusted
                        custom_issues.append(f"Row {idx} in column '{column}' contains unusually short text (less than 5 characters).")
                elif pd.isna(value):
                    custom_issues.append(f"Row {idx} in column '{column}' contains NaN or missing text.")

        # 5. Logical checks for date columns: Ensure dates are valid and follow logical order
        for column in df.select_dtypes(include=['datetime']).columns:
            if df[column].max() > pd.Timestamp.now():
                custom_issues.append(f"Column '{column}' contains future dates, which may indicate data entry errors.")
            if df[column].isnull().sum() > 0:
                custom_issues.append(f"Column '{column}' has missing date values.")
        
        # 6. General completeness check for all columns
        for column in df.columns:
            missing_values = df[column].isnull().sum()
            if missing_values > 0:
                custom_issues.append(f"Column '{column}' contains {missing_values} missing values.")
        
        return custom_issues

    def write_feature_quality_to_table(self, feature_name, issues):
        """
        Writes feature quality issues to a table or file for easy reference.
        Additionally, logs the number of issues and other relevant metrics.
        Each file is appended with the current date to differentiate multiple runs.
        """
        # Get current date in the format 'YYYY-MM-DD'
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure the results directory exists
        results_dir = 'metrics'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Prepare file paths with the current date
        quality_file = os.path.join(results_dir, f'feature_quality_table_{feature_name}_{current_date}.csv')
        metrics_file = os.path.join(results_dir, f'feature_quality_metrics_{feature_name}_{current_date}.csv')
        
        # Collect metrics
        num_issues = len(issues)
        affected_columns = len(set([issue.split()[0] for issue in issues]))  # Unique feature names affected
        
        # Write detailed issues to CSV file
        with open(quality_file, 'a') as f:
            if os.path.getsize(quality_file) == 0:  # If the file is empty, write header
                f.write('Feature,Issue,Severity\n')
            for issue in issues:
                # Assign a basic severity level (this could be extended based on logic)
                severity = "High" if "missing" in issue.lower() or "outliers" in issue.lower() else "Medium"
                f.write(f"{feature_name},{issue},{severity}\n")
        
        # Write metrics summary to a separate CSV
        with open(metrics_file, 'a') as mf:
            if os.path.getsize(metrics_file) == 0:  # If the file is empty, write header
                mf.write('Feature,Num Issues,Affected Columns,Date\n')
            mf.write(f"{feature_name},{num_issues},{affected_columns},{current_date}\n")
        
        print(f"Feature quality issues for '{feature_name}' saved to {quality_file}.")
        print(f"Feature quality metrics for '{feature_name}' saved to {metrics_file}.")

class KnowledgeBaseInterface:
    """
    Interface for storing validation results and rules in a SQLite database,
    with logging for each step of the operation.
    """
    
    def __init__(self, db_path='knowledge_base.db'):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the database and creates tables if they don't exist.
        Logs the initialization step.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_name TEXT,
                    rules TEXT,
                    created_at TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_name TEXT,
                    results TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()
        
        logging.info("Initialized database and ensured tables exist.")
        print("Database initialized and tables ready.")

    def store(self, rules, set_name):
        """
        Stores generated validation rules to the database.
        Logs the storage operation.
        """
        current_date = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO validation_rules (set_name, rules, created_at)
                VALUES (?, ?, ?)
            ''', (set_name, rules, current_date))
            conn.commit()
        
        logging.info(f"Stored validation rules for '{set_name}' in the database.")
        print(f"Validation rules for '{set_name}' stored in the database.")

    def retrieve(self, set_name):
        """
        Retrieves the validation rules for the given set name.
        Logs the retrieval operation.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT rules FROM validation_rules
                WHERE set_name = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (set_name,))
            row = cursor.fetchone()
        
        if row:
            logging.info(f"Retrieved validation rules for '{set_name}' from the database.")
            print(f"Retrieved validation rules for '{set_name}'.")
            return row[0]
        else:
            logging.warning(f"No validation rules found for '{set_name}'.")
            print(f"No validation rules found for '{set_name}'.")
            return None

    def check_existing_rules(self, set_name):
        """
        Check if validation rules for the given set name already exist in the database.
        Logs the existence check.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM validation_rules
                WHERE set_name = ?
            ''', (set_name,))
            count = cursor.fetchone()[0]
        
        if count > 0:
            logging.info(f"Validation rules for '{set_name}' already exist in the database.")
            print(f"Validation rules for '{set_name}' already exist.")
            return True
        else:
            logging.info(f"No existing validation rules found for '{set_name}'.")
            print(f"No existing validation rules found for '{set_name}'.")
            return False

    def save_validation_results(self, set_name, results, save=0):
        """
        Saves validation results to the database.
        Logs the result-saving operation.
        """
        if save == 1:
            current_date = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO validation_results (set_name, results, created_at)
                    VALUES (?, ?, ?)
                ''', (set_name, results, current_date))
                conn.commit()
            
            logging.info(f"Validation results for '{set_name}' saved in the database.")
            print(f"Validation results for '{set_name}' saved in the database.")
        else:
            logging.info(f"Skipping saving validation results for '{set_name}' as 'save' is not set to 1.")
            print(f"Skipping saving validation results for '{set_name}'.")

    def load_rules_from_kb(self, set_name):
        """
        Loads previously saved rules from the database.
        Logs the loading operation.
        """
        rules = self.retrieve(set_name)
        if rules:
            logging.info(f"Loaded validation rules for '{set_name}' from the database.")
            print(f"Loaded validation rules for '{set_name}'.")
        return rules


# RAG Validation
class RAGValidation:
    def __init__(self, knowledge_base, generator):
        self.knowledge_base = knowledge_base
        self.generator = generator

    def validate(self, feature_set, set_name):
        existing_rules = self.knowledge_base.retrieve(set_name)
        if existing_rules:
            print("Using retrieved rules for validation.")
            return existing_rules
        else:
            print("Generating new rules.")
            prompt = self.generate_prompt_from_metadata(feature_set, set_name)
            generated_rules = self.generator.generate(prompt) 
            print("---------------generated_rules----------------")
            print(generated_rules) 
            self.knowledge_base.store(generated_rules, set_name)
            return generated_rules

    def generate_prompt_from_metadata(self, feature_set, set_name):
        # Generate the prompt based on the feature set and set name
        prompt = f"Generate a set of comprehensive validation rules for the feature set '{set_name}'.\n"
        prompt += "Here are the dataset statistics:\n"
        for column in feature_set.columns:
            prompt += f"- Column '{column}' with dtype '{feature_set[column].dtype}' and sample values: {feature_set[column].unique()[:5]}\n"
        return prompt

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
            updated_rules = self.rag_validator.validate(current_data, set_name)
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


class CodeSnippetGenerator:
    def __init__(self, model):
        self.model = model  # Store the generative model
        self.template_store = {}  # Store generated code snippets for reuse

    def generate_or_retrieve(self, validation_rules, feature_set_name):
        snippets = []
        for rule in validation_rules:
            if rule in self.template_store:
                snippets.append(self.template_store[rule])
            else:
                # Generate a prompt asking the AI to create Python code based on the rules
                prompt = f"""
                Based on the following data validation rules, generate Python code to implement these rules for a pandas DataFrame named 'feature_set':

                {rule}

                The code should perform the necessary checks and save the validation outcomes, including all relevant metrics, to a CSV file named 'validation_results_{feature_set_name}.csv'.
                """
                new_snippet = self.model.generate(prompt)
                print("new_snippet---------------------")
                print(new_snippet)
                snippets.append(new_snippet)
                self.template_store[rule] = new_snippet  # Store for future use
        return snippets
    

class GenerativeModel:
    """
    A class to interface with different AI generative models based on the provided API key and model choice.
    """
    def __init__(self, api_key, model_choice):
        self.api_key = api_key
        self.model_choice = model_choice
        self.model = None
        self.configure_model()

    def configure_model(self):
        """
        Configures the specific AI model based on the choice (Gemini, Llama, or Mistral).
        """
        if self.model_choice == "gemini":
            # Setup for Gemini model
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        elif self.model_choice in ["llama", "mistral"]:
            # Setup for Mistral/Llama model using MLflow
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version('system.ai.mistral_7b_v0_1', version=2)
            self.model = mlflow.pyfunc.load_model(f"models:/system.ai.mistral_7b_v0_1/{model_version.version}")

    def generate(self, input_text):
        """
        Generates content based on the input text using the configured model.
        """
        if self.model_choice == "gemini":
            response = self.model.generate_content(input_text)
            if response:
                # Check if the response is complex (multiple parts or candidates)
                if hasattr(response, 'candidates') and response.candidates:
                    # Access the first candidate's content parts
                    return ' '.join([part.text for part in response.candidates[0].content.parts])
                elif hasattr(response, 'parts') and response.parts:
                    # Access parts directly if available
                    return ' '.join([part.text for part in response.parts])
                else:
                    # Fallback to response.text for simple cases
                    return response.text
            else:
                logging.warning("Gemini response was empty or blocked. Check safety ratings.")
                return None
            
        elif self.model_choice in ["llama", "mistral"]:
            return self.model.predict(input_text)


def load_new_feature_set():
    file_path = os.path.join('..', 'data', 'insurance_claims_report.csv')
    df = pd.read_csv(file_path)
    print(f"Read {len(df)} records")
    print("Columns after loading:", df.columns.tolist())
    
    # Extract dataset name from file path
    dataset_name = os.path.basename(file_path).split('.')[0]
    
    # Randomly split the dataset into three parts for baseline and testing
    baseline_df, set_2, set_3 = np.split(df.sample(frac=1, random_state=42), [int(.33*len(df)), int(.66*len(df))])
    return baseline_df, set_2, set_3, dataset_name

''''

if __name__ == "__main__":
    # Load sample data and split it into three sets
    baseline_data, data_set_2, data_set_3, dataset_name = load_new_feature_set()

    # Choose the AI model
    model_choice = "gemini"  # Can be "gemini" or "mistral/llama"

    # Instantiate the knowledge base and validator
    knowledge_base = KnowledgeBaseInterface(kb_directory="knowledge_base")
    validator = FeatureSetValidator(kb_interface=knowledge_base)

    # Configure the rule generator with the chosen AI model
    validator.rule_generator = ValidationRuleGenerator(
        api_key=os.getenv('GOOGLE_API_KEY'),
        kb_interface=knowledge_base,
        model_choice=model_choice
    )

    # Initialize drift detector with the baseline dataset
    drift_detector = DriftDetector(historical_data=baseline_data)

    # Validate and check for drift in subsequent datasets
    for set_number, dataset in enumerate([data_set_2, data_set_3], start=2):
        print(f"Validating dataset part {set_number}...")
        # Check for drift
        drifted_columns = drift_detector.check_drift(dataset)
        if drifted_columns:
            print(f"Drift detected in dataset part {set_number} in columns: {drifted_columns}")
            # If drift is detected, possibly re-generate validation rules
            validator.validate_feature_set(f"{dataset_name}_part_{set_number}", dataset, baseline_data=baseline_data, save=1)
        else:
            # No drift detected, proceed with normal validation
            validator.validate_feature_set(f"{dataset_name}_part_{set_number}", dataset, baseline_data=None, save=1)

    # Optionally save the rules and results
    validation_rules = "Generated validation rules based on latest data."
    validation_results = "Validation results after checking for data drift."
    knowledge_base.save_rules_to_kb(dataset_name, validation_rules)
    knowledge_base.save_validation_results(dataset_name, validation_results, save=1)
'''

def generate_metadata_and_stats(df):
    metadata = {
        'data_types': df.dtypes.apply(lambda dtype: dtype.name).to_dict(),
        'unique_counts': df.nunique().to_dict()
    }
    stats = {
        column: {'min': df[column].min(), 'max': df[column].max()}
        for column in df.columns if pd.api.types.is_numeric_dtype(df[column])
    }
    return metadata, stats

if __name__ == "__main__":
    # Load the feature sets and baseline data
    baseline_data, data_set_2, data_set_3, dataset_name = load_new_feature_set()

    # Initialize the Generative Model
    api_key = os.getenv('GOOGLE_API_KEY')
    model_choice = "gemini"  # Choose the model (gemini or mistral)
    generative_model = GenerativeModel(api_key, model_choice)

    # Initialize the RAG validation, CodeSnippetGenerator, and Knowledge Base
    knowledge_base = KnowledgeBaseInterface(db_path='knowledge_base.db')
    rag_validator = RAGValidation(knowledge_base, generative_model)

    # Initialize the DriftManager, which handles both drift detection and rule updating
    drift_manager = DriftManager(baseline_data, rag_validator)

    # Validate the baseline dataset and generate rules
    baseline_rules = rag_validator.validate(baseline_data, dataset_name)

    # Store the generated rules using the 'store' method
    knowledge_base.store(baseline_rules, dataset_name)

    # Initialize CodeSnippetGenerator with the GenerativeModel
    code_generator = CodeSnippetGenerator(model=generative_model)

    # Validate each additional dataset, check for drift, and update rules if necessary
    for set_number, dataset in enumerate([data_set_2, data_set_3], start=2):
        print(f"Validating dataset part {set_number}...")

        # Check for drift and update rules if necessary
        updated_rules, drifted_columns = drift_manager.check_drift_and_update_rules(dataset, f"{dataset_name}_part_{set_number}")

        # If drift was detected, generate new code snippets based on the updated rules
        if updated_rules:
            code_snippets = code_generator.generate_or_retrieve(updated_rules, f"{dataset_name}_part_{set_number}")
            print(f"Generated code snippets for dataset part {set_number}:\n{code_snippets}")
