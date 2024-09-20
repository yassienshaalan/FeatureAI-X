import os
import pandas as pd
import logging
import google.generativeai as genai
from datetime import datetime
import mlflow

class ValidationRuleGenerator:
    """
    This class is responsible for generating data validation rules dynamically 
    based on metadata and statistics. It retrieves previously created rules 
    from the knowledge base and only updates or creates new rules if needed.
    """
    
    def __init__(self, api_key=None, kb_interface=None, model_choice="gemini"):
        self.api_key = api_key
        self.kb_interface = kb_interface
        self.model_choice = model_choice
        self.model = None
        
        if self.model_choice == "gemini":
            # Configure Gemini AI
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        elif self.model_choice == "llama" or self.model_choice == "mistral":
            # Configure Mistral/Llama using MLflow
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version('system.ai.mistral_7b_v0_1', version=2)
            self.model = mlflow.pyfunc.load_model("models:/system.ai.mistral_7b_v0_1/2")
        
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

    def __init__(self, kb_interface):
        self.rule_generator = ValidationRuleGenerator(api_key=os.getenv('GOOGLE_API_KEY'), kb_interface=kb_interface)
        self.kb_interface = kb_interface

    
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
        print("code_snippets",code_snippets)
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


    def validate_feature_set(self, set_name, feature_set, baseline_df=None, save=0):
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
    def save_metrics(self, metrics, set_name):
        # Save the metrics to a file
        metrics_filename = os.path.join('results', f'validated_metrics_{set_name}.txt')
        with open(metrics_filename, 'w', encoding="utf-8") as file:
            for key, value in metrics.items():
                file.write(f"{key}: {value}\n")

        print(f"Metrics saved to {metrics_filename}")

    def save_generated_rules(self, rules, set_name):
        # Save the generated rules to a file
        rules_filename = os.path.join('results', f'generated_rules_{set_name}.txt')
        with open(rules_filename, 'w', encoding="utf-8") as file:
            file.write(rules)

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
    Interface for storing validation results and rules in a format that can be shared or integrated into a knowledge base.
    """

    def __init__(self, kb_directory='knowledge_base'):
        self.kb_directory = kb_directory
        if not os.path.exists(self.kb_directory):
            os.makedirs(self.kb_directory)

    def check_existing_rules(self, set_name):
        """Check if validation rules for the given set_name already exist in the knowledge base."""
        rules_dir = os.path.join('results', 'generated_rules')
        current_date = datetime.now().strftime('%Y-%m-%d')
        rules_filename = os.path.join(rules_dir, f'validation_rules_{set_name}_{current_date}.txt')

        # Check if the file exists
        if os.path.exists(rules_filename):
            print(f"Validation rules for '{set_name}' already exist in the knowledge base: {rules_filename}")
            return True
        return False
    
    def save_rules_to_kb(self, set_name, rules):
        """Saves generated validation rules to the knowledge base (a simple file for now)."""
        # Ensure the 'generated_rules' directory exists inside 'results'
        rules_dir = os.path.join('results', 'generated_rules')
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)

        # Prepare file name with current date for uniqueness
        current_date = datetime.now().strftime('%Y-%m-%d')
        rules_filename = os.path.join(rules_dir, f'validation_rules_{set_name}_{current_date}.txt')

        # Save the rules in the generated_rules directory
        with open(rules_filename, 'w') as file:
            file.write(rules)
        print(f"Validation rules for {set_name} saved to {rules_filename}")

    def save_validation_results(self, set_name, results, save=0):
        """Saves validation results to the knowledge base."""
        if save == 1:
            results_dir = os.path.join('results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Prepare file name with current date for uniqueness
            current_date = datetime.now().strftime('%Y-%m-%d')
            result_filename = os.path.join(results_dir, f'validation_results_{set_name}_{current_date}.txt')

            with open(result_filename, 'w') as file:
                file.write(results)
            print(f"Validation results for {set_name} saved to {result_filename}")

    def load_rules_from_kb(self, set_name):
        """Loads previously saved rules from the knowledge base."""
        rules_filename = os.path.join('results', 'generated_rules', f'validation_rules_{set_name}.txt')
        if os.path.exists(rules_filename):
            with open(rules_filename, 'r') as file:
                return file.read()
        return None


def load_new_feature_set():
    file_path = os.path.join('..', 'data', 'insurance_claims_report.csv')
    df = pd.read_csv(file_path)
    print(f"Read {len(df)} records")
    print("Columns after loading:", df.columns.tolist())
    
    # Extract dataset name from file path
    dataset_name = os.path.basename(file_path).split('.')[0]  
    
    return df, dataset_name


if __name__ == "__main__":
    # Sample Data
    sample_data, dataset_name = load_new_feature_set()

    # Ask user to choose the AI model (Gemini or Mistral/Llama)
    model_choice = "gemini"  # Choose between "gemini" or "mistral"

    # Instantiate the validator and knowledge base interface
    knowledge_base = KnowledgeBaseInterface()
    validator = FeatureSetValidator(kb_interface=knowledge_base)

    # Configure rule generator with chosen model
    validator.rule_generator = ValidationRuleGenerator(
        api_key=os.getenv('GOOGLE_API_KEY'),
        kb_interface=knowledge_base,
        model_choice=model_choice
    )

    # Validate the feature set
    validator.validate_feature_set(dataset_name, sample_data)

    # Save validation rules and results to the knowledge base
    validation_rules = f"Sample validation rules for {dataset_name}"
    validation_results = f"Sample validation results: Age column has values greater than 100."
    knowledge_base.save_rules_to_kb(dataset_name, validation_rules)
    knowledge_base.save_validation_results(dataset_name, validation_results)
