import os
import pandas as pd
import logging
import google.generativeai as genai
from datetime import datetime

class ValidationRuleGenerator:
    """
    This class is responsible for generating data validation rules dynamically 
    based on metadata and statistics.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

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

    def generate_validation_rules(self, metadata, stats):
        """
        Generates the validation rules using the GenAI model based on the metadata and statistics.
        """
        prompt = self.generate_prompt_from_metadata(metadata, stats)
        print(f"Generated prompt for feature validation:\n{prompt}")
        rules = self.generate_response(text=prompt)
        return rules

    def generate_response(self, text: str):
        """
        Sends the prompt to the Generative AI model and retrieves the response (rules).
        """
        print("Fetching response from GenAI, please wait...")
        response = self.model.generate_content(text)
        if response and response.text:
            return response.text
        else:
            logging.warning("Gemini response was empty or blocked. Check safety ratings.")
            return None


class FeatureSetValidator:
    """
    Validates feature sets by generating and applying validation rules, 
    and stores validation results for easy integration with a knowledge base.
    """

    def __init__(self):
        self.rule_generator = ValidationRuleGenerator(api_key=os.getenv('GOOGLE_API_KEY'))

    def validate_feature_set(self, set_name, feature_set):
        """
        Validates a feature set, applies the validation rules, and stores the results.
        """
        print(f"Validating feature set '{set_name}'...")

        # Step 1: Define Metadata
        metadata = self.define_metadata(feature_set)
        stats = feature_set.describe(include='all').T

        # Step 2: Generate Validation Rules
        rules = self.rule_generator.generate_validation_rules(metadata=metadata, stats=stats)
        print(f"Generated Validation Rules for {set_name}:\n{rules}")

        # Step 3: Custom Validation Issues
        issues = self.dynamic_custom_rules(feature_set)

        if issues:
            print(f"Data Quality Issues Found in {set_name}:")
            for issue in issues:
                print(f"- {issue}")
            self.write_feature_quality_to_table(set_name, issues)
        else:
            print(f"No data quality issues found in {set_name}.")
            self.write_feature_quality_to_table(set_name, ["No issues found"])

        # Step 4: Save Validation Metrics and Rules
        self.save_generated_rules(rules, set_name)
        print(f"Validation for '{set_name}' completed.")

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
        Applies dynamic custom rules to validate the dataset, such as checking for suspicious values.
        """
        custom_issues = []
        # Example: check if any column exceeds a threshold or has suspicious values
        if 'age' in df.columns and df['age'].max() > 100:
            custom_issues.append(f"Age column has values greater than 100.")
        return custom_issues

    def write_feature_quality_to_table(self, feature_name, issues):
        """
        Writes feature quality issues to a table or file for easy reference.
        """
        if not os.path.exists('feature_quality_table.csv'):
            with open('feature_quality_table.csv', 'w') as f:
                f.write('Feature,Issues\n')
        with open('feature_quality_table.csv', 'a') as f:
            for issue in issues:
                f.write(f"{feature_name},{issue}\n")
        print(f"Feature quality issues for '{feature_name}' saved to table.")

    def save_generated_rules(self, rules, set_name):
        """
        Saves the generated validation rules to a file for easy sharing or inclusion in a knowledge base.
        """
        

        results_dir = os.path.join('results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        rules_filename = os.path.join(results_dir, f'generated_rules_{set_name}.txt')

        with open(rules_filename, 'w', encoding="utf-8") as file:
            file.write(rules)
        print(f"Generated rules for '{set_name}' saved to {rules_filename}")


class KnowledgeBaseInterface:
    """
    Interface for storing validation results and rules in a format that can be shared or integrated into a knowledge base.
    """

    def __init__(self, kb_directory='knowledge_base'):
        self.kb_directory = kb_directory
        if not os.path.exists(self.kb_directory):
            os.makedirs(self.kb_directory)

    def save_rules_to_kb(self, set_name, rules):
        """Saves generated validation rules to the knowledge base (a simple file for now)."""
        # Ensure the 'generated_rules' directory exists inside 'results'
        rules_dir = os.path.join('results', 'generated_rules')
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)

        # Save the rules in the generated_rules directory
        rules_filename = os.path.join(rules_dir, f'validation_rules_{set_name}.txt')
        with open(rules_filename, 'w') as file:
            file.write(rules)
        print(f"Validation rules for {set_name} saved to {rules_filename}")

    def save_validation_results(self, set_name, results):
        """Saves validation results to the knowledge base."""
        results_dir = os.path.join('results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        result_filename = os.path.join(results_dir, f'validation_results_{set_name}.txt')
        with open(result_filename, 'w') as file:
            file.write(results)
        print(f"Validation results for {set_name} saved to {result_filename}")


def load_new_feature_set():
        
        file_path = os.path.join('..', 'data', 'insurance_claims_report.csv')
        df = pd.read_csv(file_path)
        print(f"Read {len(df)} records")
        print("Columns after loading:", df.columns.tolist())
        
        return df

if __name__ == "__main__":
    # Sample Data
    sample_data = load_new_feature_set()

    # Instantiate the validator and knowledge base interface
    validator = FeatureSetValidator()
    knowledge_base = KnowledgeBaseInterface()

    # Validate the feature set
    set_name = 'customer_data'
    validator.validate_feature_set(set_name, sample_data)

    # Save validation rules and results to the knowledge base
    validation_rules = "Sample validation rules for customer_data"
    validation_results = "Sample validation results: Age column has values greater than 100."
    knowledge_base.save_rules_to_kb(set_name, validation_rules)
    knowledge_base.save_validation_results(set_name, validation_results)
