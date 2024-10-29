import os 
import json
import csv 
from datetime import datetime
from generativemodel import * 
import pandas as pd 
import logging

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

    def save_validated_outcome(self,validation_outcomes,set_name):
        results_dir = os.path.join(os.getcwd(), "results")  # Path to the results directory
        os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists
        # Save validation outcomes to CSV
        validation_outcome_filename = os.path.join(results_dir, f'{set_name}_validation_outcomes.csv')
        
        try:
            if validation_outcomes:
                pd.DataFrame.from_dict(validation_outcomes, orient='index').reset_index().to_csv(validation_outcome_filename, header=["Feature", "Outcome"], index=False)
                print(f"Validation outcomes saved as {validation_outcome_filename}")
            else:
                print("No validation outcomes to save.")
        except Exception as e:
            print(f"Error saving validation outcomes: {e}")

    def validate_feature_set(self, metadata, stats, set_name, feature_set):
        # Step 1: Generate or retrieve validation rules.
        rules = self.rule_generator.generate_validation_rules(metadata, stats, set_name)
        print(f"Generated Validation Rules for {set_name}:\n{rules}")
        self.save_generated_rules(rules, set_name)
        # Step 2: Generate Python code snippets based on the rules.
        #code_snippets = self.code_generator.generate_code_snippets(rules, set_name)
        code_snippets = self.code_generator.generate_or_retrieve(rules, set_name)

        # Step 3: Execute the generated code snippets within the context of the provided dataset.
        print("Execute the generated code snippets within the context of the provided dataset")
        validation_outcomes = self.save_and_execute_code_snippets(code_snippets, set_name, feature_set)
        print(f"Validation outcomes for '{set_name}': {validation_outcomes}")
        self.save_validated_outcome(validation_outcomes,set_name)
        # Step 4: Custom Validation Issues
        issues = self.dynamic_custom_rules(feature_set)

        if issues:
            print(f"Data Quality Issues Found in {set_name}:")
            for issue in issues:
                print(f"- {issue}")
            self.write_feature_quality_to_table(set_name, issues)

            # Generate Python code snippets based on the generated rules
            code_snippets = self.code_generator.generate_or_retrieve(rules,set_name+"_dynamic_rules")
            #print(f"Generated Python code for validation:\n{code_snippets}")

            # Save and execute the generated code snippets
            validation_outcomes = self.save_and_execute_code_snippets(code_snippets, set_name+"_dynamic_rules", feature_set)
            self.save_validated_outcome(validation_outcomes,set_name+"_dynamic_rules")
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
        #self.kb_interface.save_rules_to_kb(set_name, rules)

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
        # If code_snippets is a list, process each snippet individually and remove duplicates
        if code_snippets == None:
            print('No code snippets to save in featurevalidator')
            return
        if isinstance(code_snippets, list):
            processed_snippets = []
            for snippet in code_snippets:
                snippet = snippet.strip()

                # Strip out the markdown code block markers from each snippet
                if snippet.startswith("```python"):
                    snippet = snippet[len("```python"):].strip()
                if snippet.endswith("```"):
                    snippet = snippet[:-len("```")].strip()

                processed_snippets.append(snippet)

            # Remove duplicates and store snippets as individual parts
            code_snippets = list(set(processed_snippets))

        # Define the path for the dataset file (relative path)
        results_dir = os.path.join(os.getcwd(), "results")  # Path to the results directory
        os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists

        orig_dataset_filename = f'{set_name}.csv'
        dataset_filename = os.path.join(results_dir, orig_dataset_filename)  # Path to the dataset file

        # Get the absolute path for better reliability
        abs_dataset_filename = os.path.abspath(dataset_filename)
        
        # Validate the dataset before saving
        if feature_set.empty:
            print("Error: The dataset is empty.")
            return {}

        print(f"Dataset saved as {abs_dataset_filename}")

        # Save the dataset to a CSV file
        try:
            feature_set.to_csv(dataset_filename, index=False)
            print(f"Dataset for '{set_name}' saved as {abs_dataset_filename}")
        except Exception as e:
            print(f"Error saving the dataset: {e}")
            return {}

        validation_outcomes = {}

        # Iterate through the snippets and log any replacements made
        for i, snippet in enumerate(code_snippets):
            # Replace references to 'feature_set.csv' with the absolute dataset filename
            abs_dataset_filename = abs_dataset_filename.replace("\\", "/")  # Replace backslashes with forward slashes
            snippet = snippet.replace('feature_set.csv', abs_dataset_filename)

            # Log that we are executing a snippet
            print(f"Executing code snippet part {i+1}...")

            # Debugging: Confirm file paths before execution
            #print(f"Dataset path: {abs_dataset_filename}")
            #print(f"Current working directory: {os.getcwd()}")
            #print(f"Files in results directory: {os.listdir(results_dir)}")

            # Check if the dataset file exists before execution
            if not os.path.exists(abs_dataset_filename):
                print(f"Error: The dataset file {abs_dataset_filename} does not exist.")
                continue

            # Define a file path for each snippet
            snippet_file_path = os.path.join(results_dir, f"code_snippet_part_{i+1}.py")

            # Write each snippet to its own Python file
            with open(snippet_file_path, "w") as snippet_file:
                snippet_file.write(f"# Code snippet part {i+1}\n")
                snippet_file.write(snippet)
                snippet_file.write("\n")  # Add a newline at the end for clarity

            # Execute the code snippet in a safe local scope
            local_vars = {
                'pd': pd,  # Provide pandas context for code snippets
                'feature_set': feature_set  # Provide access to the dataset
            }

            try:
                exec(snippet, {}, local_vars)  # Execute the current code snippet
                snippet_outcome = local_vars.get('validation_outcomes', {})
                validation_outcomes.update(snippet_outcome)  # Aggregate the outcomes
                print("snippet_outcome",snippet_outcome)
                print(f"Executed code snippet part {i+1} successfully.")

            except Exception as e:
                print(f"Error executing code snippet part {i+1}: {e}")

                logging.error(f"Error executing code snippet part {i+1}: {e}")
                # Instead of stopping, log the error and continue to the next snippet
                continue

        print(f"Validation outcomes for '{set_name}': {validation_outcomes}")
        return validation_outcomes
     
    def save_generated_rules(self, rules, set_name):
        # Define the JSON file path
        rules_filename = os.path.join('results', f'generated_rules_{set_name}.json')

        rules_dict = {}
        current_section = None
        
        # Handle both string and list types for rules
        if isinstance(rules, list) and len(rules) > 0:
            # Convert tuples to strings if necessary
            rules = "\n".join(f"{rule[0]}: {rule[1]}" if isinstance(rule, tuple) else rule for rule in rules)

        for line in rules.splitlines():
            if line.startswith("**") and line.endswith("**"):
                current_section = line.strip("**").strip()  
                rules_dict[current_section] = []
            elif current_section:
                rules_dict[current_section].append(line.strip())

        with open(rules_filename, 'w', encoding="utf-8") as file:
            json.dump(rules_dict, file, indent=4)  

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