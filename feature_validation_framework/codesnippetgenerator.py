
import subprocess
import os 
from generativemodel import * 
import pandas as pd 
import logging

class CodeSnippetGenerator:
    def __init__(self, model):
        self.model = model  # Store the generative model
        self.template_store = {}  # Store generated code snippets for reuse

    '''
    def generate_or_retrieve(self, validation_rules, feature_set_name):

        snippets = set()  

        for rule in validation_rules:
            # Assume rule is a tuple and get the string part
            rule_text = rule[0] if isinstance(rule, tuple) else rule  # Get the first element if it's a tuple

            if rule_text in self.template_store:
                snippets.add(self.template_store[rule_text])
            else:
                # Generate a prompt asking the AI to create Python code based on the rules
                prompt = f"""
                Based on the following data validation rules, generate valid and correct Python code that works to implement these rules for a pandas DataFrame named 'feature_set':

                {rule_text}

                Ensure the following:
                1. The DataFrame 'feature_set' contains the necessary columns for the validation rules.
                2. Handle missing columns or missing files gracefully, **without using exit(). Instead, log the errors into the validation results list**.
                3. Perform the necessary checks, and store validation outcomes in a dictionary, including relevant metrics.
                4. Ensure that the validation metrics include:
                - Count of total features
                - Count of missing values per column
                - Count of outliers detected
                - Any other relevant statistics based on the rules
                5. Save the validation results to a CSV file named feature_set_validation_outcomes_{rule_text.replace(" ", "_").replace(",", "")}.csv.
                6. **Ensure all code is within a function (e.g., 'validate_feature_set')**, and there should be no 'return' statements outside functions.
                7. The necessary libraries (e.g., pandas, os) must be imported in the code. If 'os' or 'pandas' is not defined, import them.
                8. If an error occurs during execution, log the error in the validation results list, and continue execution.
                9. **Create a main function to call the validation function and print the outcomes.**
                """

                new_snippet = self.model.generate(prompt)

                # Basic validation of the generated code
                if 'return' not in new_snippet or 'def ' not in new_snippet:
                    # Force the function structure and imports if missing
                    new_snippet = f"""
                    import pandas as pd
                    import os

                    def validate_feature_set(feature_set):
                        validation_results = []
                        metrics = {{
                            'total_features': len(feature_set.columns),
                            'missing_values': {{}} ,
                            'outliers': {{}} 
                        }}

                        # Add the generated validation code snippet here
                        {new_snippet.strip()}  # Strip leading/trailing whitespace

                        # Return the validation results and metrics
                        validation_results.append(f"Validation Metrics: {{metrics}}")
                        return validation_results
                    """
                snippets.add(new_snippet)
                self.template_store[rule_text] = new_snippet  

        # Save validation results from each snippet to CSV files
        for i, snippet in enumerate(snippets):
            local_vars = {
                'pd': pd,
                'feature_set': feature_set_name,
            }

            # Generate a unique filename for each snippet based on its rule
            output_filename = f'results/feature_set_validation_outcomes_{i + 1}.csv'

            # Add the output filename to local_vars for use in the snippet
            local_vars['output_filename'] = output_filename
            
            # Prepare a snippet that saves its validation results directly to a CSV file
            # After defining the function in the full snippet
            full_snippet = f"""
            {snippet}

            # Save the validation results to the specified CSV file
            validation_results_df = pd.DataFrame(validation_results)
            validation_results_df.to_csv(output_filename, index=False, header=True)
            """

            try:
                exec(full_snippet, {}, local_vars)
                print(f"Executed code snippet part {i + 1} successfully. Results saved to {output_filename}.")
            except Exception as e:
                # Log error in a dedicated file
                with open(output_filename, 'a') as f:
                    f.write(f"Execution Error: {str(e)}\n")
                print(f"Error executing snippet part {i + 1}: {e}")

        # Combine all individual CSV files into one final CSV
        combined_quality_file = 'feature_set_validation_outcomes_combined.csv'
        combined_results = []

        for i in range(1, len(snippets) + 1):  # Assuming snippets are sequentially numbered
            try:
                temp_df = pd.read_csv(f'feature_set_validation_outcomes_{i}.csv')
                combined_results.append(temp_df)
            except FileNotFoundError:
                print(f"Warning: {f'feature_set_validation_outcomes_{i}.csv'} not found.")

        if combined_results:
            final_combined_df = pd.concat(combined_results, ignore_index=True)
            final_combined_df.to_csv(combined_quality_file, index=False)
            print(f"All results combined into {combined_quality_file}.")
        else:
            print("No results to combine.")

        return list(snippets)
    '''
    def generate_or_retrieve(self, validation_rules, feature_set_name):
        snippets = set()

        for rule in validation_rules:
            # Assume rule is a tuple and get the string part
            rule_text = rule[0] if isinstance(rule, tuple) else rule  # Get the first element if it's a tuple

            if rule_text in self.template_store:
                snippets.add(self.template_store[rule_text])
            else:
                # Generate a prompt asking the AI to create Python code based on the rules
                prompt = f"""
                Based on the following data validation rules, generate valid and correct Python code that works to implement these rules for a pandas DataFrame named 'feature_set':

                {rule_text}

                Ensure the following:
                1. The DataFrame 'feature_set' contains the necessary columns for the validation rules.
                2. Handle missing columns or missing files gracefully, **without using exit(). Instead, log the errors into the validation results list**.
                3. Perform the necessary checks, and store validation outcomes in a dictionary, including relevant metrics.
                4. Ensure that the validation metrics include:
                - Count of total features
                - Count of missing values per column
                - Count of outliers detected
                - Any other relevant statistics based on the rules
                5. Save the validation results to a CSV file named feature_set_validation_outcomes_{rule_text.replace(" ", "_").replace(",", "")}.csv.
                6. **Ensure all code is within a function (e.g., 'validate_feature_set')**, and there should be no 'return' statements outside functions.
                7. The necessary libraries (e.g., pandas, os) must be imported in the code. If 'os' or 'pandas' is not defined, import them.
                8. If an error occurs during execution, log the error in the validation results list, and continue execution.
                9. **Create a main function to call the validation function and print the outcomes.**
                """

                new_snippet = self.model.generate(prompt)

                # Basic validation of the generated code
                if 'return' not in new_snippet or 'def ' not in new_snippet:
                    # Force the function structure and imports if missing
                    new_snippet = f"""
                    import pandas as pd
                    import os

                    def validate_feature_set(feature_set):
                        validation_results = []
                        metrics = {{
                            'total_features': len(feature_set.columns),
                            'missing_values': {{}} ,
                            'outliers': {{}} 
                        }}

                        # Add the generated validation code snippet here
                        {new_snippet.strip()}  # Strip leading/trailing whitespace

                        # Return the validation results and metrics
                        validation_results.append(f"Validation Metrics: {{metrics}}")
                        return validation_results
                    """
                snippets.add(new_snippet)
                self.template_store[rule_text] = new_snippet
            # Save validation results from each snippet to CSV files
        for i, snippet in enumerate(snippets):
            local_vars = {
                'pd': pd,
                'feature_set': feature_set_name,
            }

            # Generate a unique filename for each snippet based on its rule
            output_filename = f'results/feature_set_validation_outcomes_{i + 1}.csv'

            # Add the output filename to local_vars for use in the snippet
            local_vars['output_filename'] = output_filename
            
            # Prepare a snippet that saves its validation results directly to a CSV file
            # After defining the function in the full snippet
            full_snippet = f"""
            {snippet}

            # Save the validation results to the specified CSV file
            validation_results_df = pd.DataFrame(validation_results)
            validation_results_df.to_csv(output_filename, index=False, header=True)
            """

            try:
                exec(full_snippet, {}, local_vars)
                print(f"Executed code snippet part {i + 1} successfully. Results saved to {output_filename}.")
            except Exception as e:
                # Log error in a dedicated file
                with open(output_filename, 'a') as f:
                    f.write(f"Execution Error: {str(e)}\n")
                print(f"Error executing snippet part {i + 1}: {e}")

        # Combine all individual CSV files into one final CSV
        combined_quality_file = 'feature_set_validation_outcomes_combined.csv'
        combined_results = []

        for i in range(1, len(snippets) + 1):  # Assuming snippets are sequentially numbered
            try:
                temp_df = pd.read_csv(f'feature_set_validation_outcomes_{i}.csv')
                combined_results.append(temp_df)
            except FileNotFoundError:
                print(f"Warning: {f'feature_set_validation_outcomes_{i}.csv'} not found.")

        if combined_results:
            final_combined_df = pd.concat(combined_results, ignore_index=True)
            final_combined_df.to_csv(combined_quality_file, index=False)
            print(f"All results combined into {combined_quality_file}.")
        else:
            print("No results to combine.")

        return list(snippets)

    # Severity Mapping function (new addition)
    def assign_severity(issue):
        """
        Dynamically assign severity based on the issue detected.
        High severity: Critical data issues like missing columns or a large number of missing values.
        Medium severity: Moderate data issues like outliers or low correlation.
        Low severity: Minor issues like short text or date formatting problems.
        """
        if "missing columns" in issue or "missing values" in issue and "high count" in issue:
            return "high"
        elif "outliers" in issue or "low correlation" in issue:
            return "medium"
        else:
            return "low"

        # Save validation results from each snippet to CSV files
        for i, snippet in enumerate(snippets):
            local_vars = {
                'pd': pd,
                'feature_set': feature_set_name,
            }

            # Generate a unique filename for each snippet based on its rule
            output_filename = f'results/feature_set_validation_outcomes_{i + 1}.csv'

            # Add the output filename to local_vars for use in the snippet
            local_vars['output_filename'] = output_filename
            
            # Prepare a snippet that saves its validation results directly to a CSV file
            # After defining the function in the full snippet
            full_snippet = f"""
            {snippet}

            # Save the validation results to the specified CSV file
            validation_results_df = pd.DataFrame(validation_results)
            validation_results_df['severity'] = validation_results_df['issue'].apply(assign_severity)
            validation_results_df.to_csv(output_filename, index=False, header=True)
            """

            try:
                exec(full_snippet, {}, local_vars)
                print(f"Executed code snippet part {i + 1} successfully. Results saved to {output_filename}.")
            except Exception as e:
                # Log error in a dedicated file
                with open(output_filename, 'a') as f:
                    f.write(f"Execution Error: {str(e)}\n")
                print(f"Error executing snippet part {i + 1}: {e}")

        # Combine all individual CSV files into one final CSV
        combined_quality_file = 'feature_set_validation_outcomes_combined.csv'
        combined_results = []

        for i in range(1, len(snippets) + 1):  # Assuming snippets are sequentially numbered
            try:
                temp_df = pd.read_csv(f'feature_set_validation_outcomes_{i}.csv')
                combined_results.append(temp_df)
            except FileNotFoundError:
                print(f"Warning: {f'feature_set_validation_outcomes_{i}.csv'} not found.")

        if combined_results:
            final_combined_df = pd.concat(combined_results, ignore_index=True)
            final_combined_df.to_csv(combined_quality_file, index=False)
            print(f"All results combined into {combined_quality_file}.")
        else:
            print("No results to combine.")

        return list(snippets)
