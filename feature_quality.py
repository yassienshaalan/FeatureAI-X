import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
from langdetect import detect
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import OpenAI
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
import google.generativeai as genai
import logging
from datetime import datetime

class Logger(object):
    def __init__(self, log_filename):
        self.terminal = sys.stdout
        self.log = open(log_filename, "a", encoding="utf-8")  

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass  

def create_directories():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('results'):
        os.makedirs('results')

def generate_filename(base_dir, prefix):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Determine the run number based on existing files
    run_number = len([f for f in os.listdir(base_dir) if f.startswith(prefix + date_str)]) + 1
    filename = f"{prefix}_{date_str}_run{run_number}.log"
    
    return os.path.join(base_dir, filename)

def setup_logging():

    create_directories()
    # Generate the log filename with time and run number
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join('logs', f'log_{date_str}.log')

    # Set up logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This logs to console
        ]
    )
    logging.info("Logging setup complete.")
    # Redirect print statements to log file as well
    sys.stdout = Logger(log_filename)


def save_metrics(self, metrics, set_name):
    # Save the metrics to a file
    metrics_filename = os.path.join('results', f'validated_metrics_{set_name}.txt')
    with open(metrics_filename, 'w', encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

    print(f"Metrics saved to {metrics_filename}")

class DataDriftDetector:
    def calculate_drift(self, baseline_df, current_df):
        drift_issues = []
        for column in baseline_df.columns:
            if column in current_df.columns:
                if pd.api.types.is_numeric_dtype(baseline_df[column]):
                    stat, p_value = ks_2samp(baseline_df[column].dropna(), current_df[column].dropna())
                    if p_value < 0.05:
                        drift_issues.append(f"'{column}' shows significant drift (KS test p-value: {p_value}).")
                elif pd.api.types.is_categorical_dtype(baseline_df[column]) or pd.api.types.is_object_dtype(baseline_df[column]):
                    baseline_dist = baseline_df[column].value_counts(normalize=True)
                    current_dist = current_df[column].value_counts(normalize=True)
                    if not baseline_dist.equals(current_dist):
                        drift_issues.append(f"'{column}' shows significant drift in category distribution.")
                elif pd.api.types.is_string_dtype(baseline_df[column]):
                    baseline_len_mean = baseline_df[column].str.len().mean()
                    current_len_mean = current_df[column].str.len().mean()
                    if abs(current_len_mean - baseline_len_mean) > 0.1 * baseline_len_mean:
                        drift_issues.append(f"'{column}' shows significant drift in text length (Baseline mean length: {baseline_len_mean}, Current mean length: {current_len_mean}).")
        return drift_issues

class ValidationRuleGenerator:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')  # If using Google Colab 
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
     
    def generate_prompt_from_metadata(self, metadata, stats):
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
                prompt += f"- '{column}' should be checked for completeness, readability, sentiment, and keyword presence.\n"
        prompt += "Ensure that the rules cover completeness, consistency, uniqueness, range checks, and any potential outliers."
        prompt += " Additionally, provide guidance on detecting feature drift compared to a baseline dataset."
        return prompt

    def generate_validation_rules_with_langchain(self, metadata, stats):
        template = PromptTemplate(
            input_variables=["metadata", "stats"],
            template=self.generate_prompt_from_metadata(metadata, stats)
        )
        print("template prompt for feature validation")
        print(template)
        
        llm = OpenAI(openai_api_key='your-api-key-here')
        chain = LLMChain(prompt_template=template, llm=llm)
        rules = chain.run({"metadata": metadata, "stats": stats})
        return rules
    
    def generate_response(self, text: str):
        # Use Gemini to generate a text response
        print("Fetching response, please wait......")
        response = self.model.generate_content(text)
        if response and response.text:
            return response.text
        else:
            logging.warning("Gemini response was empty or blocked. Check safety ratings.")
            return None

    def generate_validation_rules_with_genai(self, metadata, stats):
        print("Generating validation rules with gemini")
        prompt = self.generate_prompt_from_metadata(metadata, stats)
        print("Generated prompt for feature validation:")
        print(prompt)
        # Using the Generative AI model to generate rules
        rules = self.generate_response(text=prompt)
        return rules

class FeatureSetValidator:
    def __init__(self):
        self.drift_detector = DataDriftDetector()
        self.rule_generator = ValidationRuleGenerator()
        self.api_key = os.getenv('GOOGLE_API_KEY')  # If using Google Colab 
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
       

    def generate_code_snippets(self, issues, df_name='df'):
        code_snippets = []

        # Initialize a dictionary to store the validation outcomes
        code_snippets.append("validation_outcomes = {}  # Dictionary to store outcomes\n")

        # Iterate over the issues
        for issue in issues:
            # Extract the column name from the issue (assuming the issue includes the column name)
            column_name = self.extract_column_name(issue)

            # Generate code to handle missing values
            if "missing values" in issue.lower():
                snippet = f"""
    # Check and handle missing values in '{column_name}' column
    if '{column_name}' in {df_name}.columns:
        missing_values = {df_name}['{column_name}'].isnull().sum()
        if missing_values > 0:
            validation_outcomes['{column_name}_missing_values'] = f"Missing values detected: {{missing_values}} in '{column_name}'"
            {df_name}['{column_name}'].fillna({df_name}['{column_name}'].median(), inplace=True)
            validation_outcomes['{column_name}_missing_values_filled'] = "Missing values in '{column_name}' filled with median."
        else:
            validation_outcomes['{column_name}_missing_values'] = "No missing values detected in '{column_name}'."
    else:
        validation_outcomes['{column_name}_missing_values'] = "'{column_name}' column not found."
    """
                code_snippets.append(snippet)

            # Generate code to handle drift detection
            if "drift" in issue.lower():
                snippet = f"""
    # Check for drift in '{column_name}' column distributions
    if '{column_name}' in {df_name}.columns:
        baseline_stats = baseline_df['{column_name}'].describe()
        current_stats = {df_name}['{column_name}'].describe()
        drift_detected = False
        for stat in baseline_stats.index:
            if not baseline_stats[stat] == current_stats[stat]:
                validation_outcomes['{column_name}_drift'] = f"Drift detected in '{column_name}': {{stat}}"
                drift_detected = True
        if not drift_detected:
            validation_outcomes['{column_name}_drift'] = "No drift detected in '{column_name}'."
    else:
        validation_outcomes['{column_name}_drift'] = "'{column_name}' column not found."
    """
                code_snippets.append(snippet)

            # Add more cases as needed for other types of issues
            # Example: Handling categorical consistency, text analysis, etc.

        # Join all code snippets into a single string and return
        return ''.join(code_snippets)

    def extract_column_name(self, issue):
        """
        Extract the column name from the issue description.
        This is a placeholder function. You need to implement the logic based on how the column names are mentioned in the issue.
        """
        # Example logic assuming column names are mentioned in quotes within the issue string
        import re
        match = re.search(r"'([^']*)'", issue)
        if match:
            return match.group(1)
        return "unknown_column"  # Fallback in case the column name can't be extracted

    def generate_code_snippets_with_genai(self, rules):
        # Generate a prompt that asks the AI to create Python code based on the rules
        #prompt = f"Generate Python code to validate the following data based on these rules:\n{rules}\n"
        prompt = f"""
        Based on the following data validation rules, generate Python code to implement these rules for a pandas DataFrame named 'feature_set':

        {rules}

        The code should perform the necessary checks and save the validation outcomes, including all relevant metrics, to a CSV file named 'validation_results_{set_name}.csv'.
        """
        code_snippets = self.model.generate_content(prompt).text
        return code_snippets

    def save_metrics(self, metrics, set_name):
        # Save the metrics to a file
        metrics_filename = os.path.join('results', f'validated_metrics_{set_name}.txt')
        with open(metrics_filename, 'w') as file:
            for key, value in metrics.items():
                file.write(f"{key}: {value}\n")

        print(f"Metrics saved to {metrics_filename}")

    def save_generated_rules(self, rules, set_name):
        # Save the generated rules to a file
        rules_filename = os.path.join('results', f'generated_rules_{set_name}.txt')
        with open(rules_filename, 'w', encoding="utf-8") as file:
            file.write(rules)

        print(f"Generated rules saved to {rules_filename}")

    def write_feature_quality_to_table(self, feature_set_name, issues):
        # Implement how to write feature quality issues to a table or file
        print(f"Writing feature quality issues for '{feature_set_name}'...")

    def dynamic_custom_rules(self, df):
        # Implement your dynamic custom rules logic
        return []

    def define_metadata(self, df):
        metadata = {
            "columns": df.columns.tolist(),
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "missing_values": df.isnull().sum().sum(),
            "unique_counts": df.nunique(),
            "data_types": df.dtypes.to_dict()  # Include the data types
        }
        return metadata

    def save_and_execute_code_snippets(self, code_snippets, set_name, feature_set):

        # Strip out the markdown code block markers
        if code_snippets.startswith("```python"):
            code_snippets = code_snippets[len("```python"):].strip()
        if code_snippets.endswith("```"):
            code_snippets = code_snippets[:-len("```")].strip()

         # Define the dataset file name
        dataset_filename = f'data_{set_name}.csv'
        
        # Save the dataset to a CSV file
        feature_set.to_csv(dataset_filename, index=False)
        print(f"Dataset for '{set_name}' saved as {dataset_filename}")
        
            # Replace the dataset reference in code snippets (if necessary)
        code_snippets = code_snippets.replace('feature_set.csv', f'{dataset_filename}')

        # Save the code snippets to a file
        code_snippet_filename = os.path.join('results', f'code_snippets_{set_name}.py')
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


    def validate_feature_set(self, set_name, feature_set, baseline_df):
        print(f"Validating feature set '{set_name}'...")

        # Define metadata for the feature set
        metadata = self.define_metadata(feature_set)  # Correctly define the metadata

        stats = self.analyze_feature_set(feature_set)
        drift_issues = self.drift_detector.calculate_drift(baseline_df, feature_set)

        # Generate validation rules using Generative AI
        rules = self.rule_generator.generate_validation_rules_with_genai(metadata=metadata, stats=stats)
        print(f"Generated Validation Rules for {set_name}:\n", rules)

        # Apply dynamic custom rules and check for issues (nanual if needed)
        issues = self.dynamic_custom_rules(feature_set)
        issues.extend(drift_issues)

        if issues:
            print(f"Data Quality Issues Found in {set_name}:")
            for issue in issues:
                print(f"- {issue}")
            self.write_feature_quality_to_table(set_name, issues)

            # Generate code snippets to address the issues
            #code_snippets = self.generate_code_snippets(issues, df_name='feature_set')
            code_snippets = self.generate_code_snippets_with_genai(rules)
            # Save and execute the generated code snippets
            validation_outcomes = self.save_and_execute_code_snippets(code_snippets, set_name, feature_set)

            # Merge validation outcomes into metrics
            metrics = {
                'total_features': len(feature_set.columns),
                'missing_values': feature_set.isnull().sum().sum(),
                'drift_detected': bool(drift_issues),
                'validation_outcomes': validation_outcomes  # Add validation outcomes to metrics
            }
        else:
            print(f"No data quality issues found in {set_name}.")
            self.write_feature_quality_to_table(set_name, ["No issues found"])

            metrics = {
                'total_features': len(feature_set.columns),
                'missing_values': feature_set.isnull().sum().sum(),
                'drift_detected': bool(drift_issues),
                'validation_outcomes': "No issues found"
            }

        # Save the validated metrics
        self.save_metrics(metrics, set_name=set_name)

        # Save the generated rules
        self.save_generated_rules(rules, set_name=set_name)

        print(f"Validation for '{set_name}' completed.")

    def analyze_feature_set(self, df):
        return df.describe(include='all').T

    def dynamic_custom_rules(self, df):
        custom_issues = []

        # Example dynamic custom rule: Check if 'age' exceeds a certain threshold based on context
        if 'age' in df.columns:
            max_age_threshold = 100  # Can be dynamically set based on external factors or data context
            if df['age'].max() > max_age_threshold:
                custom_issues.append(f"'age' should not exceed {max_age_threshold}. Found {df['age'].max()}.")

        # Another dynamic rule: 'annual_premium' should be within a realistic range
        if 'policy_annual_premium' in df.columns:
            if df['policy_annual_premium'].max() > 50000:
                custom_issues.append(f"'policy_annual_premium' should not exceed 50000. Found {df['policy_annual_premium'].max()}.")

        # Example dynamic custom rule for text validation: Ensure comments are non-empty, in English, and have positive sentiment
        if 'comments' in df.columns:
            for idx, text in df['comments'].items():  # Use .items() instead of .iteritems()
                if not text or len(text.strip()) == 0:
                    custom_issues.append(f"Row {idx} in 'comments' is empty.")
                else:
                    detected_lang = detect(text)
                    if detected_lang != 'en':
                        custom_issues.append(f"Row {idx} in 'comments' is not in English (detected: {detected_lang}).")
                    # Example sentiment analysis (placeholder for actual implementation)
                    sentiment = "positive"  # In practice, use a sentiment analysis model here
                    if sentiment != "positive":
                        custom_issues.append(f"Row {idx} in 'comments' has negative sentiment.")

        return custom_issues

    def write_feature_quality_to_table(self, feature_name, issues):
        if not os.path.exists('feature_quality_table.csv'):
            with open('feature_quality_table.csv', 'w') as f:
                f.write('Feature,Issues\n')
        with open('feature_quality_table.csv', 'a') as f:
            for issue in issues:
                f.write(f"{feature_name},{issue}\n")


class FeatureStoreManager:
    def load_new_feature_set(self, version='v1'):
        if version == 'v1':
            file_path = os.path.join('data', 'insurance_claims_report.csv')
        elif version == 'v2':
            file_path = os.path.join('data', 'insurance_claims_report_v2.csv')  
        df = pd.read_csv(file_path)
        print("Read",len(df),"records")
        print("Columns after loading:", df.columns.tolist())  
        return df

    def load_baseline_feature_set(self):
        return self.load_new_feature_set(version='v1')  

class GenAISyntheticFeatureGenerator:
    def __init__(self, stats, metadata, text_model_name='gpt2'):
        self.stats = stats
        self.metadata = metadata
        self.api_key = os.getenv('GOOGLE_API_KEY')  # If using Google Colab 
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize the GPT-2 model and tokenizer for text generation
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)
        self.text_model = GPT2LMHeadModel.from_pretrained(text_model_name)

    def generate_synthetic_data(self, num_rows):
        synthetic_data = {}

        for column in self.stats.index:
            dtype = self.metadata['data_types'][column]
            if pd.api.types.is_numeric_dtype(dtype):
                synthetic_data[column] = self.generate_numeric_data_with_gan(column, num_rows)
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                synthetic_data[column] = self.generate_categorical_data(column, num_rows)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                synthetic_data[column] = self.generate_datetime_data(column, num_rows)
            elif pd.api.types.is_string_dtype(dtype):
                synthetic_data[column] = self.generate_text_data_with_gpt(column, num_rows)
            else:
                synthetic_data[column] = [None] * num_rows  # Fallback for unsupported types

        return pd.DataFrame(synthetic_data)
    
    def generate_synthetic_data_with_genai(self, num_rows, stats):
        # Example of generating a prompt for synthetic data
        prompt = f"Generate {num_rows} synthetic rows based on the following statistics: {stats}"

        # Using the Generative AI model to generate synthetic data
        synthetic_response = self.model.generate_content(prompt)

        print("synthetic_response")
        print(synthetic_response.text)

        synthetic_data = synthetic_response.text  # Get the actual text content

        # Manually parse the synthetic data
        try:
            # Convert the synthetic data (table-like string) to a list of lines
            lines = d.strip().split('\n')

            # Extract the headers from the first line
            headers = [header.strip() for header in lines[0].split('|') if header.strip()]

            # Debugging: Print the headers and first few lines
            print("Headers extracted:", headers)
            print("First few lines of the data:", lines[:5])

            # Extract the data rows
            rows = []
            for line in lines[2:]:  # Start from the third line to skip the header separator
                row = [value.strip() for value in line.split('|') if value.strip()]
                rows.append(row)

            # Convert the list of rows into a DataFrame
            synthetic_df = pd.DataFrame(rows, columns=headers)

            # Convert numeric columns back to their proper data types
            #for column in ['age', 'months_as_customer', 'policy_annual_premium']:
            #    synthetic_df[column] = pd.to_numeric(synthetic_df[column], errors='coerce')
            # Detect and convert numeric columns back to their proper data types
            '''
            for column in synthetic_df.columns:
                try:
                    # Attempt to convert the column to numeric
                    synthetic_df[column] = pd.to_numeric(synthetic_df[column], errors='coerce')
                    # Check if the conversion was successful by counting non-NaN entries
                    if synthetic_df[column].notna().sum() > 0:
                        print(f"Converted column '{column}' to numeric.")
                except Exception as convert_error:
                    print(f"Could not convert column '{column}' to numeric: {convert_error}")
            '''

        except Exception as e:
            print(f"Error parsing the synthetic data: {e}")
            synthetic_df = pd.DataFrame()  # Return an empty DataFrame on error

        return synthetic_df

    def generate_numeric_data_with_gan(self, column, num_rows):
        """
        Generate synthetic numeric data using a simple GAN model.
        """
        # Parameters for GAN
        data = self.metadata['original_df'][column].dropna().values.reshape(-1, 1)
        noise_dim = 100
        epochs = 10000
        batch_size = 64
        adam = Adam(lr=0.0002, beta_1=0.5)

        # Generator Model
        generator = Sequential([
            Dense(256, input_dim=noise_dim),
            LeakyReLU(alpha=0.2),
            Dense(512),
            LeakyReLU(alpha=0.2),
            Dense(1024),
            LeakyReLU(alpha=0.2),
            Dense(1, activation='linear')
        ])
        generator.compile(loss='binary_crossentropy', optimizer=adam)

        # Training GAN
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            generated_data = generator.predict(noise)
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            combined_data = np.vstack((real_data, generated_data))
            labels = np.ones((2 * batch_size, 1))

            generator.train_on_batch(noise, np.ones((batch_size, 1)))

        noise = np.random.normal(0, 1, (num_rows, noise_dim))
        synthetic_data = generator.predict(noise)
        return synthetic_data.flatten().tolist()

    def generate_text_data_with_gpt(self, column, num_rows):
        """
        Generate synthetic text data using GPT-2 model.
        """
        generated_texts = []
        original_texts = self.metadata['original_df'][column].dropna().tolist()
        
        for _ in range(num_rows):
            input_text = np.random.choice(original_texts)
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            outputs = self.text_model.generate(inputs, max_length=50, num_return_sequences=1)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return shuffle(generated_texts)

    def generate_categorical_data(self, column, num_rows):
        original_dist = self.metadata['original_df'][column].value_counts(normalize=True)
        categories = original_dist.index.tolist()
        probabilities = original_dist.values.tolist()

        synthetic_values = np.random.choice(categories, size=num_rows, p=probabilities)
        return synthetic_values.tolist()

    def generate_datetime_data(self, column, num_rows):
        min_date = self.stats.loc[column, 'min']
        max_date = self.stats.loc[column, 'max']
        synthetic_dates = pd.to_datetime(np.random.randint(min_date.value, max_date.value, size=num_rows))
        return synthetic_dates.tolist()

def save_synthetic_data(synthetic_data, set_name):
    create_directories()
    results_filename = os.path.join('results', f'synthetic_data_{set_name}.csv')
    synthetic_data.to_csv(results_filename, index=False)
    logging.info(f"Synthetic data for '{set_name}' saved to {results_filename}")


def save_generated_rules(rules, set_name):
    create_directories()
    rules_filename = os.path.join('results', f'generated_rules_{set_name}.txt')
    
    with open(rules_filename, 'w') as file:
        file.write(rules)
    
    logging.info(f"Generated rules for '{set_name}' saved to {rules_filename}")


if __name__ == "__main__":

    setup_logging()

    print("This will be printed to the console and logged to the file.")
    logging.info("This is an info message that will also be logged.")

    # Usage Example
    print("Loading baseline and new feature sets...")
    feature_store_manager = FeatureStoreManager()
    print("Loading baseline df")
    baseline_df = feature_store_manager.load_baseline_feature_set()
    print("Loading new df")
    feature_df = feature_store_manager.load_new_feature_set()
    print("Feature sets loaded successfully.")

    print("Available columns in feature_df:", feature_df.columns.tolist())

    print("Specify the date format explicitly")
    # Specify the correct date format for 'policy_bind_date' and 'incident_date'
    feature_df['policy_bind_date'] = pd.to_datetime(feature_df['policy_bind_date'], format='%d/%m/%y')
    feature_df['incident_date'] = pd.to_datetime(feature_df['incident_date'], format='%d/%m/%y')


    # Adding simulated text feature
    print("Adding simulated text feature...")
    feature_df['comments'] = ([
        "This is a sample comment.",
        "Another example of text data.",
        "",
        "Validating text fields with GPT-4.",
        "Check language consistency.",
    ] * (len(feature_df) // 5 + 1))[:len(feature_df)]

    print("Simulated text feature added.")

    print("Handling missing values and encoding categorical features...")
    # Handling missing values
    # Fill numeric columns with the median value
    print("Fill numeric columns")
    numeric_columns = feature_df.select_dtypes(include=['number']).columns
    feature_df[numeric_columns] = feature_df[numeric_columns].fillna(feature_df[numeric_columns].median())
    print("Fill categorical columns")
    # Fill categorical columns with the mode (most frequent value)
    categorical_columns = feature_df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        feature_df[column].fillna(feature_df[column].mode()[0], inplace=True)

    # Encoding categorical variables
    le = LabelEncoder()
    for column in feature_df.select_dtypes(include=['object']).columns:
        if column not in ['comments']:  # Exclude non-categorical text features if present
            feature_df[column] = le.fit_transform(feature_df[column])

    # Ensure all required columns exist before splitting
    required_columns = ['age', 'months_as_customer', 'policy_annual_premium', 'comments']
    missing_columns = [col for col in required_columns if col not in feature_df.columns]
    feature_sets = {}
    if missing_columns:
        print(f"Error: The following columns are missing and cannot be included in the feature sets: {missing_columns}")
    else:
        # Splitting the dataset into multiple feature sets (for simulation purposes)
        feature_sets = {
            'set1': feature_df[['age', 'months_as_customer', 'policy_annual_premium', 'comments']],
            'set2': feature_df[['total_claim_amount', 'incident_severity', 'age', 'comments']],
            'set3': feature_df[['vehicle_claim', 'property_damage', 'incident_hour_of_the_day', 'comments']]
        }
        print("Feature sets created successfully.")
        print("feature_sets",feature_sets)

    print("feature_sets out",feature_sets)
    # Define metadata and stats for each feature set
    for set_name, feature_set in feature_sets.items():
        print(f"\nProcessing feature set '{set_name}'...")

        # Step 1: Define Metadata
        print("  Defining metadata...")
        metadata = {
            "columns": feature_set.columns.tolist(),
            "num_rows": feature_set.shape[0],
            "num_columns": feature_set.shape[1],
            "missing_values": feature_set.isnull().sum().sum(),
            "unique_counts": feature_set.nunique(),
            "data_types": feature_set.dtypes,
            "original_df": feature_set  # Keep a reference to the original DataFrame
        }
        print("  Metadata defined.")

        # Step 2: Generate Statistics
        print("  Generating statistics...")
        stats = feature_set.describe(include='all').T
        print("  Statistics generated.")

        # Step 3: Validate Feature Set
        print("  Validating feature set...")
        validator = FeatureSetValidator()  # Assuming you have a validator class
        validator.validate_feature_set(set_name, feature_set, baseline_df)
        print(f"  Feature set '{set_name}' validated.")

        # Step 4: Generate Synthetic Data (Optional)
        print(f"  Generating synthetic data for feature set '{set_name}'...")
        genai_synth_gen = GenAISyntheticFeatureGenerator(stats, metadata)  # Assuming a synthetic data generator class
        #synthetic_data = genai_synth_gen.generate_synthetic_data(num_rows=1000)
        synthetic_data = genai_synth_gen.generate_synthetic_data_with_genai(num_rows=1000,stats=stats)
        # Save the synthetic data
        print("Saving the synthetic data")
        save_synthetic_data(synthetic_data, set_name)

        print(f"  Synthetic data generated for feature set '{set_name}'.")

        # Step 5: Output or Save Results
        print(f"Synthetic Data for {set_name}:")
        print(synthetic_data.head())
    

    print("Processing of all feature sets completed.")


