import os
import pandas as pd
import logging
import sys
from datetime import datetime
import numpy.typing as npt
from typing import Optional
#from sentence_transformers import SentenceTransformer


from featuresetvalidator import *
from validationrulegenerator import *
from driftdetector import *
from generativemodel import * 
from knowledgebaseinterface import * 
from vectordatabase import *
from ragvalidation import *
from codesnippetgenerator import * 

from typing import List, Optional, Tuple

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


class KnowledgeBaseInterfaceDB:
    def __init__(self, db_path='knowledge_base.db'):
        """
        Interface for storing validation results and rules in a SQLite database,
        with logging for each step of the operation.
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
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

class RAGValidationDB:
    def __init__(self, knowledge_base, generator):
        self.knowledge_base = knowledge_base
        self.generator = generator

    def validate(self, feature_set, set_name):
        metadata = {
        'data_types': feature_set.dtypes.apply(lambda dtype: dtype.name).to_dict(),
        'unique_counts': feature_set.nunique().to_dict(),
        'num_rows': feature_set.shape[0] 
        }
        existing_rules = self.knowledge_base.retrieve(set_name, metadata)
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

def generate_metadata_and_stats(df):
    metadata = {
        'data_types': df.dtypes.apply(lambda dtype: dtype.name).to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'num_rows': df.shape[0] 
    }
    # Generate descriptive statistics and transpose to match the structure
    stats = df.describe(include='all').T  

    return metadata, stats

def execute_validation_code(code_snippets, feature_set):
    """
    Safely executes the generated validation code snippets on the given feature set.
    Assumes that the snippets will produce validation outcomes.
    """
    local_scope = {'feature_set': feature_set}
    try:
        # Execute the code snippets
        exec(code_snippets, globals(), local_scope)

        # Retrieve validation outcomes from the executed code
        validation_outcomes = local_scope.get('validation_outcomes', 'No validation outcomes defined.')
        return validation_outcomes
    except Exception as e:
        print(f"Error executing validation code: {e}")
        return None

def save_validation_outcomes(set_name, validation_outcomes):
    """Saves the validation outcomes to a CSV file."""
    results_filename = os.path.join('results', f'validation_outcomes_{set_name}.csv')
    
    with open(results_filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['Outcome Description', 'Value'])
        for key, value in validation_outcomes.items():
            writer.writerow([key, value])

    print(f"Validation outcomes saved to {results_filename}")


def save_feature_metrics(set_name, feature_set):
    """Saves basic feature set metrics (e.g., missing values, number of rows) to a CSV file."""
    metrics_filename = os.path.join('results', f'feature_metrics_{set_name}.csv')
    
    metrics = {
        'num_rows': feature_set.shape[0],
        'num_columns': feature_set.shape[1],
        'missing_values': feature_set.isnull().sum().sum(),
        # You can add more feature-specific metrics here if needed
    }
    
    with open(metrics_filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])

    print(f"Feature metrics saved to {metrics_filename}")


def main():
    setup_logging()

    # Load the feature sets and baseline data
    baseline_data, data_set_2, data_set_3, dataset_name = load_new_feature_set()

    # Initialize the Generative Model
    
    model_choice = "gemini"  # Choose the model (gemini or mistral)
    generative_model = GenerativeModel(model_choice)

    # Initialize the RAG validation, CodeSnippetGenerator, and Knowledge Base
    #knowledge_base = KnowledgeBaseInterfaceDB(db_path='knowledge_base.db')
    #rag_validator = RAGValidationDB(knowledge_base, generative_model)

    knowledge_base = KnowledgeBaseInterface(dimension=512) 
    rag_validator = RAGValidation(knowledge_base)

    # Initialize the DriftManager, which handles both drift detection and rule updating
    drift_manager = DriftManager(baseline_data, rag_validator)

    # Initialize CodeSnippetGenerator with the GenerativeModel
    code_generator = CodeSnippetGenerator(model=generative_model)

    # Initialize the FeatureSetValidator with Rule Generator and Code Generator
    rule_generator = ValidationRuleGenerator(generative_model, knowledge_base)
    validator = FeatureSetValidator(rule_generator, code_generator)

    # Validate the baseline dataset and generate rules
    metadata, stats = generate_metadata_and_stats(baseline_data)
    validator.validate_feature_set(metadata, stats, dataset_name, baseline_data)

    # Validate each additional dataset, check for drift, and update rules if necessary
    for set_number, dataset in enumerate([data_set_2, data_set_3], start=2):
        print(f"Validating dataset part {set_number}...")

        # Check for drift and update rules if necessary
        updated_rules, drifted_columns = drift_manager.check_drift_and_update_rules(dataset, f"{dataset_name}_part_{set_number}")

        # If drift was detected, generate new validation rules and validate dataset
        if updated_rules:
            print(f"Drift detected in dataset part {set_number}. Regenerating validation rules.")
            metadata, stats = generate_metadata_and_stats(dataset)
            validator.validate_feature_set(metadata, stats, f"{dataset_name}_part_{set_number}", dataset)
        else:
            print(f"No drift detected in dataset part {set_number}. Proceeding with validation.")
            # No drift, so use the existing rules
            metadata, stats = generate_metadata_and_stats(dataset)
            validator.validate_feature_set(metadata, stats, f"{dataset_name}_part_{set_number}", dataset)
if __name__ == "__main__":
    main()

    

