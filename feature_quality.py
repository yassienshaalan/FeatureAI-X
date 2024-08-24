import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
from langdetect import detect
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import OpenAI

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam

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
            print("template prompt for feature validation")
            print(template)
        )
        llm = OpenAI(openai_api_key='your-api-key-here')
        chain = LLMChain(prompt_template=template, llm=llm)
        rules = chain.run({"metadata": metadata, "stats": stats})
        return rules

class FeatureSetValidator:
    def __init__(self):
        self.drift_detector = DataDriftDetector()
        self.rule_generator = ValidationRuleGenerator()

    def validate_feature_set(self, feature_set_name, df, baseline_df):
        metadata = {
            "columns": df.columns.tolist(),
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "missing_values": df.isnull().sum().sum(),
            "unique_counts": df.nunique(),
            "data_types": df.dtypes
        }
        stats = self.analyze_feature_set(df)
        drift_issues = self.drift_detector.calculate_drift(baseline_df, df)
        rules = self.rule_generator.generate_validation_rules_with_langchain(metadata, stats)
        print(f"Generated Validation Rules for {feature_set_name}:\n", rules)
        issues = self.dynamic_custom_rules(df)
        issues.extend(drift_issues)
        if issues:
            print(f"Data Quality Issues Found in {feature_set_name}:")
            for issue in issues:
                print(f"- {issue}")
            self.write_feature_quality_to_table(feature_set_name, issues)
        else:
            print(f"No data quality issues found in {feature_set_name}.")
            self.write_feature_quality_to_table(feature_set_name, ["No issues found"])

    def analyze_feature_set(self, df):
        return df.describe(include='all').T

    def dynamic_custom_rules(self, df):
        custom_issues = []
        if 'age' in df.columns:
            max_age_threshold = 100
            if df['age'].max() > max_age_threshold:
                custom_issues.append(f"'age' should not exceed {max_age_threshold}. Found {df['age'].max()}.")
        if 'annual_premium' in df.columns:
            if df['annual_premium'].max() > 50000:
                custom_issues.append(f"'annual_premium' should not exceed 50000. Found {df['annual_premium'].max()}.")
        if 'comments' in df.columns:
            for idx, text in df['comments'].iteritems():
                if not text or len(text.strip()) == 0:
                    custom_issues.append(f"Row {idx} in 'comments' is empty.")
                else:
                    detected_lang = detect(text)
                    if detected_lang != 'en':
                        custom_issues.append(f"Row {idx} in 'comments' is not in English (detected: {detected_lang}).")
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
        return df

    def load_baseline_feature_set(self):
        return self.load_new_feature_set(version='v1')  




class GenAISyntheticFeatureGenerator:
    def __init__(self, stats, metadata, text_model_name='gpt2'):
        self.stats = stats
        self.metadata = metadata
        
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


# Usage Example
print("Loading baseline and new feature sets...")
feature_store_manager = FeatureStoreManager()
print("Loading baseline df")
baseline_df = feature_store_manager.load_baseline_feature_set()
print("Loading new df")
feature_df = feature_store_manager.load_new_feature_set()
print("Feature sets loaded successfully.")


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

# Splitting the dataset into multiple feature sets (for simulation purposes)
print("Splitting the dataset into multiple feature sets...")
feature_sets = {
    'set1': feature_df[['age', 'policy_tenure', 'annual_premium', 'comments']],
    'set2': feature_df[['vehicle_age', 'policy_number', 'age', 'comments']],
    'set3': feature_df[['annual_premium', 'policy_tenure', 'vehicle_age', 'comments']]
}
print("Feature sets created.")

# Define metadata and stats for each feature set
for set_name, feature_set in feature_sets.items():
    print(f"\nProcessing feature set '{set_name}'...")

    # Define metadata
