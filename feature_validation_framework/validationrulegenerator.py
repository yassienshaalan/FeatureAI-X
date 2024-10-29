import pandas as pd 

class ValidationRuleGenerator:
    def __init__(self, generative_model, kb_interface):
        self.generative_model = generative_model
        self.kb_interface = kb_interface

    def generate_validation_rules(self, metadata, stats, set_name):
        existing_rules = self.kb_interface.retrieve(set_name, metadata)
        if existing_rules:
            print(f"Using existing validation rules for '{set_name}'.")
            return existing_rules
        print("Generating new rules as no existing was found")
        prompt = self.generate_prompt_from_metadata(metadata, stats)
        rules = self.generative_model.generate(prompt)
        self.kb_interface.store(set_name, rules,metadata)
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
            elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype):
                prompt += f"- '{column}' should only contain valid categories based on historical data.\n"
                if metadata['unique_counts'][column] == metadata['num_rows']:
                    prompt += f"- '{column}' should be unique.\n"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                prompt += f"- '{column}' should have valid and sequential dates.\n"
            if pd.api.types.is_string_dtype(dtype):
                prompt += f"- '{column}' should be checked for completeness, readability, and keyword presence.\n"
        prompt += "Ensure that the rules cover completeness, consistency, uniqueness, range checks, and any potential outliers."
        
        return prompt
    
    '''
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
    '''
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