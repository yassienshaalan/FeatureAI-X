from typing import List, Optional, Tuple
import logging

class RAGValidation:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def validate(self, feature_set, set_name, metadata):
        # Check for existing validation rules using both set_name and metadata
        existing_rules = self.knowledge_base.retrieve(set_name, metadata)
        existing_code = self.knowledge_base.retrieve_code(set_name, metadata)

        if existing_rules:
            print("Using retrieved rules for validation.")
            # Generate code based on the existing rules or retrieved code
            validation_code = self._generate_validation_code(existing_rules, set_name, existing_code)
            return existing_rules, validation_code
        else:
            print("Generating new rules.")
            prompt = self.generate_prompt_from_metadata(feature_set, set_name)
            generated_rules = self.generator.generate(prompt)
            print("Generated Rules:", generated_rules)
            self.knowledge_base.store(generated_rules, set_name, metadata)

            # Generate and store validation code
            validation_code = self._generate_validation_code(generated_rules, set_name)
            self.knowledge_base.store_code(validation_code, set_name, metadata)

            return generated_rules, validation_code

    def _generate_validation_code(self, rules: str, set_name: str, existing_code: Optional[List[Tuple[str]]]) -> str:
        """
        Generates validation code based on the given rules and previously stored knowledge.
        """
        new_code = """
        def validate_feature_set(feature_set):
            validation_results = []
            metrics = {
                'total_features': len(feature_set.columns),
                'missing_values': {},
                'outliers': {}
            }
        """

        # If there's existing code, use it as a base
        if existing_code:
            existing_code_str = "\n".join(code[0] for code in existing_code if isinstance(code, tuple))  # Extract code part
            new_code += f"\n    # Utilizing previous code for validation logic\n    {existing_code_str.strip()}\n"
        else:
            new_code += """
            # Default validation logic
            # Check for missing values
            for column in feature_set.columns:
                missing_count = feature_set[column].isnull().sum()
                if missing_count > 0:
                    metrics['missing_values'][column] = missing_count
                    validation_results.append(f"Column '{{column}}' has '{{missing_count}}' missing values.")
            """

        # Finalize the code structure
        new_code += """
            # Log validation results and metrics
            validation_results.append(f"Validation Metrics: {metrics}")
            return validation_results
        """

        return new_code

    def generate_prompt_from_metadata(self, feature_set, set_name):
        # Generate the prompt based on the feature set and set name
        prompt = f"Generate a set of comprehensive validation rules for the feature set '{set_name}'.\n"
        prompt += "Here are the dataset statistics:\n"
        for column in feature_set.columns:
            prompt += f"- Column '{column}' with dtype '{feature_set[column].dtype}' and sample values: {feature_set[column].unique()[:5]}\n"
        return prompt