import google.generativeai as genai
import mlflow
from typing import List, Optional, Tuple
import os 
import logging
class GenerativeModel:
    """
    A class to interface with different AI generative models based on the provided API key and model choice.
    """
    def __init__(self, model_choice):

        self.api_key = os.getenv('GOOGLE_API_KEY')
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