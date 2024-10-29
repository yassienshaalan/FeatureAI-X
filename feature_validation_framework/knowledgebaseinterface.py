import numpy as np 
from vectordatabase import *
import tensorflow_hub as hub
import json
from generativemodel import * 
import logging

class KnowledgeBaseInterface:
    def __init__(self, dimension: int, model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        self.vector_db = VectorDatabase(dimension)
        self.model = hub.load(model_url)  # Load the Universal Sentence Encoder model

    def store(self, rules: str, set_name: str, metadata: dict):
        """
        Stores generated validation rules to the vector database.
        """
        rule_vector = self._generate_vector(rules)
        self.vector_db.add(rule_vector, f"{set_name}|{json.dumps(metadata)}")  # Store rules with metadata
        print(f"Validation rules for '{set_name}' stored in the vector database.")

    def store_code(self, code: str, set_name: str, metadata: dict):
        """
        Stores generated validation code to the vector database with metadata.
        """
        code_vector = self._generate_vector(code)
        self.vector_db.add(code_vector, f"{set_name}|{json.dumps(metadata)}")  # Store code with metadata
        print(f"Validation code for '{set_name}' stored in the vector database.")

    def retrieve(self, set_name: str, metadata: dict):
        """
        Retrieves validation rules for the given set name and metadata.
        """
        query_vector = self._generate_vector(f"{set_name}|{json.dumps(metadata)}")
        return self.vector_db.search(query_vector)

    def retrieve_code(self, set_name: str, metadata: dict):
        """
        Retrieves validation code for the given set name and metadata.
        """
        query_vector = self._generate_vector(f"{set_name}|{json.dumps(metadata)}")
        return self.vector_db.search(query_vector)

    def _generate_vector(self, text: str) -> np.ndarray:
        """
        Generates a vector representation of the input text using a transformer model.
        """
        embeddings = self.model([text])  # The model expects a list of strings
        vector = embeddings.numpy()[0]  # Convert to numpy array and get the first (and only) vector
        return vector