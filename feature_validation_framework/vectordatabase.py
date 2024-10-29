from typing import List, Optional, Tuple
import numpy as np 
import faiss
import logging
class VectorDatabase:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance for vector similarity
        self.data = []  # To keep track of the original data (rules or code)

    def add(self, vector: np.ndarray, text: str):
        """
        Adds a vector and associated text (rules or code) to the database.
        """
        vector = np.array(vector, dtype=np.float32)  # Ensure the vector is in the correct format
        self.index.add(np.array([vector]))  # Add the vector to the FAISS index
        self.data.append(text)  # Store the associated text

    def search(self, query_vector: np.ndarray, k: int = 5) -> Optional[List[Tuple[str, float]]]:
        """
        Searches for the nearest vectors to the query vector.
        Returns the associated texts (rules or code) of the nearest neighbors.
        """
        query_vector = np.array(query_vector, dtype=np.float32)  # Ensure the query vector is in the correct format
        distances, indices = self.index.search(np.array([query_vector]), k)  # Search the index

        results = []
        for j, i in enumerate(indices[0]):
            if 0 <= i < len(self.data):  # Check if index is valid
                results.append((self.data[i], distances[0][j]))
        return results if results else None  # Return None if no valid results found