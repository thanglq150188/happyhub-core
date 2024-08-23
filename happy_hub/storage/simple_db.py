import re
from openai import OpenAI
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import xml.etree.ElementTree as ET


class SimpleVectorDB:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents):
        embeddings = self.model.encode(documents, convert_to_tensor=True)
        self.documents.extend(documents)
        self.embeddings.extend(embeddings.cpu().numpy()) # type: ignore

    def add_document(self, document):
        self.add_documents([document])

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().numpy() # type: ignore
        query_embedding = np.array(query_embedding)
        embeddings_array = np.array(self.embeddings)
        similarities = util.pytorch_cos_sim(torch.tensor(query_embedding), torch.tensor(embeddings_array))[0].numpy()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.documents, self.embeddings), f)

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.documents, self.embeddings = pickle.load(f)