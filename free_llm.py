import requests
from semantic_cache import SemanticCache

class FreeLLM:
    def __init__(self):
        self.cahe = SemanticCache(threshold=0.9)
        self.groq_key = ''

    def complete(self, prompt):
        cached = self.cahe.get(prompt)
        if cached:
            print("Cache hit - saved API call")
            return cached
        
        response = requests
