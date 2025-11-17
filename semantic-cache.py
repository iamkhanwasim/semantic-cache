import numpy as np
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, threshold=0.85):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.conn = sqlite3.connect(':memory:')
        self.threshold = threshold
        self.setup_db()

    def setup_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY,
                embedding BLOB,
                response TEXT,
                hits INTEGER DEFAULT 0
            )
        ''')

    def close(self):
        self.conn.close()

    def get(self, prompt):
        try:
            prompt_embeddings = self.encoder.encode([prompt], convert_to_tensor=True)

            for row in self.conn.execute('SELECT * FROM cache'):
                cached_embedings = pickle.loads(row[1])

                # Compute cosine similarity
                cosine_similarity = np.dot(prompt_embeddings, cached_embedings) / (
                    np.linalg.norm(prompt_embeddings) * np.linalg.norm(cached_embedings))
                
                if cosine_similarity >= self.threshold:
                    self.conn.execute('UPDATE cache SET hits=hits+1 WHERE id=?', row[0])
                    self.conn.commit()
                    return row[2]  # return cached response
                
            return None
        except Exception as e:
            print(f"Error during cache retrieval: {e}")
            return None
    
    def set (self, prompt, response):
        try:
            prompt_embeddings = self.encoder.encode(prompt, convert_to_tensor=True)
            self.conn.execute('Insert INTO cache (embedding, response) VALUES (?, ?)', pickle.dumps(prompt_embeddings), response)
            self.conn.commit()
        except Exception as e:
            print(f"Error during cache storage: {e}")

    

