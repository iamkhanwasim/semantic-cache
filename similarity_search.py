import faiss
import mysql.connector
from dataclasses import dataclass
from typing import Optional
import hashlib
import pickle

@dataclass
class CacheEntry:
    id: int
    query: str
    response: str
    embedding: np.ndarray
    model_source: str

class SemanticCache:
    def __init__(
        self,
        embedder: SemanticCacheEmbedder,
        db_config: dict,
        similarity_threshold: float = 0.92
    ):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.db_config = db_config

        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(embedder.embedding_dim)
        self.id_map: list[int] = []  # Maps FAISS idx -> MySQL id

        # Load existing embeddings from MySQL into FAISS
        self._load_index_from_db()

    def _get_connection(self):
        return mysql.connector.connect(**self.db_config)

    def _load_index_from_db(self):
        """Load all embeddings from MySQL into FAISS on startup."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM semantic_cache")

        for row_id, emb_blob in cursor.fetchall():
            embedding = pickle.loads(emb_blob)
            self.index.add(embedding.reshape(1, -1))
            self.id_map.append(row_id)

        cursor.close()
        conn.close()

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """Two-tier lookup: exact hash in MySQL, then semantic via FAISS."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)

        # Tier 1: Exact match via indexed hash
        query_hash = self._hash_query(query)
        cursor.execute(
            "SELECT * FROM semantic_cache WHERE query_hash = %s",
            (query_hash,)
        )
        row = cursor.fetchone()

        if row:
            self._increment_hit_count(cursor, row['id'])
            conn.commit()
            entry = self._row_to_entry(row)
            cursor.close(); conn.close()
            return entry

        # Tier 2: Semantic search via FAISS
        if self.index.ntotal == 0:
            cursor.close(); conn.close()
            return None

        query_embedding = self.embedder.embed(query).reshape(1, -1)
        similarities, indices = self.index.search(query_embedding, k=5)

        if similarities[0][0] >= self.threshold:
            mysql_id = self.id_map[indices[0][0]]
            cursor.execute(
                "SELECT * FROM semantic_cache WHERE id = %s",
                (mysql_id,)
            )
            row = cursor.fetchone()
            if row:
                self._increment_hit_count(cursor, row['id'])
                conn.commit()
                entry = self._row_to_entry(row)
                cursor.close(); conn.close()
                return entry

        cursor.close(); conn.close()
        return None

    def store(self, query: str, response: str, model_source: str) -> int:
        """Store new cache entry in MySQL and update FAISS index."""
        embedding = self.embedder.embed(query)
        query_hash = self._hash_query(query)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO semantic_cache
               (query_hash, query, response, embedding, model_source)
               VALUES (%s, %s, %s, %s, %s)""",
            (query_hash, query, response,
             pickle.dumps(embedding), model_source)
        )
        conn.commit()
        new_id = cursor.lastrowid

        # Update FAISS index
        self.index.add(embedding.reshape(1, -1))
        self.id_map.append(new_id)

        cursor.close(); conn.close()
        return new_id

    def _increment_hit_count(self, cursor, entry_id: int):
        cursor.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = %s",
            (entry_id,)
        )

    def _row_to_entry(self, row: dict) -> CacheEntry:
        return CacheEntry(
            id=row['id'],
            query=row['query'],
            response=row['response'],
            embedding=pickle.loads(row['embedding']),
            model_source=row['model_source']
        )
