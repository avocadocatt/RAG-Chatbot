from pinecone import Pinecone, ServerlessSpec
from src.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_VECTOR_DIMENSION
import time

class PineconeVectorStore:
    def __init__(self, api_key, index_name, dimension, metric='cosine'):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        self.pc = Pinecone(api_key=self.api_key)
        
        self.index = None
        self._connect_or_create_index()

    def _connect_or_create_index(self):
        indexs = self.pc.list_indexes()
        if self.index_name not in [index.name for index in indexs]:
            print(f"Index '{self.index_name}' not found. Creating new serverless index...")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                while not self.pc.describe_index(self.index_name).status['ready']:
                    print("Waiting for index to be ready...")
                    time.sleep(5)
                print(f"Index '{self.index_name}' created successfully.")
            except Exception as e:
                print(f"Error creating Pinecone index: {e}")
                if "already exists" in str(e).lower():
                     print(f"Index '{self.index_name}' may already exist with different configuration or is in a transient state.")
                raise
        else:
            print(f"Connecting to existing index '{self.index_name}'...")

        self.index = self.pc.Index(self.index_name)
        print(f"Successfully connected to index '{self.index_name}'.")
        print(self.index.describe_index_stats())

    def upsert_vectors(self, vectors_with_metadata, batch_size=100):
        """
        Upserts vectors with metadata to Pinecone.
        vectors_with_metadata: list of tuples or dicts, e.g.,
                               [('id1', [0.1, 0.2, ...], {'text': 'some text', 'source': 'doc1'}), ...]
                               or [{'id': 'id1', 'values': [0.1,...], 'metadata': {'text': ..., 'source': ...}}]
        """
        if not self.index:
            print("Pinecone index not initialized.")
            return None
        if not vectors_with_metadata:
            print("No vectors to upsert.")
            return

        formatted_vectors = []
        for item in vectors_with_metadata:
            if isinstance(item, dict) and 'id' in item and 'values' in item:
                formatted_vectors.append(item)
            elif isinstance(item, tuple) and len(item) >= 2: # (id, values, metadata_opt)
                # make sure the tuple has at least 2 elements
                # and at most 3 elements
                if len(item) == 2: # (id, values)
                    formatted_vectors.append({'id': item[0], 'values': item[1]})
                elif len(item) == 3: # (id, values, metadata)
                    formatted_vectors.append({'id': item[0], 'values': item[1], 'metadata': item[2]})
                else:
                    print(f"Skipping malformed tuple for upsert: {item}")
                    continue
            else:
                if 'id' in item and 'embedding' in item and 'text' in item and 'metadata' in item:
                    formatted_vectors.append({
                        'id': item['id'],
                        'values': item['embedding'],
                        'metadata': {**item['metadata'], 'text_chunk': item['text']}
                    })
                else:
                    print(f"Skipping malformed item for upsert: {item}")
                    continue
        
        if not formatted_vectors:
            print("No valid formatted vectors to upsert.")
            return

        upserted_count = 0
        for i in range(0, len(formatted_vectors), batch_size):
            batch = formatted_vectors[i:i + batch_size]
            try:
                print(f"Upserting batch of {len(batch)} vectors...")
                upsert_response = self.index.upsert(vectors=batch)
                upserted_count += upsert_response.upserted_count
                print(f"Batch upserted. Total so far: {upserted_count}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {e}")
        print(f"Total vectors upserted to '{self.index_name}': {upserted_count}")
        return upserted_count

    def query_vectors(self, query_vector, top_k=5, filter_criteria=None):
        if not self.index:
            print("Pinecone index not initialized.")
            return None
        if query_vector is None:
            print("Query vector is None.")
            return None

        try:
            query_results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_criteria
            )
            return query_results.get('matches', [])
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

    def delete_index(self):
        if self.index_name in self.pc.list_indexes().names:
            print(f"Deleting index '{self.index_name}'...")
            self.pc.delete_index(self.index_name)
            time.sleep(5) 
            print(f"Index '{self.index_name}' deleted.")
        else:
            print(f"Index '{self.index_name}' not found, cannot delete.")


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    pinecone_api_key_from_env = PINECONE_API_KEY
    
    if not pinecone_api_key_from_env:
        print("PINECONE_API_KEY not found in .env file.")
    else:
        print(f"Attempting to connect with Index: {PINECONE_INDEX_NAME}, Dimension: {PINECONE_VECTOR_DIMENSION}")
        vector_db = PineconeVectorStore(
            api_key=pinecone_api_key_from_env,
            index_name=PINECONE_INDEX_NAME,
            dimension=PINECONE_VECTOR_DIMENSION 
        )