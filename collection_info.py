from pymilvus import MilvusClient

MILVUS_DB_PATH = "queries.db"
COLLECTION_NAME = "allQuestions"

client = MilvusClient(MILVUS_DB_PATH)
collection_info = client.describe_collection(collection_name=COLLECTION_NAME)
indexes_info = client.list_indexes(collection_name=COLLECTION_NAME)

print(f"Collection {COLLECTION_NAME} Info:\n{collection_info}")
print(f"\nIndexes for collection {COLLECTION_NAME}:\n{indexes_info}")