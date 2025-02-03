from pymilvus import MilvusClient

MILVUS_DB_PATH = "queries.db"
COLLECTION_NAME = "allQuestions"

# Initialize Milvus Client
print("Initializing Milvus Client...")
client = MilvusClient(MILVUS_DB_PATH)
print("Milvus Client initialized.")

# ✅ Check if collection exists before deleting
if client.has_collection(collection_name=COLLECTION_NAME):
    print(f"⚠️ Deleting collection: {COLLECTION_NAME}...")
    client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"✅ Collection {COLLECTION_NAME} deleted successfully!")
else:
    print(f"❌ Collection {COLLECTION_NAME} does not exist!")
