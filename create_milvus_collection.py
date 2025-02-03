from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

MILVUS_DB_PATH = "queries.db"
COLLECTION_NAME = "allQuestions"

print("Initializing Milvus Client...")
client = MilvusClient(MILVUS_DB_PATH)
print("Milvus Client initialized.")

# Define schema
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="question_text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="response_text", dtype=DataType.VARCHAR, max_length=4096),
    ],
    description="Collection for storing question-response pairs",
    enable_dynamic_field=True  # Allows adding new fields later
)

# Create collection
if not client.has_collection(collection_name=COLLECTION_NAME):
    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"✅ Collection {COLLECTION_NAME} created successfully!")
else:
    print(f"✅ Collection {COLLECTION_NAME} already exists!")
