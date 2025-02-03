from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, Collection

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

# Create collection if it does not exist
if not client.has_collection(collection_name=COLLECTION_NAME):
    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"✅ Collection {COLLECTION_NAME} created successfully!")
else:
    print(f"✅ Collection {COLLECTION_NAME} already exists!")

# ✅ Corrected Index Parameters
index_params = client.prepare_index_params()

# 4. Add indexes
# - For a vector field
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 1024}
)

# 6. Create indexes
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)


# Load the collection for searching
client.load_collection(collection_name=COLLECTION_NAME)
print(f"✅ Collection `{COLLECTION_NAME}` loaded into memory and ready for search!")
