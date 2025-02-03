import os
import csv
import json
from openai import OpenAI
import time
from pymilvus import MilvusClient, model
from groq import Groq

# ==== Configuration ====
DATA_FILE = "questions.csv"
MILVUS_DB_PATH = "queries.db"
COLLECTION_NAME = "allQuestions"
DIMENSION = 768
# CSV_FILE = "milvusRes.csv"
# JSON_FILE = "responses.json"
OUTPUT_JSON_FILE = "structured_results.json"

COSINE_SIMILARITY_THRESHOLD = 0.7

print("=== Starting the script ===")
print(f"Data file: {DATA_FILE}")
print(f"Database file: {MILVUS_DB_PATH}")
print(f"Collection: {COLLECTION_NAME}")
# print(f"CSV output: {CSV_FILE}")
# print(f"JSON output: {JSON_FILE}")

# # Ensure response JSON exists or create empty
# if not os.path.exists(JSON_FILE):
#     print(f"{JSON_FILE} does not exist. Creating an empty JSON.")
#     with open(JSON_FILE, "w") as f:
#         json.dump({}, f)

print("Initializing Milvus Client...")
client = MilvusClient(MILVUS_DB_PATH)
print("Milvus Client initialized.")

# Check if collection exists
if not client.has_collection(collection_name=COLLECTION_NAME):
    print(f"Collection {COLLECTION_NAME} does not exist in {MILVUS_DB_PATH}. Please create it before running this script.")
    exit(1)

print("Initializing embedding function...")
embedding_fn = model.DefaultEmbeddingFunction()
print("Embedding function initialized.")

# # Open CSV for appending milvus results
# csv_exists = os.path.exists(CSV_FILE)
# csvfile = open(CSV_FILE, "a", newline="", encoding="utf-8")
# csv_writer = csv.writer(csvfile)
# if not csv_exists:
#     csv_writer.writerow(["query_id", "cache_id", "query_question", "cache_question", "distance"])
# print("CSV file ready for writing.")

print("Setting up OpenAI client for Llama 1B...")
openai_client = OpenAI(
    api_key="027d952d-f652-409d-9a03-07d0eb613db0",  # This is the default and can be omitted
)
# openai.api_base = "https://api.sambanova.ai/v1"
if openai_client.api_key is None:
    print("No API key set for Llama 1B. Llama 1B queries may fail.")
else:
    print("Llama 1B client ready.")

print("Setting up GROQ client for Llama 70B...")
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("GROQ_API_KEY not found in environment. Llama 70B queries may fail.")
groq_client = Groq(api_key=groq_api_key)
print("Llama 70B client ready.")

def insert_question_into_milvus(question_text, qid, response_text):
    qid = int(qid)
    print(f"Inserting question_id {qid} into Milvus: {question_text}")
    vec = embedding_fn.encode_queries([question_text])[0]
    data = [{"id": qid, "vector": vec, "question_text": question_text, "response_text": response_text}]
    res = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Insert result: {res}")

def search_milvus(question_text, exclude_id=None):
    """
    Search Milvus for the most similar question to the input question_text.
    Exclude the current question ID (exclude_id) from the results.
    """
    print(f"Searching Milvus for similar question to: {question_text}")
    query_vec = embedding_fn.encode_queries([question_text])
    
    # Request top N results for fallback logic
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vec,
        limit=5,  # Request more results to ensure valid matches
        output_fields=["question_text", "response_text"]
    )
    
    if res and len(res[0]) > 0:
        for hit in res[0]:
            if exclude_id is not None and hit["id"] == exclude_id:
                # Skip this hit if it's the same ID as the current question
                print(f"Skipping result ID={hit['id']} as it matches exclude_id={exclude_id}")
                continue
            
            # Found a valid match
            print(f"Found match: ID={hit['id']}, Text={hit['entity']['question_text']}, Distance={hit['distance']}")
            return hit["id"], hit["entity"]["question_text"], hit["distance"], hit["entity"]["response_text"]
    
    # If no valid match is found
    print("No suitable match found in Milvus.")
    return None, None, 0, None

def get_cache_response_for_matched_qid(matched_qid):
    """
    Fetches the cached response from Milvus instead of JSON.
    """
    if matched_qid is None:
        return None

    # Query Milvus to retrieve the stored response text
    query_res = client.query(
        collection_name=COLLECTION_NAME,
        expr=f"id == {matched_qid}",
        output_fields=["response_text"]
    )
    if not query_res or len(query_res) == 0:
        raise ValueError(f"No cached response found for matched_qid={matched_qid}.")

    return query_res[0]["response_text"]  # Return the stored response from Milvus

def query_llama_1b(new_question, cache_question, cache_resp):
    prompt = (
        f"You are an assistant tasked with refining and adjusting a response to align with a slightly modified question.\n"
        f"Below is a cached response that is similar but perhaps not perfectly suited to the new question.\n"
        f"Your goal is to make minimal yet effective modifications to ensure the response is relevant to the NEW question, and answers it accurately while preserving fluency, correctness, and completeness.\n\n"
        f"Also, try to maintain the original response's length. Do not exceed 3 sentneces."
        
        f"New Question: {new_question}\n"
        f"Cached Question: {cache_question}\n"
        f"Cached Response: {cache_resp}\n"
        
        f"Instructions:\n"
        f"1. Identify the key differences between the New Question and the Cached Question.\n"
        f"2. Modify the Cached Response only as needed to fully address the New Question.\n"
        f"3. Ensure clarity and coherence, keeping the response concise and informative.\n"
        f"4. Do not introduce unnecessary changes—only tweak for relevance.\n\n"
        
        f"Example:\n"
        f"- New Question: \"What is the step-by-step guide to invest in the share market in India?\"\n"
        f"- Cached Question: \"What is the step-by-step guide to invest in the share market?\"\n"
        f"- Cached Response: \"[Step-by-step guide for general share market investing]\"\n"
        f"- Modified Response: \"[Step-by-step guide with India-specific regulations, brokers, and taxation details]\"\n\n"
        
        f"Now, generate the revised response:\n"
        f"Reponse format:\n"
        f"Response for user: <tweaked response>\n"
    )

    print("Querying Llama 1B:")

    # response = openai.ChatCompletion.create(
    #     model="Meta-Llama-3.2-1B-Instruct",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant who follows instructions carefully."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.1,
    #     top_p=0.1
    # )

    # response = openai_client.chat.completions.create(
    #     model="Meta-Llama-3.2-1B-Instruct",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant who follows instructions carefully."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.1,
    #     top_p=0.1,
    # )

    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
    )

    r = response.choices[0].message.content.strip()
    print("Llama 1B response:")
    print(r)
    return r

def query_llama_70b(question):
    print("Querying Llama 70B with question:")
    print(question)
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": "Please limit your response for the following to a maximum of 3 sentences: " + question}],
        model="llama-3.3-70b-versatile",
    )
    r = chat_completion.choices[0].message.content.strip()
    print("Llama 70B response:")
    print(r)
    return r

def parse_1b_response(response_text):
    # Extract fields
    lines = response_text.split("\n")
    response_for_user = ""
    
    for line in lines:
        lower_line = line.strip().lower()
        if lower_line.startswith("response for user:"):
            response_for_user = line.split(":", 1)[1].strip()
       
    return response_for_user

def process_question(qid, new_question):
    """
    Handles processing of a single question.
    - Searches Milvus for a cached response.
    - Uses a small model to refine (cache hit) or a large model to generate (cache miss).
    - Stores and retrieves responses from Milvus instead of a JSON file.
    - Returns a structured dictionary for logging results.
    """
    current_qid = int(qid)
    print(f"Processing question_id {current_qid}: {new_question}")

    matched_qid, matched_q_text, distance, matched_q_response = search_milvus(new_question, exclude_id=current_qid)
    cache_hit = distance >= COSINE_SIMILARITY_THRESHOLD

    llama_1b_response = ""
    llama_70b_response = ""
    response_source = ""

    if cache_hit:
        # Query smaller model to tweak the response
        llama_1b_response = parse_1b_response(query_llama_1b(new_question, matched_q_text, matched_q_response))
        response_source = "cache_hit (Llama 1B)"
    else:
        # Query large model to generate response from scratch
        llama_70b_response = query_llama_70b(new_question)
        response_source = "cache_miss (Llama 70B)"

    final_response = llama_1b_response if llama_1b_response else llama_70b_response

    # Store the response **directly in Milvus**
    insert_question_into_milvus(new_question, current_qid, final_response)

    print(f"Finished processing question_id {current_qid}")

    # Return structured data for JSON output
    return {
        "qid": current_qid,
        "question": new_question,
        "matched_qid": matched_qid,
        "match_cosine_similarity": distance,
        "response": final_response,
        "generated_by": response_source,
    }


print("Starting to process data from questions.csv...")

kill_switch = 0
KILL_LIMIT = 5
structured_responses = []
with open(DATA_FILE, "r", encoding="utf-8") as csvf:
    reader = csv.DictReader(csvf)
    
    for row in reader:
        time.sleep(3)
        kill_switch += 1
        if kill_switch > KILL_LIMIT:
            break

        qid1 = row["qid1"]
        qid2 = row["qid2"]
        question1 = row["question1"].strip('"')
        question2 = row["question2"].strip('"')

        response1 = process_question(qid1, question1)
        response2 = process_question(qid2, question2)

        # Store results in the structured_responses list
        structured_responses.append({
            "qid1": qid1,
            "question1": question1,
            "qid2": qid2,
            "question2": question2,
            # "similarity_score": response2["match_cosine_similarity"],  # Use distance from Milvus
            "responses": [response1, response2]
        })

print("Finished processing questions.csv.")

with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(structured_responses, f, indent=2)

print(f"Saved structured responses to {OUTPUT_JSON_FILE}.")


print("=== Processing complete ===")
