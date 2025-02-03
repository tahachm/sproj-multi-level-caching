import os
import csv
import json
import openai
import time
from pymilvus import MilvusClient, model
from groq import Groq

# ==== Configuration ====
DATA_FILE = "questions.csv"
MILVUS_DB_PATH = "queries.db"
COLLECTION_NAME = "allQuestions"
DIMENSION = 768
CSV_FILE = "milvusRes.csv"
JSON_FILE = "responses.json"

print("=== Starting the script ===")
print(f"Data file: {DATA_FILE}")
print(f"Database file: {MILVUS_DB_PATH}")
print(f"Collection: {COLLECTION_NAME}")
print(f"CSV output: {CSV_FILE}")
print(f"JSON output: {JSON_FILE}")

# Ensure response JSON exists or create empty
if not os.path.exists(JSON_FILE):
    print(f"{JSON_FILE} does not exist. Creating an empty JSON.")
    with open(JSON_FILE, "w") as f:
        json.dump({}, f)

# Load existing responses
print("Loading existing responses from JSON...")
with open(JSON_FILE, "r") as f:
    responses_cache = json.load(f)
print(f"Loaded {len(responses_cache)} existing responses.")

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

# Open CSV for appending milvus results
csv_exists = os.path.exists(CSV_FILE)
csvfile = open(CSV_FILE, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csvfile)
if not csv_exists:
    csv_writer.writerow(["query_id", "cache_id", "query_question", "cache_question", "distance"])
print("CSV file ready for writing.")

print("Setting up OpenAI client for Llama 1B...")
openai.api_key = "027d952d-f652-409d-9a03-07d0eb613db0"
openai.api_base = "https://api.sambanova.ai/v1"
if not openai.api_key:
    print("No API key set for Llama 1B. Llama 1B queries may fail.")
else:
    print("Llama 1B client ready.")

print("Setting up GROQ client for Llama 70B...")
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("GROQ_API_KEY not found in environment. Llama 70B queries may fail.")
groq_client = Groq(api_key=groq_api_key)
print("Llama 70B client ready.")

def insert_question_into_milvus(question_text, qid):
    qid = int(qid)
    print(f"Inserting question_id {qid} into Milvus: {question_text}")
    vec = embedding_fn.encode_queries([question_text])[0]
    data = [{"id": qid, "vector": vec, "text": question_text}]
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
        output_fields=["text"]
    )
    
    if res and len(res[0]) > 0:
        for hit in res[0]:
            if exclude_id is not None and hit["id"] == exclude_id:
                # Skip this hit if it's the same ID as the current question
                print(f"Skipping result ID={hit['id']} as it matches exclude_id={exclude_id}")
                continue
            
            # Found a valid match
            print(f"Found match: ID={hit['id']}, Text={hit['entity']['text']}, Distance={hit['distance']}")
            return hit["id"], hit["entity"]["text"], hit["distance"]
    
    # If no valid match is found
    print("No suitable match found in Milvus.")
    return None, None, None


def get_cache_response_for_matched_qid(matched_qid):
    if matched_qid is None:
        return ""
    matched_qid_str = str(matched_qid)
    if matched_qid_str not in responses_cache:
        return ""
    entry = responses_cache[matched_qid_str]
    extracted = entry.get("extracted_response", -1)
    if extracted != -1 and extracted != "-1":
        return extracted
    if "70b_response" in entry and entry["70b_response"] != -1:
        return entry["70b_response"]
    return entry.get("1b_response", "")

def query_llama_1b(new_question, cache_question, cache_resp):
    prompt = (
        f"You are an assistant deciding if we can reuse a cached response for a new user query.\n"
        f"User's new question: {new_question}\n"
        f"Cached question: {cache_question}\n"
        f"Cached response: {cache_resp}\n\n"
        "Step 1: Rate similarity between the new_question and cache_question on a scale of 0 to 1.\n"
        "If they are identical or nearly identical in meaning, similarity should be 1.\n"
        "If somewhat related but not identical, choose a value between 0 and 1 that reflects how closely they match.\n\n"
        "Step 2: DECISION:\n"
        "- If you can reuse the cached response as is or with slight tweaks to answer the new query well, DECISION: possible.\n"
        "- If you think that the cached response can not be reused, even with slight tweak, DECISION: not possible.\n\n"
        "If DECISION: possible and similarity > 0.7, provide the reused or tweaked response in 'Response for user:' line.\n"
        "If DECISION: not possible, set 'Response for user:' to -1.\n\n"
        "Finally, provide a Reason line explaining your thinking.\n\n"
        "Format:\n"
        "SIMILARITY: <value>\n"
        "Reason: <justification>\n"
        "Response for user: <tweaked response or -1>\n"
        "DECISION: possible or not possible\n"
    )

    print("Querying Llama 1B with prompt:")
    print(prompt)

    response = openai.ChatCompletion.create(
        model="Meta-Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who follows instructions carefully."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        top_p=0.1
    )

    r = response.choices[0].message.content.strip()
    print("Llama 1B response:")
    print(r)
    return r

def query_llama_70b(question):
    print("Querying Llama 70B with question:")
    print(question)
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": "Please limit your response for the following to a maximum of 3 paragraphs: " + question}],
        model="llama-3.1-70b-versatile",
    )
    r = chat_completion.choices[0].message.content.strip()
    print("Llama 70B response:")
    print(r)
    return r

def parse_1b_response(response_text):
    # Extract fields
    lines = response_text.split("\n")
    similarity = 0.0
    decision = "not possible"
    response_for_user = ""
    reason = ""

    for line in lines:
        lower_line = line.strip().lower()
        if lower_line.startswith("similarity:"):
            try:
                similarity_str = line.split(":", 1)[1].strip()
                similarity = float(similarity_str)
            except:
                similarity = 0.0
        elif lower_line.startswith("decision:"):
            dec_str = line.split(":", 1)[1].strip().lower()
            if "possible" in dec_str:
                decision = "possible"
            else:
                decision = "not possible"
        elif lower_line.startswith("response for user:"):
            response_for_user = line.split(":", 1)[1].strip()
        elif lower_line.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return similarity, decision, response_for_user, reason

print("Starting to process data from questions.csv...")

killSwitch = 0

with open(DATA_FILE, "r", encoding="utf-8") as csvf:
    reader = csv.DictReader(csvf)
    for row in reader:
        time.sleep(3)
        killSwitch += 1
        if killSwitch > 2000:
            break

        # ---- Process question 1 ----
        qid1 = row["qid1"]
        current_qid = int(qid1)
        new_question = row["question1"].strip('"')
        print(f"Processing question_id {current_qid}: {new_question}")

        matched_qid, matched_q_text, distance = search_milvus(new_question, exclude_id=current_qid)
        if matched_qid is not None and matched_q_text is not None:
            csv_writer.writerow([current_qid, matched_qid, new_question, matched_q_text, distance])
            csvfile.flush()  # Flush after every write
            cache_resp = get_cache_response_for_matched_qid(matched_qid)
            llama_1b_raw = query_llama_1b(new_question, matched_q_text, cache_resp)
        else:
            matched_qid = -1
            matched_q_text = ""
            distance = -1
            csv_writer.writerow([current_qid, matched_qid, new_question, matched_q_text, distance])
            csvfile.flush()  # Flush after every write
            llama_1b_raw = query_llama_1b(new_question, "", "")

        similarity, decision, response_for_user, reason = parse_1b_response(llama_1b_raw)
        if decision == "possible" and response_for_user != "-1" and response_for_user.strip():
            extracted_response = response_for_user
            responses_cache[str(current_qid)] = {
                "1b_response": llama_1b_raw,
                "70b_response": -1,
                "extracted_response": extracted_response,
                "1b_distance": similarity
            }
        else:
            extracted_response = -1
            llama_70b_answer = query_llama_70b(new_question)
            responses_cache[str(current_qid)] = {
                "1b_response": llama_1b_raw,
                "70b_response": llama_70b_answer,
                "extracted_response": extracted_response,
                "1b_distance": similarity
            }

        # Write JSON after each processed question
        with open(JSON_FILE, "w") as f:
            json.dump(responses_cache, f, indent=2)

        insert_question_into_milvus(new_question, current_qid)
        print(f"Finished processing question_id {current_qid}")


        # ---- Process question 2 ----
        qid2 = row["qid2"]
        current_qid = int(qid2)
        new_question = row["question2"].strip('"')
        print(f"Processing question_id {current_qid}: {new_question}")

        matched_qid, matched_q_text, distance = search_milvus(new_question, exclude_id=current_qid)
        if matched_qid is not None and matched_q_text is not None:
            csv_writer.writerow([current_qid, matched_qid, new_question, matched_q_text, distance])
            csvfile.flush()  # Flush after every write
            cache_resp = get_cache_response_for_matched_qid(matched_qid)
            llama_1b_raw = query_llama_1b(new_question, matched_q_text, cache_resp)
        else:
            matched_qid = -1
            matched_q_text = ""
            distance = -1
            csv_writer.writerow([current_qid, matched_qid, new_question, matched_q_text, distance])
            csvfile.flush()  # Flush after every write
            llama_1b_raw = query_llama_1b(new_question, "", "")

        similarity, decision, response_for_user, reason = parse_1b_response(llama_1b_raw)

        if decision == "possible" and response_for_user != "-1":
            extracted_response = response_for_user
        else:
            extracted_response = -1

        if response_for_user == "-1":
            llama_70b_answer = query_llama_70b(new_question)
            responses_cache[str(current_qid)] = {
                "1b_response": llama_1b_raw,
                "70b_response": llama_70b_answer,
                "extracted_response": extracted_response,
                "1b_distance": similarity
            }
        else:
            responses_cache[str(current_qid)] = {
                "1b_response": llama_1b_raw,
                "70b_response": -1,
                "extracted_response": extracted_response,
                "1b_distance": similarity
            }

        # Write JSON after each processed question
        with open(JSON_FILE, "w") as f:
            json.dump(responses_cache, f, indent=2)

        insert_question_into_milvus(new_question, current_qid)
        print(f"Finished processing question_id {current_qid}")

print("Finished reading questions.csv.")

# Close CSV file
csvfile.close()
print("CSV file closed.")

print("=== Processing complete ===")
