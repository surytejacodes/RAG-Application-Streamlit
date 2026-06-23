import os, json, boto3
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))

def embed(text):
    response = client.invoke_model(
        modelId=os.getenv("BEDROCK_EMBEDDING_MODEL"),
        body=json.dumps({"inputText": text})
    )
    body = json.loads(response["body"].read())
    return body["embedding"]

def chat(question, context):

    prompt = f"""
Answer only using the supplied context.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say that it was not found in the uploaded documents.
"""

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 1000,
            "temperature": 0.2,
            "topP": 0.9
        }
    }

    response = client.invoke_model(
        modelId=os.getenv("BEDROCK_LLM_MODEL"),
        body=json.dumps(payload)
    )

    result = json.loads(
        response["body"].read()
    )

    return result["output"]["message"]["content"][0]["text"]
