
# AWS Bedrock RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Amazon Bedrock, Amazon Titan Embeddings V2, Amazon Nova Pro, FAISS, and Streamlit.

The application allows users to upload PDF and TXT documents, create a searchable knowledge base using vector embeddings, and ask questions grounded in the uploaded documents.

- **Features:**
  * PDF document ingestion
  * TXT document ingestion
  * Multi-document support
  * Amazon Titan Embeddings V2 for vector generation
  * FAISS vector database for semantic search
  * Amazon Nova Pro for answer generation
  * Source attribution for retrieved documents
  * Streamlit-based chat interface
  * Local vector storage
  * AWS-native GenAI architecture
 
- **Tech Stack:**
  * Frontend
    + Streamlit
  * AWS Services
    + Amazon Bedrock
    + Amazon Titan Embeddings V2
    + Amazon Nova Pro
  * Vector Database
    + FAISS
  * Language
    + Python
   
**IAM Permissions**

The IAM user requires access to Amazon Bedrock:

{
  "Version":"2012-10-17", </br>
  "Statement":[ </br>
    { </br>
      "Effect":"Allow", </br>
      "Action":[ </br>
        "bedrock:InvokeModel", </br>
        "bedrock:InvokeModelWithResponseStream" </br>
      ], </br>
      "Resource":"*" </br>
    } </br>
  ] </br>
}

**How to run**
- Create Virtual Environment: </br>
    python -m venv venv </br>

- Activate Virtual Environment: </br>
Windows:</br>
venv\Scripts\activate</br>
Linux/Mac:</br>
source venv/bin/activate

- Install Dependencies:</br>
pip install -r requirements.txt

- Final Run:</br>
streamlit run app.py </br>
or</br>
python -m streamlit run app.py


**Example Workflow**
- Upload one or more PDF/TXT files.
- Build the knowledge base.
- Ask questions related to the uploaded documents.
- View AI-generated answers along with document sources.
