import os
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
from supabase import create_client, Client
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the key from your .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# fix path
script_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(script_dir, "..", "data", "Ebook_Agentic_AI.pdf")

TABLE_NAME = "pdf_pages"

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
google_client = genai.Client(api_key=api_key)

def get_chat_response(prompt_text):
    """Function to get response from Gemini"""
    try:
        response = google_client.models.generate_content(
            model="gemini-3-flash",
            contents=prompt_text
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def upload_pdf_to_supabase():
    """Reads PDF, chunks text, embeds, and uploads to Supabase."""

    # 1. Initialize Supabase Client
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "Error: Supabase credentials not found in .env file."

    # 2. Check if file exists
    if not os.path.exists(PDF_PATH):
        return f"Error: File not found at {PDF_PATH}"

    # 3. Read PDF and Extract Text
    try:
        reader = PdfReader(PDF_PATH)
        pdf_name = os.path.basename(PDF_PATH)
        total_pages = len(reader.pages)
        
        print(f"Processing '{pdf_name}' with {total_pages} pages...")
        
        # Initialize Text Splitter
        EMBEDDING_MODEL = "text-embedding-004"
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )

        rows_to_insert = []

        for i, page in enumerate(reader.pages):
            # Extract text
            text_content = page.extract_text() or ""
            
            # Split the page text into smaller chunks
            chunks = text_splitter.split_text(text_content)

            for chunk in chunks:
                # Generate Embedding for each chunk
                # We use the new Google Gen AI SDK syntax
                response = google_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk
                )
                # Extract the embedding vector
                embedding_vector = response.embeddings[0].values
                
                # Create data object matching your table schema
                row = {
                    "pdf_name": pdf_name,
                    "page_number": i + 1,
                    "content": chunk,
                    "embedding": embedding_vector
                }
                rows_to_insert.append(row)

        # 4. Insert into Supabase
        print(f"Extracted {len(rows_to_insert)} pages. Uploading to Supabase...")

        # Insert into Supabase
        print(f"Uploading {len(rows_to_insert)} total chunks to Supabase...")

        # We use a single batch insert for efficiency
        response = supabase.table(TABLE_NAME).insert(rows_to_insert).execute()
        print("Success! Data uploaded.")
        # iprint(response)

    except Exception as e:
        print(f"An error occurred: {e}")

def query_documents(query_text, use_hybrid=True):
    """Retrieves context and generates a response."""

    # 1. Embed the user's query
    # Must use the same model as the ingestion phase ('text-embedding-004')
    response = google_client.models.embed_content(
        model="text-embedding-004",
        contents=query_text
    )
    query_vector = response.embeddings[0].values

    # 2. Query Supabase using the RPC function we just created
    # This finds chunks with similarity > 0.5 (adjust as needed)
    rpc_response = supabase.rpc(
        "match_documents_hybrid",
        {
            "query_embedding": query_vector,
            "query_text": query_text,
            "match_threshold": 0.3,  # Lower threshold for hybrid
            "match_count": 5,
            "semantic_weight": 0.7
        }
    ).execute()

    # Check if we actually got data back
    if not rpc_response.data:
        return "No relevant documents found in the database to answer your question."
    
    # 4. Build context with metadata
    context_parts = []
    for i, match in enumerate(rpc_response.data, 1):
        score = match.get('combined_score', 0)
        context_parts.append(
            f"[Source {i} - {match.get('pdf_name', 'Unknown')} - "
            f"Page {match.get('page_number', '?')} - Score: {score:.3f}]\n"
            f"{match['content']}\n"
        )

    context_text = "\n---\n".join(context_parts)

    # 4. prompt Gemini with the Context
    full_prompt = f"""
    You are a helpful AI assistant. Answer the question based ONLY on the following context:
    
    CONTEXT:
    {context_text}
    
    QUESTION:
    {query_text}
    """

    return get_chat_response(full_prompt)

#upload_pdf_to_supabase()

if __name__ == "__main__":
    # Uncomment the line below ONLY if you need to upload the PDF
    # Test the query
    user_query = "What is agentic AI?"
    print(f"Querying: {user_query}")
    answer = query_documents(user_query)
    print("\nAnswer:")
    print(answer)