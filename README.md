# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Google's Gemini AI and Supabase vector database. This application allows users to query PDF documents using natural language and receive contextually relevant answers based on the document content.

## Features

- **PDF Document Processing**: Automatically extracts and chunks text from PDF files
- **Vector Embeddings**: Uses Google's `text-embedding-004` model for semantic search
- **Hybrid Search**: Combines semantic similarity with keyword matching for better retrieval
- **Interactive UI**: Streamlit-based chat interface for easy interaction
- **Persistent Storage**: Supabase database for storing embeddings and document chunks
- **Context-Aware Responses**: Generates answers using retrieved context with source citations

## Architecture

The application follows a RAG (Retrieval-Augmented Generation) pattern:

1. **Document Ingestion**: PDFs are read, chunked, embedded, and stored in Supabase
2. **Query Processing**: User queries are embedded using the same embedding model
3. **Retrieval**: Hybrid search retrieves the most relevant document chunks
4. **Generation**: Gemini Flash generates answers based on retrieved context

## Tech Stack

- **LLM**: Google Gemini 3 Flash
- **Embeddings**: Google `text-embedding-004`
- **Vector Database**: Supabase (PostgreSQL with pgvector)
- **UI Framework**: Streamlit
- **PDF Processing**: pypdf
- **Text Splitting**: LangChain Text Splitters

## Prerequisites

- Python 3.8+
- Google AI API key (for Gemini)
- Supabase account with a project set up
- Supabase database with the required RPC function (see Database Setup)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "RAG Chatbot"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create or update the `app/.env` file with your credentials:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

   **⚠️ Important**: Never commit your `.env` file to version control. Add it to `.gitignore`.

## Database Setup

### Supabase Table Schema

Create a table named `pdf_pages` in your Supabase database:

```sql
CREATE TABLE pdf_pages (
    id BIGSERIAL PRIMARY KEY,
    pdf_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index for faster vector similarity search
CREATE INDEX ON pdf_pages USING ivfflat (embedding vector_cosine_ops);
```

### Required RPC Function

Create a hybrid search function in Supabase SQL Editor:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table
CREATE TABLE IF NOT EXISTS pdf_pages (
    id BIGSERIAL PRIMARY KEY,
    pdf_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Drop existing function if it exists
DROP FUNCTION IF EXISTS match_documents_hybrid(vector, text, double precision, integer, double precision);

-- Create search function
CREATE OR REPLACE FUNCTION match_documents_hybrid(
    query_embedding VECTOR(768),
    query_text TEXT,
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5,
    semantic_weight FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id BIGINT,
    pdf_name TEXT,
    page_number INTEGER,
    content TEXT,
    combined_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pdf_pages.id,
        pdf_pages.pdf_name,
        pdf_pages.page_number,
        pdf_pages.content,
        (1 - (pdf_pages.embedding <=> query_embedding)) AS combined_score
    FROM pdf_pages
    WHERE pdf_pages.embedding IS NOT NULL
    ORDER BY pdf_pages.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

## Usage

### 1. Upload PDF Documents

First, ensure your PDF is placed in the `data/` directory. The default setup uses `data/Ebook_Agentic_AI.pdf`.

To upload and process the PDF into Supabase:

```python
# In app/app.py, uncomment the upload function call
upload_pdf_to_supabase()
```

Or run directly:

```bash
cd app
python app.py
```

**Note**: Only run the upload once per document to avoid duplicates.

### 2. Run the Streamlit UI

Launch the interactive chat interface:

```bash
streamlit run app/ui.py
```

The application will open in your browser at `http://localhost:8501`.

### 3. Query Documents

Type your questions in the chat interface. The bot will:
- Search for relevant document chunks
- Provide context-aware answers
- Include source citations with page numbers and similarity scores

## Project Structure

```
RAG Chatbot/
├── app/
│   ├── .env              # Environment variables (API keys)
│   ├── app.py            # Main application logic
│   └── ui.py             # Streamlit user interface
├── data/
│   └── Ebook_Agentic_AI.pdf  # Sample PDF document
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Configuration

### Chunking Parameters

Adjust text splitting in `app.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each text chunk
    chunk_overlap=150,      # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)
```

### Search Parameters

Modify retrieval settings in `query_documents()`:

```python
{
    "match_threshold": 0.3,    # Minimum similarity score
    "match_count": 5,          # Number of results to retrieve
    "semantic_weight": 0.7     # Weight for semantic vs keyword search
}
```

## Key Functions

### `upload_pdf_to_supabase()`
Reads the PDF, splits text into chunks, generates embeddings, and uploads to Supabase.

### `query_documents(query_text, use_hybrid=True)`
Performs hybrid search and generates AI responses based on retrieved context.

### `get_chat_response(prompt_text)`
Sends prompts to Google Gemini and returns generated responses.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| google-genai | 1.61.0 | Google Gemini AI API |
| streamlit | 1.53.1 | Web UI framework |
| pypdf | 6.6.2 | PDF text extraction |
| supabase | 2.27.2 | Vector database client |
| langchain-text-splitters | 1.1.0 | Text chunking utilities |
| dotenv | 0.9.9 | Environment variable management |

## Troubleshooting

### Common Issues

1. **"No relevant documents found"**
   - Ensure PDF has been uploaded to Supabase
   - Lower the `match_threshold` value
   - Check if the query matches document content

2. **API Key Errors**
   - Verify `.env` file is in the `app/` directory
   - Ensure API keys are valid and active
   - Check for trailing spaces in environment variables

3. **Embedding Dimension Mismatch**
   - Ensure Supabase table uses `VECTOR(768)` for `text-embedding-004`
   - Recreate table if dimensions are incorrect

4. **Slow Performance**
   - Create appropriate indexes on the `embedding` column
   - Reduce `match_count` to retrieve fewer results
   - Consider using `ivfflat` index for faster searches

## Security Notes

- **Never commit API keys**: Always use `.env` files and add them to `.gitignore`
- **Rotate credentials**: The API keys in the example `.env` should be replaced
- **Use environment-specific configs**: Separate development and production credentials

## Future Enhancements

- [ ] Support for multiple PDF uploads
- [ ] Document management interface
- [ ] User authentication
- [ ] Conversation history persistence
- [ ] Export chat transcripts
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Fine-tuned embedding models
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is provided as-is for educational purposes.

## Acknowledgments

- Built with [Google Gemini](https://deepmind.google/technologies/gemini/)
- Vector storage powered by [Supabase](https://supabase.com/)
- UI created with [Streamlit](https://streamlit.io/)

## Contact

For questions or issues, please open an issue in the repository.

---

**Happy Querying! 🚀**
