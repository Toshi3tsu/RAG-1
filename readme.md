# RAG Application with Streamlit

## Overview
This is a **Retrieval-Augmented Generation (RAG)** system built with Streamlit, leveraging **OpenAI's GPT model**, FAISS for vector-based document search, and Neo4j for graph-based metadata exploration. The application enables efficient question-answering and document metadata extraction.

## Features
- **Question Answering**: Combines semantic search with a retrieval-augmented generation pipeline.
- **Document Metadata Extraction**: Extracts key information such as title, author, themes, and synopsis from documents.
- **Graph Integration**: Integrates Neo4j for advanced metadata querying.
- **Customizable Chunking**: Processes documents in pre-defined or dynamically sized chunks.

## Demo
![App Screenshot](screenshot.png)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-app.git
   cd rag-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Add your OpenAI API key:
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Upload a Document**:
   - Use the file uploader in the sidebar to upload `.txt` files for processing.

2. **Process the Document**:
   - Select chunking options and click "Process Document" to split and vectorize the document.

3. **Ask Questions**:
   - Input a question in the RAG or Simple Chat mode and receive answers based on the processed documents.

4. **Download Results**:
   - Metadata, chunks, and vectorized data can be downloaded for further use.

## Technologies Used
- **Streamlit**: User interface for document upload and interaction.
- **OpenAI GPT**: Natural language understanding and generation.
- **FAISS**: Fast vector similarity search.
- **Neo4j**: Graph-based metadata storage and querying.

## Project Structure
```
rag-app/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── data/                # Sample data or processed files
├── README.md            # Project description
└── other-scripts/       # Supporting modules or utilities
```

## Contributing
Feel free to fork this repository and contribute by submitting pull requests.

## License
This project is licensed under the MIT License. See `LICENSE` for details.