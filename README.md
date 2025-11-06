# 🤖 RAG ChatBot - Intelligent Document Q&A System

A powerful **Retrieval-Augmented Generation (RAG)** application that enables intelligent question-answering over your document collections using AI. Upload your documents and get accurate, contextual answers with source citations.

## 🚀 Features

### 📄 Document Processing
- **Multiple Formats**: PDF, TXT, DOCX, PPTX, Markdown
- **Advanced Extraction**: Text, tables, metadata
- **Smart Chunking**: Configurable text segmentation
- **Batch Processing**: Handle multiple documents

### 🔍 Intelligent Retrieval
- **Vector Search**: Semantic similarity using embeddings
- **Multiple Strategies**: Similarity, MMR (Maximum Marginal Relevance), Score Threshold
- **ChromaDB Integration**: Persistent vector storage
- **Metadata Filtering**: Search by document properties

### 🤖 AI-Powered Q&A
- **LangChain Integration**: Advanced RAG pipelines
- **Multiple LLM Support**: OpenAI, Hugging Face models
- **Conversation Mode**: Context-aware follow-up questions
- **Source Attribution**: Track answer sources

### 🖥️ User Interfaces
- **Streamlit Web App**: Beautiful, interactive interface
- **REST API**: FastAPI with auto-documentation
- **Document Management**: Upload, organize, search documents
- **Analytics Dashboard**: Usage statistics and insights

### 📊 Advanced Features
- **Document Management**: Version control, metadata, search
- **Analytics & Monitoring**: Query tracking, performance metrics
- **Conversation History**: Persistent chat sessions
- **Export Capabilities**: Chat history, analytics reports
- **Backup & Recovery**: Collection backup/restore

## 🛠️ Installation

### Quick Setup

```bash
# Clone the repository
cd "Document Q&A System"

# Run the setup script
python scripts/setup.py
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements_advanced.txt

# Create necessary directories
mkdir -p chroma_db document_storage analytics backups logs

# Copy environment template
cp .env.example .env
```

### Environment Configuration

Edit the `.env` file:

```env
# OpenAI API Key (optional)
OPENAI_API_KEY=your_openai_api_key_here

# System Configuration
DEBUG=false
LOG_LEVEL=INFO

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_PROVIDER=huggingface
LLM_MODEL=microsoft/DialoGPT-medium
```

## 🚀 Usage

### Web Interface (Streamlit)

```bash
# Start the web interface
streamlit run ui/streamlit_app.py

# Or use the convenience script
./run_streamlit.sh
```

Navigate to `http://localhost:8501` in your browser.

### REST API (FastAPI)

```bash
# Start the API server
python -m uvicorn api.fastapi_app:app --reload

# Or use the convenience script
./run_api.sh
```

API documentation available at:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Command Line Interface

```python
from core.document_processor import DocumentProcessor
from core.vector_store import ChromaDBManager
from core.qa_engine import DocumentQAEngine

# Initialize components
processor = DocumentProcessor()
vector_store = ChromaDBManager()
qa_engine = DocumentQAEngine(vector_store=vector_store)

# Process documents
documents = processor.process_file("path/to/document.pdf")
vector_store.add_documents(documents)

# Ask questions
result = qa_engine.ask_question("What is the main topic of the document?")
print(result['answer'])
```

## 📚 API Reference

### Core Components

#### DocumentProcessor
```python
processor = DocumentProcessor()
documents = processor.process_file("document.pdf")
```

#### ChromaDBManager
```python
vector_store = ChromaDBManager()
vector_store.add_documents(documents)
results = vector_store.similarity_search("query", k=5)
```

#### DocumentQAEngine
```python
qa_engine = DocumentQAEngine(vector_store=vector_store)
result = qa_engine.ask_question("What is machine learning?")
```

### REST API Endpoints

#### Document Management
```http
POST /documents/upload          # Upload documents
GET  /documents/collection      # Get collection info
POST /documents/search          # Search documents
DELETE /documents/collection    # Clear collection
```

#### Question Answering
```http
POST /qa/ask                    # Ask a question
GET  /qa/conversation/history   # Get chat history
POST /qa/conversation/clear     # Clear conversation
```

#### Analytics
```http
GET /analytics/stats            # Get system statistics
GET /analytics/export-history   # Export query history
```

## ⚙️ Configuration

### Model Configuration

```python
# Embedding models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # Fast, good quality
EMBEDDING_MODEL = "all-mpnet-base-v2"         # Better quality, slower
EMBEDDING_MODEL = "all-distilroberta-v1"      # Balanced

# LLM models
LLM_MODEL = "gpt-3.5-turbo"                   # OpenAI (requires API key)
LLM_MODEL = "microsoft/DialoGPT-medium"       # Hugging Face
LLM_MODEL = "google/flan-t5-base"            # Google T5
```

### Processing Configuration

```python
CHUNK_SIZE = 1000          # Text chunk size
CHUNK_OVERLAP = 200        # Overlap between chunks
EXTRACT_TABLES = True      # Extract table content
EXTRACT_IMAGES = False     # Extract image descriptions
```

### Retrieval Configuration

```python
SEARCH_TYPE = "similarity"           # similarity, mmr, similarity_score_threshold
K = 4                               # Number of documents to retrieve
SCORE_THRESHOLD = 0.5               # Minimum similarity score
LAMBDA_MULT = 0.5                   # MMR diversity parameter
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_system.py::TestDocumentProcessor -v
python -m pytest tests/test_system.py::TestQAEngine -v

# Use convenience script
./run_tests.sh
```

## 📊 Architecture

```
Document Q&A System
├── Core Components
│   ├── DocumentProcessor      # Document parsing and chunking
│   ├── ChromaDBManager       # Vector storage and retrieval
│   └── DocumentQAEngine      # Question answering pipeline
├── User Interfaces
│   ├── Streamlit App         # Web interface
│   └── FastAPI Server        # REST API
├── Utilities
│   ├── DocumentManager       # Document lifecycle management
│   └── QAAnalytics          # Usage analytics and monitoring
└── Configuration
    └── Centralized Config    # System-wide settings
```

## 🔧 Advanced Usage

### Custom Retrieval Strategies

```python
# Similarity search
result = qa_engine.ask_question(
    "What is AI?",
    retrieval_strategy="similarity",
    k=5
)

# Maximum Marginal Relevance (for diversity)
result = qa_engine.ask_question(
    "What is AI?", 
    retrieval_strategy="mmr",
    lambda_mult=0.7,  # Higher = more diversity
    fetch_k=20
)

# Score threshold filtering
result = qa_engine.ask_question(
    "What is AI?",
    retrieval_strategy="similarity_score_threshold",
    score_threshold=0.8  # Only high-confidence matches
)
```

### Conversation Mode

```python
# Enable conversation mode for follow-up questions
result1 = qa_engine.ask_question("What is machine learning?", use_conversation=True)
result2 = qa_engine.ask_follow_up("What are its applications?")
result3 = qa_engine.ask_follow_up("How does it differ from AI?")

# View conversation history
history = qa_engine.get_conversation_history()
```

### Document Management

```python
from utils.document_manager import DocumentManager

doc_manager = DocumentManager()

# Add document with metadata
result = doc_manager.add_document(
    "report.pdf", 
    document_id="quarterly_report_2023",
    metadata={"department": "sales", "year": 2023}
)

# Search within specific documents
results = doc_manager.search_documents(
    "revenue growth",
    document_ids=["quarterly_report_2023"]
)

# Get document statistics
stats = doc_manager.get_statistics()
```

### Analytics and Monitoring

```python
from utils.analytics import QAAnalytics

analytics = QAAnalytics()

# Get usage statistics
stats = analytics.get_query_statistics(days=7)
trends = analytics.get_usage_trends(days=30)

# Generate comprehensive report
report = analytics.generate_report(days=7)

# Export analytics data
export_path = analytics.export_report(days=30)
```

## 🔒 Security Considerations

- **API Keys**: Store in environment variables, never commit to code
- **File Validation**: Automatic file type and size validation
- **Access Control**: Implement authentication for production use
- **Data Privacy**: Local processing, no data sent to external services (except chosen LLM APIs)

## 🚀 Performance Optimization

### Model Selection
- **Embedding Models**: Balance between speed and quality
- **LLM Models**: Local models for privacy, cloud models for performance
- **Quantization**: Use quantized models for faster inference

### System Optimization
```python
# Batch processing
documents = processor.process_directory("documents/")
vector_store.add_documents(documents)

# Parallel processing
import multiprocessing
processor.config.parallel_workers = multiprocessing.cpu_count()

# Memory management
processor.clear_cache()  # Clear processing cache
vector_store.optimize()  # Optimize vector storage
```

## 🛠️ Troubleshooting

### Common Issues

**1. ChromaDB Connection Issues**
```bash
# Clear ChromaDB data
rm -rf chroma_db/
# Restart application
```

**2. Out of Memory Errors**
```python
# Reduce chunk size
config.document_processing.chunk_size = 500

# Use smaller embedding model
config.embedding.model_name = "all-MiniLM-L6-v2"
```

**3. Slow Performance**
```python
# Reduce retrieval k
config.retrieval.k = 3

# Use faster models
config.llm.model_name = "gpt-3.5-turbo"  # If using OpenAI
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements_advanced.txt --force-reinstall
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m uvicorn api.fastapi_app:app --log-level debug
```

## 📈 Scalability

### Horizontal Scaling
- **Multiple Workers**: Use multiple FastAPI workers
- **Load Balancing**: Deploy behind a load balancer
- **Database Sharding**: Distribute documents across multiple ChromaDB instances

### Vertical Scaling
- **GPU Acceleration**: Use CUDA-enabled models
- **Memory Optimization**: Implement document streaming
- **Caching**: Add Redis for query caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the amazing RAG framework
- **ChromaDB**: For efficient vector storage
- **Hugging Face**: For open-source models and transformers
- **Streamlit**: For the beautiful web interface
- **FastAPI**: For the high-performance API framework

## 📞 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions for questions and ideas

---

**Made with ❤️ for the AI community**