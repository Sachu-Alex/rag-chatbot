"""
Centralized Configuration Management for RAG ChatBot System

This module provides a comprehensive configuration management system for the Document Q&A
AI application. It uses Python dataclasses to organize settings into logical groups,
making the system highly configurable and maintainable.

Features:
- Dataclass-based configuration for type safety and IDE support
- Environment variable overrides for flexible deployment
- Modular configuration groups for different system components
- Validation functions to ensure configuration integrity
- Prompt template management for consistent AI interactions

Configuration Groups:
- EmbeddingConfig: Text embedding model settings and parameters
- ChromaDBConfig: Vector database configuration and persistence settings
- LLMConfig: Language model configuration (OpenAI, Hugging Face, etc.)
- DocumentProcessingConfig: Text processing, chunking, and file handling
- RetrievalConfig: Document retrieval strategies and search parameters
- UIConfig: User interface settings and display preferences
- SystemConfig: Main configuration that orchestrates all components

Environment Variables:
- OPENAI_API_KEY: OpenAI API key for GPT models
- DEBUG: Enable debug mode (true/false)
- LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)

Usage Examples:
    from config import config, setup_environment
    
    # Initialize system environment
    setup_environment()
    
    # Access configuration settings
    model_name = config.embedding.model_name
    chunk_size = config.document_processing.chunk_size
    api_key = config.openai_api_key
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("Configuration errors:", errors)
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class EmbeddingConfig:
    """
    Configuration for text embedding models used for document vectorization.
    
    This class configures the sentence transformer models that convert text into
    numerical vectors for semantic similarity search. The choice of model affects
    both quality and performance of the RAG system.
    
    Model Recommendations:
        - "all-MiniLM-L6-v2": Fast, good quality, 384 dimensions (default)
        - "all-mpnet-base-v2": Better quality, slower, 768 dimensions
        - "all-distilroberta-v1": Balanced speed/quality, 768 dimensions
        - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": Multilingual support
    
    Attributes:
        model_name (str): Hugging Face model identifier for sentence transformers
        model_kwargs (Dict): Additional model initialization parameters
                           Example: {"device": "cuda", "trust_remote_code": True}
        encode_kwargs (Dict): Text encoding configuration
                            - normalize_embeddings: L2 normalize vectors (recommended: False for cosine similarity)
                            - convert_to_tensor: Return tensors instead of numpy arrays
                            - show_progress_bar: Display progress during batch encoding
    """
    model_name: str = "all-MiniLM-L6-v2"  # Fast and reliable default model
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=lambda: {"normalize_embeddings": False})

@dataclass
class ChromaDBConfig:
    """
    Configuration for ChromaDB vector database settings.
    
    ChromaDB is the vector database that stores document embeddings and enables
    similarity search. This configuration controls where and how the embeddings
    are stored and retrieved.
    
    Distance Functions:
        - "cosine": Measures angle between vectors (recommended for text, range: 0-2)
        - "euclidean": Straight-line distance in vector space (sensitive to magnitude)
        - "manhattan": Sum of absolute differences (L1 norm, good for sparse vectors)
    
    Attributes:
        persist_directory (str): Local directory for persistent storage of vector data
                               Should be backed up regularly for production use
        collection_name (str): Unique identifier for the document collection
                             Multiple collections can exist in the same database
        distance_function (str): Similarity metric for vector comparisons
                               "cosine" is optimal for normalized text embeddings
    """
    persist_directory: str = "./chroma_db"  # Local storage for vector database
    collection_name: str = "document_qa"   # Collection name for document embeddings
    distance_function: str = "cosine"       # Optimal for text similarity
    
@dataclass
class LLMConfig:
    """
    Configuration for Large Language Models (LLMs) used for answer generation.
    
    This configuration supports multiple LLM providers, allowing flexibility between
    cloud-based APIs (OpenAI) and local models (Hugging Face). The choice affects
    cost, privacy, and response quality.
    
    Provider Options:
        - "openai": High-quality responses, requires API key and internet, costs money
        - "huggingface": Free local models, no API key needed, varying quality
    
    OpenAI Models (when provider="openai"):
        - "gpt-3.5-turbo": Fast, cost-effective, good quality
        - "gpt-4": Highest quality, slower, more expensive
        - "gpt-4-turbo": Balance of quality and speed
    
    Hugging Face Models (when provider="huggingface"):
        - "google/flan-t5-base": Good for Q&A, relatively fast
        - "microsoft/DialoGPT-medium": Conversational, but larger
        - "distilbert-base-uncased-distilled-squad": Fast Q&A, smaller model
    
    Attributes:
        provider (str): LLM service provider ("openai" or "huggingface")
        model_name (str): OpenAI model identifier (used when provider="openai")
        temperature (float): Response randomness (0.0=deterministic, 1.0=creative)
                           Lower values for factual Q&A, higher for creative tasks
        max_tokens (int): Maximum response length in tokens
                        Balance between completeness and speed/cost
        hf_model_name (str): Hugging Face model identifier (used when provider="huggingface")
        device (str): Compute device for local models ("auto", "cpu", "cuda")
                     "auto" automatically selects GPU if available
    """
    provider: str = "huggingface"              # Default to free local models
    model_name: str = "gpt-3.5-turbo"         # OpenAI model for API usage
    temperature: float = 0.7                   # Balanced creativity for Q&A
    max_tokens: int = 500                      # Reasonable response length
    
    # Hugging Face model configuration (free local inference)
    hf_model_name: str = "google/flan-t5-base"  # Reliable Q&A model
    device: str = "auto"                         # Auto-detect GPU availability

@dataclass
class DocumentProcessingConfig:
    """
    Configuration for document processing and text chunking.
    
    This configuration controls how documents are loaded, processed, and split into
    chunks for embedding. The chunking strategy significantly impacts retrieval
    quality and system performance.
    
    Chunking Guidelines:
        - Smaller chunks: Better precision, more specific matches
        - Larger chunks: Better context, but may include irrelevant information
        - Overlap ensures important information at boundaries isn't lost
    
    Recommended Settings:
        - Technical documents: chunk_size=800, overlap=100
        - General text: chunk_size=1000, overlap=200 (default)
        - Short documents: chunk_size=500, overlap=50
    
    Attributes:
        chunk_size (int): Maximum characters per text chunk
                         Balance between context and precision
        chunk_overlap (int): Characters shared between adjacent chunks
                           Prevents information loss at boundaries
        separators (List[str]): Text splitting delimiters in order of preference
                              Higher priority separators preserve semantic structure
        supported_extensions (List[str]): File types the system can process
                                        Add new types as needed
        extract_images (bool): Whether to extract and process embedded images
                             Currently not implemented
        extract_tables (bool): Whether to extract table data as text
                             Improves retrieval from structured documents
        preserve_formatting (bool): Whether to maintain document structure markers
                                  Helps preserve headings and emphasis
    """
    chunk_size: int = 1000                     # Optimal for general documents
    chunk_overlap: int = 200                   # 20% overlap prevents information loss
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])  # Semantic splitting
    
    # File format support - easily extensible
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf", ".txt", ".docx", ".pptx", ".md"]
    )
    
    # Processing features - configure based on document types
    extract_images: bool = False               # Image processing not yet implemented
    extract_tables: bool = True                # Essential for structured documents
    preserve_formatting: bool = True           # Maintains document structure

@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval and search strategies.
    
    This configuration controls how the system finds and ranks relevant document
    chunks in response to user queries. Different strategies optimize for
    different aspects of retrieval quality.
    
    Search Strategies:
        - "similarity": Pure cosine similarity, fast and straightforward
        - "mmr": Maximum Marginal Relevance, balances relevance and diversity
        - "similarity_score_threshold": Only returns results above confidence threshold
    
    Performance vs Quality Trade-offs:
        - Higher k: More context but more noise and slower processing
        - Lower k: Faster but may miss relevant information
        - MMR: Better diversity but computationally more expensive
    
    Attributes:
        search_type (str): Retrieval strategy algorithm
                         "similarity" recommended for most use cases
        k (int): Number of document chunks to retrieve
               4-6 chunks typically provide good context/speed balance
        score_threshold (float): Minimum similarity score for inclusion
                               Used with "similarity_score_threshold" strategy
        fetch_k (int): Initial candidate pool size for MMR algorithm
                     Should be 3-5x larger than k for good diversity
        lambda_mult (float): MMR diversity parameter (0=diversity, 1=relevance)
                           0.5 provides balanced relevance and diversity
    """
    search_type: str = "similarity"            # Fast and reliable default
    k: int = 4                                 # Good balance of context and speed
    score_threshold: float = 0.5               # Moderate confidence threshold
    fetch_k: int = 20                          # Large candidate pool for MMR
    lambda_mult: float = 0.5                   # Balanced relevance/diversity

    
@dataclass
class UIConfig:
    """
    Configuration for the Streamlit user interface.
    
    This configuration controls the appearance and behavior of the web-based
    user interface built with Streamlit.
    
    Layout Options:
        - "wide": Uses full browser width, better for analytics dashboards
        - "centered": Narrower layout, better for reading and forms
    
    Attributes:
        title (str): Browser tab title and main application header
        page_icon (str): Emoji or icon displayed in browser tab
        layout (str): Streamlit layout mode ("wide" or "centered")
        max_upload_size (int): Maximum file upload size in megabytes
                             Prevents memory issues with large documents
    """
    title: str = "RAG ChatBot - Document Q&A System"  # Descriptive title
    page_icon: str = "🤖"                            # Robot emoji for AI theme
    layout: str = "wide"                             # Full-width layout for dashboards
    max_upload_size: int = 200                       # 200MB limit for document uploads

@dataclass
class SystemConfig:
    """
    Main system configuration that orchestrates all component settings.
    
    This is the root configuration class that combines all subsystem configurations
    and handles environment variable loading. It provides a single point of access
    for all system settings.
    
    Environment Variables:
        - OPENAI_API_KEY: Required for OpenAI LLM provider
        - DEBUG: Enable debug mode (true/false, default: false)
        - LOG_LEVEL: Logging verbosity (DEBUG/INFO/WARNING/ERROR, default: INFO)
    
    Attributes:
        embedding: Text embedding model configuration
        chromadb: Vector database settings
        llm: Language model configuration
        document_processing: Document loading and chunking settings
        retrieval: Search and retrieval strategy settings
        ui: User interface appearance and behavior
        openai_api_key: OpenAI API key from environment (auto-loaded)
        debug: Debug mode flag (auto-loaded from environment)
        log_level: Logging level (auto-loaded from environment)
    """
    # Component configurations - modular and independently configurable
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Environment-based settings - automatically loaded from environment variables
    openai_api_key: Optional[str] = None      # Auto-loaded from OPENAI_API_KEY
    debug: bool = False                       # Auto-loaded from DEBUG
    log_level: str = "INFO"                   # Auto-loaded from LOG_LEVEL
    
    def __post_init__(self):
        """
        Post-initialization hook to load environment variables.
        
        This method is automatically called after the dataclass is initialized,
        allowing us to override default values with environment variables.
        """
        # Load OpenAI API key securely from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Parse debug flag from environment (supports various true/false formats)
        debug_env = os.getenv("DEBUG", "false").lower()
        self.debug = debug_env in ("true", "1", "yes", "on")
        
        # Set logging level from environment with validation
        log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.log_level = log_level_env if log_level_env in valid_levels else "INFO"

# Global configuration instance - automatically loads environment variables
config = SystemConfig()

def setup_environment():
    """
    Initialize the system environment and validate configuration.
    
    This function prepares the system for operation by:
    1. Creating necessary directories for data persistence
    2. Validating configuration settings
    3. Setting up environment variables for external services
    4. Configuring logging based on current settings
    
    Returns:
        SystemConfig: The validated and initialized configuration object
        
    Raises:
        ValueError: If critical configuration validation fails
    """
    # Create necessary directories for persistent storage
    try:
        # Ensure vector database directory exists
        db_path = Path(config.chromadb.persist_directory)
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Create backup directory if it doesn't exist
        backup_path = db_path.parent / "backups"
        backup_path.mkdir(parents=True, exist_ok=True)
        
    except Exception as e:
        raise ValueError(f"Failed to create necessary directories: {e}")
    
    # Validate configuration before proceeding
    validation_errors = validate_config(config)
    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
        raise ValueError(error_msg)
    
    # Set up OpenAI API key in environment if available
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    
    # Configure Python logging based on current settings
    import logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return config

def get_prompt_templates():
    """Get prompt templates for different use cases."""
    return {
        "qa_prompt": """Context: {context}

Question: {question}

Answer the question based only on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."

Answer:""",
        
        "condense_prompt": """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""",
        
        "summarize_prompt": """Please provide a concise summary of the following text:

{text}

Summary:""",
        
        "classify_prompt": """Classify the following question into one of these categories:
- FACTUAL: Questions asking for specific facts or information
- ANALYTICAL: Questions requiring analysis or interpretation
- COMPARATIVE: Questions comparing different concepts or items
- PROCEDURAL: Questions asking how to do something

Question: {question}

Category:"""
    }

# Configuration validation functions
def validate_config(config: SystemConfig) -> List[str]:
    """
    Validate system configuration for common issues and requirements.
    
    This function performs comprehensive validation of all configuration settings
    to catch potential issues before system startup. It checks for:
    - Required API keys and credentials
    - Valid parameter ranges and types
    - Logical consistency between related settings
    - File system permissions and paths
    
    Args:
        config (SystemConfig): The configuration object to validate
        
    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    errors = []
    
    # LLM configuration validation
    if config.llm.provider == "openai" and not config.openai_api_key:
        errors.append("OpenAI API key is required when LLM provider is set to 'openai'")
    
    if config.llm.temperature < 0 or config.llm.temperature > 1:
        errors.append("LLM temperature must be between 0.0 and 1.0")
    
    if config.llm.max_tokens <= 0:
        errors.append("LLM max_tokens must be greater than 0")
    
    # Retrieval configuration validation
    if config.retrieval.k <= 0:
        errors.append("Retrieval k (number of documents) must be greater than 0")
    
    if config.retrieval.score_threshold < 0 or config.retrieval.score_threshold > 1:
        errors.append("Retrieval score threshold must be between 0.0 and 1.0")
    
    if config.retrieval.lambda_mult < 0 or config.retrieval.lambda_mult > 1:
        errors.append("MMR lambda multiplier must be between 0.0 and 1.0")
    
    if config.retrieval.fetch_k < config.retrieval.k:
        errors.append("MMR fetch_k must be greater than or equal to k")
    
    # Document processing validation
    if config.document_processing.chunk_size <= 0:
        errors.append("Document chunk size must be greater than 0")
    
    if config.document_processing.chunk_overlap < 0:
        errors.append("Document chunk overlap cannot be negative")
    
    if config.document_processing.chunk_overlap >= config.document_processing.chunk_size:
        errors.append("Document chunk overlap must be smaller than chunk size")
    
    # ChromaDB configuration validation
    try:
        db_path = Path(config.chromadb.persist_directory)
        if db_path.exists() and not db_path.is_dir():
            errors.append(f"ChromaDB persist directory exists but is not a directory: {db_path}")
    except Exception as e:
        errors.append(f"Invalid ChromaDB persist directory path: {e}")
    
    # UI configuration validation
    if config.ui.max_upload_size <= 0:
        errors.append("Maximum upload size must be greater than 0")
    
    if config.ui.layout not in ["wide", "centered"]:
        errors.append("UI layout must be either 'wide' or 'centered'")
    
    # Embedding configuration validation
    if not config.embedding.model_name or not config.embedding.model_name.strip():
        errors.append("Embedding model name cannot be empty")
    
    # Search type validation
    valid_search_types = ["similarity", "mmr", "similarity_score_threshold"]
    if config.retrieval.search_type not in valid_search_types:
        errors.append(f"Invalid search type. Must be one of: {', '.join(valid_search_types)}")
    
    return errors