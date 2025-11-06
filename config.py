"""
Configuration Management for Document Q&A AI System

This module provides centralized configuration management for all system components.
It uses dataclasses to organize settings into logical groups and supports environment
variable overrides for flexible deployment.

Main Configuration Groups:
- EmbeddingConfig: Settings for text embedding models
- ChromaDBConfig: Vector database configuration
- LLMConfig: Language model settings (OpenAI, Hugging Face)
- DocumentProcessingConfig: Text processing and chunking parameters
- RetrievalConfig: Document retrieval strategies and parameters
- SystemConfig: Main configuration that combines all components

Usage:
    from config import config
    
    # Access specific configuration
    embedding_model = config.embedding.model_name
    chunk_size = config.document_processing.chunk_size
    
    # Setup environment
    setup_environment()
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class EmbeddingConfig:
    """
    Configuration for text embedding models used for document vectorization.
    
    Attributes:
        model_name (str): Name of the sentence transformer model from Hugging Face
                         Common options:
                         - "all-MiniLM-L6-v2": Fast, good quality (default)
                         - "all-mpnet-base-v2": Better quality, slower
                         - "all-distilroberta-v1": Balanced speed/quality
        model_kwargs (Dict): Additional arguments passed to the model initialization
        encode_kwargs (Dict): Arguments for text encoding process
                             - normalize_embeddings: Whether to normalize output vectors
    """
    model_name: str = "all-MiniLM-L6-v2"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=lambda: {"normalize_embeddings": False})

@dataclass
class ChromaDBConfig:
    """
    Configuration for ChromaDB vector database settings.
    
    Attributes:
        persist_directory (str): Directory where ChromaDB stores data persistently
                               Default: "./chroma_db"
        collection_name (str): Name of the collection to store document embeddings
                              Default: "document_qa"
        distance_function (str): Distance metric for similarity calculations
                               Options: "cosine", "euclidean", "manhattan"
                               Default: "cosine" (recommended for text embeddings)
    """
    persist_directory: str = "./chroma_db"
    collection_name: str = "document_qa"
    distance_function: str = "cosine"
    
@dataclass
class LLMConfig:
    """
    Configuration for Language Models.
    
    Attributes:
        provider (str): LLM provider - "huggingface" (free, local) or "openai" (requires API key)
        model_name (str): OpenAI model name (when provider="openai")
        temperature (float): Randomness in generation (0.0 = deterministic, 1.0 = creative)
        max_tokens (int): Maximum tokens in response
        hf_model_name (str): Hugging Face model name (when provider="huggingface")
        device (str): Device for Hugging Face models ("auto", "cpu", "cuda")
    """
    provider: str = "huggingface"  # Default to free Hugging Face models
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500  # Reduced for faster local generation
    
    # For Hugging Face models (free, no API key needed)
    hf_model_name: str = "google/flan-t5-base"  # Good free model for Q&A
    device: str = "auto"

@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    
    # Supported file types
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf", ".txt", ".docx", ".pptx", ".md"]
    )
    
    # Processing options
    extract_images: bool = False
    extract_tables: bool = True
    preserve_formatting: bool = True

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    search_type: str = "similarity"  # similarity, mmr, similarity_score_threshold
    k: int = 4  # Number of documents to retrieve
    score_threshold: float = 0.5
    fetch_k: int = 20  # For MMR
    lambda_mult: float = 0.5  # For MMR diversity

    
@dataclass
class UIConfig:
    """Configuration for UI settings."""
    title: str = "Document Q&A AI System"
    page_icon: str = "🤖"
    layout: str = "wide"
    max_upload_size: int = 200  # MB

@dataclass
class SystemConfig:
    """Main system configuration."""
    # Component configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Environment settings
    openai_api_key: Optional[str] = None
    debug: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Load environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

# Global configuration instance
config = SystemConfig()

# Environment setup
def setup_environment():
    """Setup environment variables and paths."""
    # Create necessary directories
    Path(config.chromadb.persist_directory).mkdir(parents=True, exist_ok=True)
    
    # Set OpenAI API key if available
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    
    return config

def get_prompt_templates():
    """Get prompt templates for different use cases."""
    return {
        "qa_prompt": """Answer this question based on the given information:

Information: {context}

Question: {question}

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

# Validation functions
def validate_config(config: SystemConfig) -> List[str]:
    """Validate configuration settings."""
    errors = []
    
    if config.llm.provider == "openai" and not config.openai_api_key:
        errors.append("OpenAI API key is required when using OpenAI models")
    
    if config.retrieval.k <= 0:
        errors.append("Retrieval k must be greater than 0")
    
    if config.document_processing.chunk_size <= 0:
        errors.append("Chunk size must be greater than 0")
    
    if config.retrieval.score_threshold < 0 or config.retrieval.score_threshold > 1:
        errors.append("Score threshold must be between 0 and 1")
    
    return errors