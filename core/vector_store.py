"""
Advanced ChromaDB Vector Store Manager

This module provides a sophisticated interface to ChromaDB for storing and retrieving
document embeddings. It handles all vector database operations including document
storage, similarity search, and metadata management.

Key Features:
- Persistent vector storage with ChromaDB
- Multiple search strategies (similarity, MMR, threshold-based)
- Comprehensive metadata tracking and management
- Collection backup and restore capabilities
- Integration with LangChain's retriever interface
- Automatic embedding generation with configurable models

Classes:
    ChromaDBManager: Main interface for vector database operations

Usage Example:
    # Initialize vector store
    vector_store = ChromaDBManager()
    
    # Add documents with embeddings
    documents = [Document(page_content="text", metadata={"source": "file.txt"})]
    ids = vector_store.add_documents(documents)
    
    # Search for similar documents
    results = vector_store.similarity_search("query text", k=5)
    
    # Use as LangChain retriever
    retriever = vector_store.as_retriever(search_type="mmr", k=4)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path

# ChromaDB imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# LangChain imports
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Configuration
from config import config

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Advanced ChromaDB vector store manager."""
    
    def __init__(self, 
                 persist_directory: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 embedding_config: Optional[Dict[str, Any]] = None):
        """Initialize ChromaDB manager."""
        
        self.persist_directory = persist_directory or config.chromadb.persist_directory
        self.collection_name = collection_name or config.chromadb.collection_name
        
        # Setup embeddings
        self.embedding_config = embedding_config or config.embedding.__dict__
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_config['model_name'],
            model_kwargs=self.embedding_config.get('model_kwargs', {}),
            encode_kwargs=self.embedding_config.get('encode_kwargs', {})
        )
        
        # Initialize ChromaDB client
        self._initialize_client()
        
        # Initialize LangChain Chroma vectorstore
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Metadata tracking
        self.metadata_file = Path(self.persist_directory) / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with proper settings."""
        try:
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized with persist directory: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load collection metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "document_count": 0,
            "total_chunks": 0,
            "collections": {}
        }
    
    def _save_metadata(self):
        """Save collection metadata to file."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store."""
        try:
            logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
            
            # Add documents to vectorstore
            ids = self.vectorstore.add_documents(documents, **kwargs)
            
            # Update metadata
            self.metadata["total_chunks"] += len(documents)
            
            # Track unique documents
            unique_sources = set()
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                unique_sources.add(source)
            
            self.metadata["document_count"] += len(unique_sources)
            
            # Update collection-specific metadata
            if self.collection_name not in self.metadata["collections"]:
                self.metadata["collections"][self.collection_name] = {
                    "created_at": datetime.now().isoformat(),
                    "chunk_count": 0,
                    "document_sources": []
                }
            
            collection_meta = self.metadata["collections"][self.collection_name]
            collection_meta["chunk_count"] += len(documents)
            
            for source in unique_sources:
                if source not in collection_meta["document_sources"]:
                    collection_meta["document_sources"].append(source)
            
            self._save_metadata()
            
            logger.info(f"Successfully added {len(documents)} documents with IDs: {ids[:5]}{'...' if len(ids) > 5 else ''}")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4, 
                         filter: Optional[Dict[str, Any]] = None,
                         **kwargs) -> List[Document]:
        """Perform similarity search."""
        try:
            logger.debug(f"Performing similarity search for query: '{query}' with k={k}")
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            
            logger.debug(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 4,
                                   filter: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        try:
            logger.debug(f"Performing similarity search with scores for query: '{query}' with k={k}")
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            
            logger.debug(f"Found {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search with scores failed: {e}")
            raise
    
    def max_marginal_relevance_search(self, 
                                    query: str,
                                    k: int = 4,
                                    fetch_k: int = 20,
                                    lambda_mult: float = 0.5,
                                    filter: Optional[Dict[str, Any]] = None,
                                    **kwargs) -> List[Document]:
        """Perform Maximum Marginal Relevance search for diversity."""
        try:
            logger.debug(f"Performing MMR search for query: '{query}' with k={k}, fetch_k={fetch_k}")
            
            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs
            )
            
            logger.debug(f"MMR search returned {len(results)} diverse documents")
            return results
            
        except Exception as e:
            logger.error(f"MMR search failed: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed information about the collection."""
        try:
            # Get collection from ChromaDB directly
            collection = self.client.get_collection(name=self.collection_name)
            count = collection.count()
            
            info = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_model": self.embedding_config['model_name'],
                "persist_directory": self.persist_directory,
                **self.metadata.get("collections", {}).get(self.collection_name, {})
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "error": str(e)
            }
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            logger.info(f"Deleting {len(ids)} documents")
            
            # Delete from vectorstore
            self.vectorstore.delete(ids=ids)
            
            # Update metadata
            self.metadata["total_chunks"] -= len(ids)
            if self.collection_name in self.metadata["collections"]:
                self.metadata["collections"][self.collection_name]["chunk_count"] -= len(ids)
            
            self._save_metadata()
            
            logger.info(f"Successfully deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            logger.info(f"Deleting collection: {self.collection_name}")
            
            # Delete from ChromaDB
            self.client.delete_collection(name=self.collection_name)
            
            # Remove from metadata
            if self.collection_name in self.metadata["collections"]:
                del self.metadata["collections"][self.collection_name]
            
            self.metadata["total_chunks"] = 0
            self.metadata["document_count"] = 0
            self._save_metadata()
            
            # Reinitialize vectorstore
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Successfully deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def search_by_metadata(self, 
                          metadata_filter: Dict[str, Any], 
                          limit: Optional[int] = None) -> List[Document]:
        """Search documents by metadata filters."""
        try:
            logger.debug(f"Searching by metadata filter: {metadata_filter}")
            
            # Use ChromaDB's where clause functionality
            collection = self.client.get_collection(name=self.collection_name)
            
            results = collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            # Convert to LangChain Documents
            documents = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    documents.append(Document(
                        page_content=doc,
                        metadata=metadata
                    ))
            
            logger.debug(f"Found {len(documents)} documents matching metadata filter")
            return documents
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def get_document_sources(self) -> List[str]:
        """Get all unique document sources in the collection."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            
            # Get all documents with metadata
            results = collection.get(include=['metadatas'])
            
            sources = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
            
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Failed to get document sources: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        try:
            info = self.get_collection_info()
            sources = self.get_document_sources()
            
            stats = {
                "collection_info": info,
                "total_sources": len(sources),
                "sources": sources,
                "metadata": self.metadata,
                "embedding_model": self.embedding_config['model_name'],
                "persist_directory": self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def backup_collection(self, backup_path: str) -> bool:
        """Create a backup of the collection."""
        try:
            import shutil
            
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy the persist directory
            source_dir = Path(self.persist_directory)
            dest_dir = backup_path / f"backup_{self.collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            shutil.copytree(source_dir, dest_dir)
            
            logger.info(f"Collection backed up to: {dest_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def as_retriever(self, **kwargs):
        """Return a LangChain retriever interface."""
        search_type = kwargs.get('search_type', config.retrieval.search_type)
        search_kwargs = {
            'k': kwargs.get('k', config.retrieval.k),
        }
        
        if search_type == "similarity_score_threshold":
            search_kwargs['score_threshold'] = kwargs.get('score_threshold', config.retrieval.score_threshold)
        elif search_type == "mmr":
            search_kwargs.update({
                'fetch_k': kwargs.get('fetch_k', config.retrieval.fetch_k),
                'lambda_mult': kwargs.get('lambda_mult', config.retrieval.lambda_mult)
            })
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )