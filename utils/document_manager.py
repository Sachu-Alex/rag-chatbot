"""
Advanced document management utilities for the Document Q&A System.
"""
import json
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import mimetypes

# Local imports
from core.vector_store import ChromaDBManager
from core.document_processor import DocumentProcessor
from config import config

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

class DocumentManager:
    """Advanced document management system."""
    
    def __init__(self, 
                 vector_store: Optional[ChromaDBManager] = None,
                 document_processor: Optional[DocumentProcessor] = None,
                 storage_dir: str = "./document_storage"):
        """Initialize document manager."""
        self.vector_store = vector_store or ChromaDBManager()
        self.document_processor = document_processor or DocumentProcessor()
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Document registry file
        self.registry_file = self.storage_dir / "document_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load document registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
        
        return {
            "documents": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": None,
                "total_documents": 0
            }
        }
    
    def _save_registry(self):
        """Save document registry to file."""
        try:
            self.registry["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""
    
    def add_document(self, 
                    file_path: str, 
                    document_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a document to the managed collection.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom document ID
            metadata: Optional additional metadata
        
        Returns:
            Dictionary with processing results
        """
        try:
            file_path = Path(file_path).resolve()
            
            if not file_path.exists():
                return {"error": "File does not exist", "success": False}
            
            # Generate document ID if not provided
            if document_id is None:
                document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.stem}"
            
            # Compute file hash
            file_hash = self._compute_file_hash(str(file_path))
            
            # Check if document already exists
            existing_doc = self._find_document_by_hash(file_hash)
            if existing_doc:
                return {
                    "error": f"Document already exists with ID: {existing_doc['id']}",
                    "success": False,
                    "existing_document": existing_doc
                }
            
            # Get file information
            file_stats = file_path.stat()
            file_info = {
                "original_path": str(file_path),
                "filename": file_path.name,
                "file_size": file_stats.st_size,
                "file_type": file_path.suffix.lower(),
                "mime_type": mimetypes.guess_type(str(file_path))[0],
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
            
            # Copy file to storage directory
            stored_file_path = self.storage_dir / f"{document_id}{file_path.suffix}"
            shutil.copy2(file_path, stored_file_path)
            
            # Process document
            logger.info(f"Processing document: {document_id}")
            documents = self.document_processor.process_file(str(stored_file_path))
            
            if not documents:
                # Clean up stored file if processing failed
                stored_file_path.unlink(missing_ok=True)
                return {"error": "No content extracted from document", "success": False}
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents)
            
            # Create registry entry
            registry_entry = {
                "id": document_id,
                "file_hash": file_hash,
                "stored_path": str(stored_file_path),
                "file_info": file_info,
                "processing_info": {
                    "processed_at": datetime.now().isoformat(),
                    "chunks_created": len(documents),
                    "vector_ids": document_ids,
                    "processor_config": {
                        "chunk_size": self.document_processor.config.chunk_size,
                        "chunk_overlap": self.document_processor.config.chunk_overlap
                    }
                },
                "metadata": metadata or {},
                "status": "active"
            }
            
            # Add to registry
            self.registry["documents"][document_id] = registry_entry
            self.registry["metadata"]["total_documents"] += 1
            self._save_registry()
            
            logger.info(f"Successfully added document: {document_id} with {len(documents)} chunks")
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(documents),
                "file_info": file_info,
                "processing_info": registry_entry["processing_info"]
            }
            
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            return {"error": str(e), "success": False}
    
    def remove_document(self, document_id: str, remove_file: bool = False) -> Dict[str, Any]:
        """
        Remove a document from the collection.
        
        Args:
            document_id: ID of the document to remove
            remove_file: Whether to also delete the stored file
        
        Returns:
            Dictionary with removal results
        """
        try:
            if document_id not in self.registry["documents"]:
                return {"error": "Document not found", "success": False}
            
            doc_info = self.registry["documents"][document_id]
            
            # Remove from vector store
            vector_ids = doc_info.get("processing_info", {}).get("vector_ids", [])
            if vector_ids:
                success = self.vector_store.delete_documents(vector_ids)
                if not success:
                    logger.warning(f"Failed to delete some vector store entries for {document_id}")
            
            # Remove file if requested
            if remove_file:
                stored_path = Path(doc_info.get("stored_path", ""))
                if stored_path.exists():
                    stored_path.unlink()
                    logger.info(f"Deleted stored file: {stored_path}")
            
            # Remove from registry
            del self.registry["documents"][document_id]
            self.registry["metadata"]["total_documents"] -= 1
            self._save_registry()
            
            logger.info(f"Successfully removed document: {document_id}")
            
            return {
                "success": True,
                "document_id": document_id,
                "file_removed": remove_file,
                "chunks_removed": len(vector_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return {"error": str(e), "success": False}
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get detailed information about a document."""
        if document_id not in self.registry["documents"]:
            return {"error": "Document not found"}
        
        doc_info = self.registry["documents"][document_id].copy()
        
        # Add current status information
        stored_path = Path(doc_info.get("stored_path", ""))
        doc_info["file_exists"] = stored_path.exists()
        
        if stored_path.exists():
            file_stats = stored_path.stat()
            doc_info["current_file_size"] = file_stats.st_size
            doc_info["current_modified_at"] = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        
        return doc_info
    
    def list_documents(self, 
                      status_filter: Optional[str] = None,
                      file_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all managed documents with optional filters."""
        documents = []
        
        for doc_id, doc_info in self.registry["documents"].items():
            # Apply filters
            if status_filter and doc_info.get("status") != status_filter:
                continue
            
            if file_type_filter and doc_info.get("file_info", {}).get("file_type") != file_type_filter:
                continue
            
            # Create summary info
            summary = {
                "id": doc_id,
                "filename": doc_info.get("file_info", {}).get("filename", "Unknown"),
                "file_type": doc_info.get("file_info", {}).get("file_type", "Unknown"),
                "file_size": doc_info.get("file_info", {}).get("file_size", 0),
                "chunks_created": doc_info.get("processing_info", {}).get("chunks_created", 0),
                "processed_at": doc_info.get("processing_info", {}).get("processed_at", "Unknown"),
                "status": doc_info.get("status", "unknown"),
                "metadata": doc_info.get("metadata", {})
            }
            
            documents.append(summary)
        
        # Sort by processed_at date (newest first)
        documents.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
        
        return documents
    
    def update_document_metadata(self, 
                                document_id: str, 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update metadata for a document."""
        try:
            if document_id not in self.registry["documents"]:
                return {"error": "Document not found", "success": False}
            
            # Update metadata
            current_metadata = self.registry["documents"][document_id].get("metadata", {})
            current_metadata.update(metadata)
            self.registry["documents"][document_id]["metadata"] = current_metadata
            
            # Update last modified
            self.registry["documents"][document_id]["last_modified"] = datetime.now().isoformat()
            
            self._save_registry()
            
            return {
                "success": True,
                "document_id": document_id,
                "updated_metadata": current_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {document_id}: {e}")
            return {"error": str(e), "success": False}
    
    def reprocess_document(self, 
                          document_id: str, 
                          new_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reprocess a document with new configuration."""
        try:
            if document_id not in self.registry["documents"]:
                return {"error": "Document not found", "success": False}
            
            doc_info = self.registry["documents"][document_id]
            stored_path = doc_info.get("stored_path")
            
            if not Path(stored_path).exists():
                return {"error": "Stored file not found", "success": False}
            
            # Remove existing chunks from vector store
            old_vector_ids = doc_info.get("processing_info", {}).get("vector_ids", [])
            if old_vector_ids:
                self.vector_store.delete_documents(old_vector_ids)
            
            # Update processor configuration if provided
            if new_config:
                for key, value in new_config.items():
                    if hasattr(self.document_processor.config, key):
                        setattr(self.document_processor.config, key, value)
                
                # Reinitialize text splitter with new config
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                self.document_processor.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.document_processor.config.chunk_size,
                    chunk_overlap=self.document_processor.config.chunk_overlap,
                    separators=self.document_processor.config.separators,
                    length_function=len
                )
            
            # Reprocess document
            documents = self.document_processor.process_file(stored_path)
            
            if not documents:
                return {"error": "No content extracted during reprocessing", "success": False}
            
            # Add to vector store
            new_vector_ids = self.vector_store.add_documents(documents)
            
            # Update registry
            doc_info["processing_info"].update({
                "reprocessed_at": datetime.now().isoformat(),
                "chunks_created": len(documents),
                "vector_ids": new_vector_ids,
                "processor_config": {
                    "chunk_size": self.document_processor.config.chunk_size,
                    "chunk_overlap": self.document_processor.config.chunk_overlap
                }
            })
            
            self._save_registry()
            
            logger.info(f"Successfully reprocessed document: {document_id}")
            
            return {
                "success": True,
                "document_id": document_id,
                "old_chunks": len(old_vector_ids),
                "new_chunks": len(documents),
                "processing_info": doc_info["processing_info"]
            }
            
        except Exception as e:
            logger.error(f"Failed to reprocess document {document_id}: {e}")
            return {"error": str(e), "success": False}
    
    def get_document_content(self, document_id: str, chunk_limit: Optional[int] = None) -> Dict[str, Any]:
        """Get the content of a document."""
        try:
            if document_id not in self.registry["documents"]:
                return {"error": "Document not found", "success": False}
            
            doc_info = self.registry["documents"][document_id]
            
            # Search for document chunks in vector store
            filter_criteria = {"source": doc_info.get("stored_path")}
            documents = self.vector_store.search_by_metadata(filter_criteria, limit=chunk_limit)
            
            if not documents:
                return {"error": "No content found in vector store", "success": False}
            
            # Combine chunks
            chunks = []
            full_text = ""
            
            for doc in documents:
                chunk_info = {
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                chunks.append(chunk_info)
                full_text += doc.page_content + "\n\n"
            
            return {
                "success": True,
                "document_id": document_id,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "full_text": full_text.strip(),
                "document_info": doc_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get content for {document_id}: {e}")
            return {"error": str(e), "success": False}
    
    def search_documents(self, 
                        query: str, 
                        document_ids: Optional[List[str]] = None,
                        k: int = 5) -> Dict[str, Any]:
        """Search for content within specific documents."""
        try:
            # If document_ids specified, create filter
            filter_criteria = None
            if document_ids:
                valid_ids = [doc_id for doc_id in document_ids if doc_id in self.registry["documents"]]
                if not valid_ids:
                    return {"error": "No valid document IDs provided", "success": False}
                
                # Get stored paths for filtering
                stored_paths = [self.registry["documents"][doc_id]["stored_path"] for doc_id in valid_ids]
                # Note: ChromaDB filtering might need to be adjusted based on implementation
                # For now, we'll search all and filter results
            
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=k*2)  # Get more to filter
            
            # Filter results if document_ids specified
            if document_ids:
                filtered_results = []
                for doc, score in results:
                    doc_source = doc.metadata.get("source", "")
                    # Check if this chunk belongs to one of the specified documents
                    for doc_id in document_ids:
                        if doc_id in self.registry["documents"]:
                            if doc_source == self.registry["documents"][doc_id]["stored_path"]:
                                filtered_results.append((doc, score))
                                break
                
                results = filtered_results[:k]  # Limit to requested k
            else:
                results = results[:k]
            
            # Format results
            search_results = []
            for doc, score in results:
                # Find document ID from source
                doc_id = "unknown"
                for registry_id, doc_info in self.registry["documents"].items():
                    if doc_info["stored_path"] == doc.metadata.get("source"):
                        doc_id = registry_id
                        break
                
                result_info = {
                    "document_id": doc_id,
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                }
                search_results.append(result_info)
            
            return {
                "success": True,
                "query": query,
                "results": search_results,
                "total_found": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return {"error": str(e), "success": False}
    
    def _find_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find a document by its file hash."""
        for doc_id, doc_info in self.registry["documents"].items():
            if doc_info.get("file_hash") == file_hash:
                return {"id": doc_id, **doc_info}
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about managed documents."""
        docs = list(self.registry["documents"].values())
        
        if not docs:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "file_types": {},
                "storage_stats": {"total_size": 0}
            }
        
        # Basic counts
        total_documents = len(docs)
        total_chunks = sum(doc.get("processing_info", {}).get("chunks_created", 0) for doc in docs)
        
        # File type distribution
        file_types = {}
        for doc in docs:
            file_type = doc.get("file_info", {}).get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        # Storage statistics
        total_size = sum(doc.get("file_info", {}).get("file_size", 0) for doc in docs)
        
        # Processing statistics
        avg_chunks = total_chunks / total_documents if total_documents > 0 else 0
        
        # Status distribution
        status_distribution = {}
        for doc in docs:
            status = doc.get("status", "unknown")
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "avg_chunks_per_document": round(avg_chunks, 2),
            "file_types": file_types,
            "status_distribution": status_distribution,
            "storage_stats": {
                "total_size": total_size,
                "avg_size": round(total_size / total_documents, 2) if total_documents > 0 else 0,
                "storage_directory": str(self.storage_dir)
            },
            "registry_info": self.registry["metadata"]
        }
    
    def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """Clean up files in storage directory that are not in registry."""
        try:
            registry_files = set()
            for doc_info in self.registry["documents"].values():
                stored_path = doc_info.get("stored_path")
                if stored_path:
                    registry_files.add(Path(stored_path))
            
            # Find all files in storage directory
            storage_files = set(self.storage_dir.glob("*"))
            storage_files.discard(self.registry_file)  # Don't delete registry file
            
            # Find orphaned files
            orphaned_files = storage_files - registry_files
            
            removed_files = []
            for file_path in orphaned_files:
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_files.append(str(file_path))
                        logger.info(f"Removed orphaned file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove {file_path}: {e}")
            
            return {
                "success": True,
                "orphaned_files_found": len(orphaned_files),
                "files_removed": removed_files,
                "registry_files": len(registry_files),
                "storage_files_before": len(storage_files)
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return {"error": str(e), "success": False}
    
    def export_registry(self, export_path: Optional[str] = None) -> str:
        """Export the document registry to a JSON file."""
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = self.storage_dir / f"registry_export_{timestamp}.json"
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "registry": self.registry,
                "statistics": self.get_statistics()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Registry exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return ""