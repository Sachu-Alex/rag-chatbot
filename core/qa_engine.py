"""
Advanced Question-Answering Engine with RAG (Retrieval-Augmented Generation)

This module implements a sophisticated Q&A system that combines document retrieval
with language model generation to provide accurate, contextual answers based on
uploaded documents. It uses LangChain for orchestrating the RAG pipeline.

Architecture:
1. Query Processing: Analyze and potentially rephrase user questions
2. Document Retrieval: Find relevant document chunks using vector similarity
3. Context Assembly: Combine retrieved documents into coherent context
4. Answer Generation: Use LLM to generate answers based on context
5. Response Processing: Format answers with source attribution

Key Features:
- Multiple retrieval strategies (similarity, MMR, threshold-based)
- Conversational Q&A with memory and context
- Support for multiple LLM providers (OpenAI, Hugging Face)
- Source attribution and confidence scoring
- Question classification and analysis
- Document summarization capabilities
- Query history and analytics

Classes:
    QACallbackHandler: Custom callback for monitoring Q&A operations
    LLMFactory: Factory for creating different types of LLMs
    DocumentQAEngine: Main Q&A engine with RAG capabilities

Usage Example:
    # Initialize the Q&A engine
    qa_engine = DocumentQAEngine(vector_store=vector_store)
    
    # Ask a question
    result = qa_engine.ask_question("What is machine learning?")
    
    # Use conversation mode
    result = qa_engine.ask_question("What is AI?", use_conversation=True)
    follow_up = qa_engine.ask_follow_up("How does it relate to ML?")
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# LLM imports
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline

# Local imports
from core.vector_store import ChromaDBManager
from config import config, get_prompt_templates

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

class QACallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Q&A operations."""
    
    def __init__(self):
        self.start_time = None
        self.tokens_used = 0
        self.retrieval_time = 0
        self.generation_time = 0
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts running."""
        self.start_time = datetime.now()
        logger.debug("Q&A chain started")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends."""
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            logger.debug(f"Q&A chain completed in {total_time:.2f} seconds")

class LLMFactory:
    """Factory for creating different types of LLMs."""
    
    @staticmethod
    def create_llm(llm_config: Optional[Dict[str, Any]] = None) -> BaseLLM:
        """Create an LLM instance based on configuration."""
        llm_config = llm_config or config.llm.__dict__
        provider = llm_config.get('provider', 'openai')
        
        if provider == 'openai':
            return LLMFactory._create_openai_llm(llm_config)
        elif provider == 'huggingface':
            return LLMFactory._create_huggingface_llm(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def _create_openai_llm(llm_config: Dict[str, Any]) -> ChatOpenAI:
        """Create OpenAI LLM."""
        import os
        
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please either:\n"
                "1. Set OPENAI_API_KEY environment variable, or\n"
                "2. Change config.llm.provider to 'huggingface' for free local models"
            )
        
        return ChatOpenAI(
            model_name=llm_config.get('model_name', 'gpt-3.5-turbo'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 1000),
            openai_api_key=api_key
        )
    
    @staticmethod
    def _create_huggingface_llm(llm_config: Dict[str, Any]) -> HuggingFacePipeline:
        """Create Hugging Face LLM."""
        from transformers import pipeline
        import torch
        
        # Use a model specifically designed for Q&A
        model_name = "google/flan-t5-small"  # T5 model good for Q&A
        device = llm_config.get('device', 'auto')
        
        # Determine device - prefer CPU for stability
        device_id = -1  # Force CPU for better compatibility
        
        try:
            logger.info(f"Creating Hugging Face LLM with model: {model_name}")
            
            # Create a text2text generation pipeline (T5 models use this)
            pipe = pipeline(
                "text2text-generation",
                model=model_name,
                device=device_id,
                max_length=200,  # T5 uses max_length instead of max_new_tokens
                do_sample=True,
                temperature=llm_config.get('temperature', 0.7),
                truncation=True      # Enable truncation
            )
            
            logger.info("Hugging Face LLM created successfully")
            return HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={
                    "max_length": 200,
                    "do_sample": True,
                    "temperature": 0.7,
                    "truncation": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {model_name}: {e}")
            # Fallback to a simple Q&A model
            logger.info("Falling back to distilbert-base-uncased-distilled-squad")
            try:
                pipe = pipeline(
                    "question-answering",
                    model="distilbert-base-uncased-distilled-squad",
                    device=-1  # Force CPU
                )
                logger.info("Fallback Q&A model loaded successfully")
                return HuggingFacePipeline(
                    pipeline=pipe,
                    model_kwargs={}
                )
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raise RuntimeError(f"Could not load any Hugging Face model. Original error: {e}, Fallback error: {e2}")

class DocumentQAEngine:
    """Advanced document Q&A engine with multiple retrieval strategies."""
    
    def __init__(self,
                 vector_store: Optional[ChromaDBManager] = None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 retrieval_config: Optional[Dict[str, Any]] = None):
        """Initialize the Q&A engine."""
        
        # Initialize vector store
        self.vector_store = vector_store or ChromaDBManager()
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(llm_config)
        
        # Retrieval configuration
        self.retrieval_config = retrieval_config or config.retrieval.__dict__
        
        # Load prompt templates
        self.prompts = get_prompt_templates()
        
        # Initialize memory for conversational Q&A
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Callback handler
        self.callback_handler = QACallbackHandler()
        
        # Initialize chains
        self._initialize_chains()
        
        # Query history
        self.query_history = []
        
        # Test LLM functionality
        try:
            test_response = self.llm("Hello, please respond with 'Working'")
            logger.info(f"LLM test successful: {repr(test_response)}")
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            logger.warning("LLM may not be working properly!")
    
    def _initialize_chains(self):
        """Initialize various Q&A chains."""
        
        # Basic Q&A prompt with context truncation
        def format_prompt_with_truncation(context, question):
            truncated_context = self._truncate_context(context, max_tokens=400)
            return self.prompts["qa_prompt"].format(context=truncated_context, question=question)
        
        qa_prompt = PromptTemplate(
            template=self.prompts["qa_prompt"],
            input_variables=["context", "question"]
        )
        
        # Conversational Q&A prompt
        condense_prompt = PromptTemplate(
            template=self.prompts["condense_prompt"],
            input_variables=["chat_history", "question"]
        )
        
        # Create retriever with very limited k to prevent token overflow
        self.retriever = self.vector_store.as_retriever(
            search_type=self.retrieval_config['search_type'],
            k=2,  # Limit to only 2 documents for HF models to prevent token overflow
            score_threshold=self.retrieval_config.get('score_threshold', 0.5),
            fetch_k=self.retrieval_config.get('fetch_k', 20),
            lambda_mult=self.retrieval_config.get('lambda_mult', 0.5)
        )
        
        # Basic RetrievalQA chain with truncated prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            callbacks=[self.callback_handler]
        )
        
        # Conversational RetrievalQA chain
        self.conversational_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=condense_prompt,
            return_source_documents=True,
            callbacks=[self.callback_handler]
        )
    
    def ask_question(self, 
                    question: str, 
                    use_conversation: bool = False,
                    retrieval_strategy: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: The question to ask
            use_conversation: Whether to use conversational context
            retrieval_strategy: Override retrieval strategy for this query
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Update retrieval strategy if specified
            if retrieval_strategy:
                self._update_retrieval_strategy(retrieval_strategy, **kwargs)
            
            # Get relevant documents first and truncate them
            docs = self.retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} documents from vector store")
            
            if not docs:
                logger.warning("No documents retrieved! Vector store might be empty.")
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant documents to answer your question. Please make sure documents have been uploaded and processed.",
                    "sources": [],
                    "source_documents": [],
                    "timestamp": datetime.now().isoformat()
                }
            
            truncated_docs = []
            for i, doc in enumerate(docs):
                logger.info(f"Document {i}: {len(doc.page_content)} chars, metadata: {doc.metadata}")
                # Create a copy to avoid modifying original
                truncated_content = self._truncate_context(doc.page_content, max_tokens=150)
                logger.info(f"Truncated document {i}: {len(truncated_content)} chars")
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata=doc.metadata
                )
                truncated_docs.append(truncated_doc)
            
            logger.info(f"All documents processed and truncated")
            
            # Choose chain based on conversation preference
            chain = self.conversational_qa_chain if use_conversation else self.qa_chain
            
            # For now, let's use a simpler direct approach to avoid token issues
            # Prepare context manually
            context = "\n\n".join([doc.page_content for doc in truncated_docs])
            
            # Try direct LLM approach with simpler input
            try:
                logger.info(f"Context length: {len(context)} characters")
                logger.info(f"Question: {question}")
                logger.info(f"Context preview: {context[:200]}...")
                
                # Check if this is a question-answering model (fallback) or text generation
                if hasattr(self.llm, 'pipeline') and self.llm.pipeline.task == 'question-answering':
                    # For Q&A models, use context and question directly
                    logger.info("Using question-answering pipeline")
                    result_dict = self.llm.pipeline(question=question, context=context)
                    answer = result_dict.get('answer', 'No answer found')
                    logger.info(f"Q&A model response: {repr(answer)}")
                else:
                    # For text generation models, use formatted prompt
                    prompt_text = self.prompts["qa_prompt"].format(context=context, question=question)
                    logger.info(f"Using text generation with prompt length: {len(prompt_text)}")
                    logger.info(f"Prompt preview: {prompt_text[:200]}...")
                    
                    answer = self.llm(prompt_text)
                    logger.info(f"Text generation response: {repr(answer)}")
                
                result = {
                    "answer": answer,
                    "source_documents": truncated_docs
                }
            except Exception as llm_error:
                logger.error(f"Direct LLM call failed: {llm_error}")
                logger.info("Falling back to chain approach")
                # Fallback to chain approach
                try:
                    if use_conversation:
                        result = chain({"question": question})
                    else:
                        result = chain({"query": question})
                    logger.info(f"Chain result: {result}")
                except Exception as chain_error:
                    logger.error(f"Chain approach also failed: {chain_error}")
                    result = {"answer": f"Both approaches failed. Direct: {llm_error}, Chain: {chain_error}"}
            
            # Extract information with better error handling and debugging
            logger.info(f"Full result before extraction: {result}")
            
            raw_answer = result.get("answer") or result.get("result")
            logger.info(f"Raw answer extracted: {repr(raw_answer)}")
            
            if not raw_answer:
                answer = "I couldn't generate an answer - no response from AI model."
            else:
                answer = raw_answer
            
            source_documents = result.get("source_documents", truncated_docs)
            
            # Clean up the answer if it contains unwanted text
            if isinstance(answer, str):
                # Remove common unwanted prefixes/suffixes from HF models
                answer = answer.strip()
                if answer.startswith("Answer:"):
                    answer = answer[7:].strip()
                if answer.startswith("A:"):
                    answer = answer[2:].strip()
                    
            logger.debug(f"Generated answer: {answer[:100]}...")
            
            # Process source documents
            sources = self._process_source_documents(source_documents)
            
            # Create response
            response = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_documents": source_documents,
                "retrieval_strategy": self.retrieval_config['search_type'],
                "timestamp": datetime.now().isoformat(),
                "conversation_used": use_conversation
            }
            
            # Add to query history
            self.query_history.append(response)
            
            logger.info(f"Question answered successfully with {len(source_documents)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "source_documents": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _update_retrieval_strategy(self, strategy: str, **kwargs):
        """Update retrieval strategy for the current query."""
        # Temporarily update retrieval config
        old_config = self.retrieval_config.copy()
        
        self.retrieval_config['search_type'] = strategy
        
        if strategy == "similarity_score_threshold":
            self.retrieval_config['score_threshold'] = kwargs.get('score_threshold', 0.5)
        elif strategy == "mmr":
            self.retrieval_config['fetch_k'] = kwargs.get('fetch_k', 20)
            self.retrieval_config['lambda_mult'] = kwargs.get('lambda_mult', 0.5)
        
        # Recreate retriever with new config
        self.retriever = self.vector_store.as_retriever(
            search_type=self.retrieval_config['search_type'],
            k=kwargs.get('k', self.retrieval_config['k']),
            **{k: v for k, v in self.retrieval_config.items() 
               if k in ['score_threshold', 'fetch_k', 'lambda_mult']}
        )
        
        # Update chains
        self.qa_chain.retriever = self.retriever
        self.conversational_qa_chain.retriever = self.retriever
    
    def _truncate_context(self, context: str, max_tokens: int = 200) -> str:
        """Truncate context to fit within token limits for HF models."""
        # More conservative: 1 token ≈ 3 characters for safety
        max_chars = max_tokens * 3
        
        if len(context) <= max_chars:
            return context
            
        # Truncate more aggressively and add indication
        truncated = context[:max_chars-50] + "\n[Context truncated for brevity...]"
        logger.info(f"Context truncated from {len(context)} to {len(truncated)} characters")
        return truncated
    
    def _process_source_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Process source documents into a structured format."""
        sources = []
        
        for doc in documents:
            source_info = {
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "filename": doc.metadata.get("filename", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "file_type": doc.metadata.get("file_type", "Unknown")
            }
            sources.append(source_info)
        
        return sources
    
    def ask_follow_up(self, question: str, **kwargs) -> Dict[str, Any]:
        """Ask a follow-up question using conversation context."""
        return self.ask_question(question, use_conversation=True, **kwargs)
    
    def clear_conversation(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        "question": messages[i].content,
                        "answer": messages[i + 1].content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def summarize_document(self, document_source: str) -> Dict[str, Any]:
        """Generate a summary of a specific document."""
        try:
            # Search for all chunks from the specific document
            filter_criteria = {"source": document_source}
            documents = self.vector_store.search_by_metadata(filter_criteria)
            
            if not documents:
                return {
                    "source": document_source,
                    "summary": "Document not found in the knowledge base.",
                    "error": "Document not found"
                }
            
            # Combine all chunks
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Create summary prompt
            summary_prompt = PromptTemplate(
                template=self.prompts["summarize_prompt"],
                input_variables=["text"]
            )
            
            # Generate summary
            summary_chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=summary_prompt)
            result = summary_chain({"input_documents": documents}, return_only_outputs=True)
            
            return {
                "source": document_source,
                "summary": result["output_text"],
                "chunk_count": len(documents),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error summarizing document {document_source}: {e}")
            return {
                "source": document_source,
                "summary": f"Error generating summary: {str(e)}",
                "error": str(e)
            }
    
    def classify_question(self, question: str) -> Dict[str, Any]:
        """Classify the type of question being asked."""
        try:
            classify_prompt = PromptTemplate(
                template=self.prompts["classify_prompt"],
                input_variables=["question"]
            )
            
            result = self.llm(classify_prompt.format(question=question))
            
            return {
                "question": question,
                "classification": result.strip(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            return {
                "question": question,
                "classification": "UNKNOWN",
                "error": str(e)
            }
    
    def get_similar_questions(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar questions from query history."""
        try:
            if not self.query_history:
                return []
            
            # Simple similarity based on shared words (can be enhanced with embeddings)
            question_words = set(question.lower().split())
            
            similarities = []
            for query in self.query_history:
                query_words = set(query["question"].lower().split())
                similarity = len(question_words & query_words) / len(question_words | query_words)
                
                if similarity > 0.3:  # Minimum similarity threshold
                    similarities.append({
                        "question": query["question"],
                        "answer": query["answer"],
                        "similarity": similarity,
                        "timestamp": query["timestamp"]
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return []
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q&A engine."""
        try:
            vector_stats = self.vector_store.get_statistics()
            
            stats = {
                "total_queries": len(self.query_history),
                "conversation_active": len(self.memory.chat_memory.messages) > 0,
                "retrieval_config": self.retrieval_config,
                "vector_store_stats": vector_stats,
                "llm_provider": config.llm.provider,
                "llm_model": config.llm.model_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting engine stats: {e}")
            return {"error": str(e)}
    
    def export_query_history(self, filepath: str) -> bool:
        """Export query history to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.query_history, f, indent=2, default=str)
            
            logger.info(f"Query history exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting query history: {e}")
            return False