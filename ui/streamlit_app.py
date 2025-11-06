"""
Advanced Streamlit Web Interface for Document Q&A System

This module provides a comprehensive web-based user interface built with Streamlit
for the Document Q&A system. It offers an intuitive way to upload documents,
ask questions, and manage the knowledge base.

Features:
- Document Upload & Management: Support for PDF, TXT, DOCX, PPTX, Markdown
- Interactive Q&A Interface: Ask questions and get AI-powered answers
- Conversation Mode: Context-aware follow-up questions
- Analytics Dashboard: Usage statistics and performance metrics
- Document Explorer: Browse and search through uploaded documents
- Configuration Settings: Adjust retrieval strategies and model parameters
- Export Capabilities: Download chat history and analytics reports

Interface Components:
1. Sidebar: System status, navigation, and configuration
2. Document Upload: Multi-file upload with processing options
3. Q&A Interface: Question input with advanced options
4. Document Manager: View, search, and manage uploaded documents
5. Analytics: Visual dashboards and statistics

Navigation Pages:
- Q&A Interface: Main chat interface for asking questions
- Upload Documents: Document upload and processing
- Manage Documents: Document management and organization
- Analytics: Usage statistics and system performance

Usage:
    streamlit run ui/streamlit_app.py
    
    Then navigate to http://localhost:8501 in your browser
    
Technical Notes:
- Uses Streamlit's session state for maintaining application state
- Implements real-time processing feedback and status updates
- Provides responsive design with multiple column layouts
- Includes error handling and user-friendly error messages
"""
import streamlit as st
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

# Add project root to path for imports
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Local imports
from core.document_processor import DocumentProcessor
from core.vector_store import ChromaDBManager
from core.qa_engine import DocumentQAEngine
from config import config, setup_environment

# Page configuration
st.set_page_config(
    page_title=config.ui.title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state="expanded"
)

# Modern Dark UI with Visible Labels
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #4facfe;
    }
    
    .stat-label {
        color: #ccc;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* Success/Error messages */
    .success-banner {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-banner {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* SIDEBAR: Keep white text */
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* MAIN CONTENT: Dark theme with readable text */
    .main *,
    div[data-testid="stAppViewContainer"] *,
    section[data-testid="stAppViewContainer"] * {
        color: #fafafa !important;
    }
    
    /* Headers: Light colored and bold */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
        font-weight: bold !important;
    }
    
    /* CRITICAL: ALL LABELS must be visible with white text on dark background */
    label,
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] *,
    .stTextArea label,
    .stTextInput label,
    .stSelectbox label,
    .stNumberInput label,
    .stCheckbox label,
    .stFileUploader label,
    .stSlider label,
    .stRadio label,
    .stMultiSelect label,
    div[class*="stWidget"] label,
    span[class*="css-"],
    .streamlit-expanderHeader,
    div[data-testid="stExpander"] > div:first-child {
        color: #FFFFFF !important;
        background: rgba(0,0,0,0.8) !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        display: inline-block !important;
        margin-bottom: 6px !important;
        border: 1px solid #FFFFFF !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Button styling with proper contrast */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        color: white !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar buttons */
    div[data-testid="stSidebar"] .stButton > button,
    div[data-testid="stSidebar"] .stButton > button * {
        color: white !important;
    }
    
    /* Input fields: Clean white background with dark text */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 2px solid #e9ecef !important;
        border-radius: 8px !important;
    }
    
    .stTextArea > div > div > textarea:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        background: rgba(102, 126, 234, 0.1) !important;
        padding: 2rem !important;
    }
    
    /* Alert messages with proper contrast */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
        color: white !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%) !important;
        color: white !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important;
        color: #333 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important;
        color: #333 !important;
    }
    
    /* Ensure all elements are visible */
    * {
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat history styling */
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        margin: 0.5rem 0 !important;
    }
    
    .stSuccess {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 18px 18px 18px 4px !important;
        margin: 0.5rem 0 !important;
        border: 1px solid #e9ecef !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    This function sets up all necessary session state variables for the application,
    including component instances, chat history, and UI state. Session state in
    Streamlit persists across user interactions and page reloads.
    
    Session State Variables:
    - document_processor: DocumentProcessor instance for handling file processing
    - vector_store: ChromaDBManager instance for vector database operations
    - qa_engine: DocumentQAEngine instance for question-answering
    - chat_history: List of previous Q&A interactions
    - uploaded_files_info: Information about uploaded files
    - system_initialized: Boolean flag for system initialization status
    - conversation_mode: Boolean flag for conversation context
    - current_retrieval_strategy: Active retrieval strategy setting
    - processing_status: Current status of document processing
    """
    session_vars = {
        'document_processor': None,
        'vector_store': None,
        'qa_engine': None,
        'chat_history': [],
        'uploaded_files_info': [],
        'system_initialized': False,
        'conversation_mode': False,
        'current_retrieval_strategy': config.retrieval.search_type,
        'processing_status': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def initialize_system():
    """Initialize the complete document Q&A system."""
    try:
        with st.spinner("🔧 Initializing Document Q&A System..."):
            # Setup environment
            setup_environment()
            
            # Initialize components
            if st.session_state.document_processor is None:
                st.info("📄 Initializing document processor...")
                st.session_state.document_processor = DocumentProcessor()
            
            if st.session_state.vector_store is None:
                st.info("🗄️ Initializing vector store...")
                st.session_state.vector_store = ChromaDBManager()
            
            if st.session_state.qa_engine is None:
                st.info("🤖 Initializing AI Q&A engine...")
                try:
                    st.session_state.qa_engine = DocumentQAEngine(
                        vector_store=st.session_state.vector_store
                    )
                    st.success("✅ AI engine initialized successfully!")
                except Exception as e:
                    st.error(f"❌ AI engine initialization failed: {str(e)}")
                    st.error("This might be due to missing dependencies or model loading issues.")
                    return False
            
            st.session_state.system_initialized = True
            st.success("🎉 System initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        st.error("Please check your environment and dependencies.")
        return False

def display_system_status():
    """Display modern system status in the sidebar."""
    # Sidebar header
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; font-size: 1.5rem;">🤖 RAG ChatBot</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Intelligent Document Q&A</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.system_initialized:
        st.sidebar.markdown("### 📊 System Status")
        st.sidebar.markdown('<div class="success-banner">✅ System Online & Ready</div>', unsafe_allow_html=True)
        
        # Get system statistics
        if st.session_state.qa_engine:
            stats = st.session_state.qa_engine.get_engine_stats()
            
            # Display stats in cards
            st.sidebar.markdown("### 📈 Statistics")
            
            doc_count = stats.get('vector_store_stats', {}).get('total_sources', 0)
            query_count = stats.get('total_queries', 0)
            chunks_count = stats.get('vector_store_stats', {}).get('total_chunks', 0)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.sidebar.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{doc_count}</div>
                    <div class="stat-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.sidebar.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{chunks_count}</div>
                    <div class="stat-label">Text Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.sidebar.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{query_count}</div>
                    <div class="stat-label">Queries</div>
                </div>
                """, unsafe_allow_html=True)
                
                memory_usage = "Active" if st.session_state.conversation_mode else "Inactive"
                st.sidebar.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="font-size: 1rem; color: {'#28a745' if st.session_state.conversation_mode else '#6c757d'}">{memory_usage}</div>
                    <div class="stat-label">Memory</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.sidebar.markdown("### ⚙️ Configuration")
            
            # Retrieval strategy
            strategies = {
                "similarity": "🔍 Similarity Search",
                "mmr": "🎯 Maximum Marginal Relevance", 
                "similarity_score_threshold": "📊 Threshold-based"
            }
            
            current_strategy = st.sidebar.selectbox(
                "Retrieval Strategy",
                list(strategies.keys()),
                format_func=lambda x: strategies[x],
                index=list(strategies.keys()).index(st.session_state.current_retrieval_strategy)
            )
            st.session_state.current_retrieval_strategy = current_strategy
            
            # Conversation mode toggle
            st.sidebar.markdown("### 💭 Conversation Settings")
            st.session_state.conversation_mode = st.sidebar.toggle(
                "🔄 Enable Context Memory",
                value=st.session_state.conversation_mode,
                help="Remember previous questions and answers for context-aware responses"
            )
            
            # Clear conversation button
            if st.session_state.conversation_mode:
                if st.sidebar.button("🧹 Clear Conversation", help="Clear conversation memory"):
                    if st.session_state.qa_engine:
                        st.session_state.qa_engine.clear_conversation()
                        st.sidebar.success("Conversation cleared!")
            
    else:
        st.sidebar.markdown("### ⚠️ System Status")
        st.sidebar.markdown('<div class="error-banner">System Not Initialized</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown("### 🚀 Quick Start")
        if st.sidebar.button("Initialize System", type="primary", use_container_width=True):
            initialize_system()
        
        st.sidebar.markdown("""
        **Steps to get started:**
        1. Click 'Initialize System' above
        2. Upload your documents
        3. Start asking questions!
        """)

def handle_file_upload():
    """Handle document upload and processing - simplified."""
    st.title("📄 Upload Documents")
    
    # Simple file uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload:",
        type=['pdf', 'txt', 'docx', 'pptx', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")
        
        # Show file names
        for file in uploaded_files:
            st.write(f"• {file.name}")
        
        # Simple process button
        if st.button("Process Files", type="primary"):
            process_uploaded_files(uploaded_files, 1000, 200, True)

def process_uploaded_files(uploaded_files, chunk_size, chunk_overlap, extract_tables):
    """Process uploaded files and add to vector store."""
    if not st.session_state.system_initialized:
        st.error("Please initialize the system first!")
        return
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Update processor configuration
        st.session_state.document_processor.config.chunk_size = chunk_size
        st.session_state.document_processor.config.chunk_overlap = chunk_overlap
        st.session_state.document_processor.config.extract_tables = extract_tables
        
        # Reinitialize text splitter with new config
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        st.session_state.document_processor.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=st.session_state.document_processor.config.separators,
            length_function=len
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_documents = []
        processed_files = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}")
            
            # Save file temporarily
            temp_file_path = Path(temp_dir) / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process file
                documents = st.session_state.document_processor.process_file(str(temp_file_path))
                all_documents.extend(documents)
                processed_files.append({
                    "filename": uploaded_file.name,
                    "chunks": len(documents),
                    "status": "✅ Success"
                })
                
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                processed_files.append({
                    "filename": uploaded_file.name,
                    "chunks": 0,
                    "status": f"❌ Error: {str(e)}"
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Add documents to vector store
        if all_documents:
            status_text.text("Adding documents to vector store...")
            st.session_state.vector_store.add_documents(all_documents)
        
        # Display results
        status_text.text("✅ Processing completed!")
        
        st.subheader("📊 Processing Results")
        results_df = pd.DataFrame(processed_files)
        st.dataframe(results_df, use_container_width=True)
        
        # Update stored file info
        st.session_state.uploaded_files_info.extend(processed_files)
        
        # Success message
        total_chunks = sum(file["chunks"] for file in processed_files if file["chunks"] > 0)
        st.success(f"🎉 Successfully processed {len(processed_files)} files, created {total_chunks} document chunks!")
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

def display_qa_interface():
    """Display a very simple, clean Q&A interface."""
    st.title("💬 Ask Questions")
    
    if not st.session_state.system_initialized:
        st.error("⚠️ Please initialize the system first using the sidebar.")
        return
    
    # Check if documents are available
    collection_info = st.session_state.vector_store.get_collection_info()
    if collection_info.get('total_chunks', 0) == 0:
        st.warning("📚 No documents found. Please upload documents first.")
        return
    
    # Simple status
    st.info(f"📚 Ready! {collection_info.get('total_chunks', 0)} chunks from {collection_info.get('total_sources', 0)} documents")
    
    # Use a form to handle question input and auto-clear
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Your Question:",
            placeholder="What would you like to know?",
            height=100,
            key="question_input_form"
        )
        
        submitted = st.form_submit_button("Ask Question", type="primary")
        
        if submitted and question.strip():
            with st.spinner("Getting answer..."):
                try:
                    result = st.session_state.qa_engine.ask_question(
                        question=question.strip(),
                        use_conversation=st.session_state.conversation_mode
                    )
                    st.session_state.chat_history.append(result)
                    st.success("✅ Question answered!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Simple chat history
    if st.session_state.chat_history:
        st.subheader("💭 Recent Answers")
        
        # Show last 2 conversations only
        for chat in reversed(st.session_state.chat_history[-2:]):
            st.write(f"**Q:** {chat['question']}")
            st.write(f"**A:** {chat['answer']}")
            st.write("---")
        
        # Simple clear button
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()

def process_question(question: str, k_docs: int, score_threshold=None, lambda_mult=None):
    """Process a user question and display the answer."""
    if not st.session_state.qa_engine:
        st.error("Q&A engine not initialized!")
        return
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Prepare retrieval parameters
            retrieval_params = {"k": k_docs}
            
            if score_threshold is not None:
                retrieval_params["score_threshold"] = score_threshold
            
            if lambda_mult is not None:
                retrieval_params["lambda_mult"] = lambda_mult
            
            # Ask question
            result = st.session_state.qa_engine.ask_question(
                question=question,
                use_conversation=st.session_state.conversation_mode,
                retrieval_strategy=st.session_state.current_retrieval_strategy,
                **retrieval_params
            )
            
            # Add to chat history
            st.session_state.chat_history.append(result)
            
            # Display result immediately
            st.success("✅ Question answered!")
            
            # Auto-refresh to show in history
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

def display_system_stats():
    """Display comprehensive system statistics."""
    st.subheader("📊 System Statistics")
    
    if not st.session_state.qa_engine:
        st.warning("System not initialized.")
        return
    
    stats = st.session_state.qa_engine.get_engine_stats()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", 
                 stats.get('vector_store_stats', {}).get('total_sources', 0))
    
    with col2:
        st.metric("Document Chunks", 
                 stats.get('vector_store_stats', {}).get('collection_info', {}).get('total_chunks', 0))
    
    with col3:
        st.metric("Total Queries", 
                 stats.get('total_queries', 0))
    
    with col4:
        st.metric("LLM Provider", 
                 stats.get('llm_provider', 'N/A'), delta=None)
    
    # Charts
    if st.session_state.chat_history:
        # Query frequency over time
        chat_data = []
        for chat in st.session_state.chat_history:
            try:
                timestamp = datetime.fromisoformat(chat.get('timestamp', ''))
                chat_data.append({
                    'timestamp': timestamp,
                    'question_length': len(chat.get('question', '')),
                    'answer_length': len(chat.get('answer', '')),
                    'source_count': len(chat.get('sources', []))
                })
            except:
                continue
        
        if chat_data:
            df = pd.DataFrame(chat_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Query timeline
                fig = px.scatter(df, x='timestamp', y='source_count',
                               title='Query Sources Over Time',
                               labels={'source_count': 'Number of Sources'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Question vs Answer length
                fig = px.scatter(df, x='question_length', y='answer_length',
                               title='Question vs Answer Length',
                               labels={'question_length': 'Question Length (chars)',
                                      'answer_length': 'Answer Length (chars)'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Document sources
    sources = st.session_state.vector_store.get_document_sources()
    if sources:
        st.subheader("📄 Document Sources")
        source_df = pd.DataFrame([{"Document": Path(s).name, "Full Path": s} for s in sources])
        st.dataframe(source_df, use_container_width=True)

def export_chat_history():
    """Export chat history to JSON."""
    if not st.session_state.chat_history:
        st.warning("No chat history to export.")
        return
    
    # Prepare data for export
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_queries": len(st.session_state.chat_history),
        "chat_history": st.session_state.chat_history
    }
    
    # Convert to JSON
    json_str = json.dumps(export_data, indent=2, default=str)
    
    # Provide download
    st.download_button(
        label="💾 Download Chat History (JSON)",
        data=json_str,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def display_document_manager():
    """Display document management interface."""
    st.header("🗂️ Document Management")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first.")
        return
    
    # Collection overview
    collection_info = st.session_state.vector_store.get_collection_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collection", collection_info.get('collection_name', 'N/A'))
    with col2:
        st.metric("Total Chunks", collection_info.get('total_chunks', 0))
    with col3:
        st.metric("Embedding Model", collection_info.get('embedding_model', 'N/A'))
    
    # Document sources
    sources = st.session_state.vector_store.get_document_sources()
    
    if sources:
        st.subheader("📄 Indexed Documents")
        
        # Create dataframe with document info
        doc_data = []
        for source in sources:
            # Get document metadata
            docs = st.session_state.vector_store.search_by_metadata({"source": source}, limit=1)
            if docs:
                metadata = docs[0].metadata
                doc_data.append({
                    "Filename": Path(source).name,
                    "Type": metadata.get('file_type', 'Unknown'),
                    "Chunks": metadata.get('total_chunks', 'N/A'),
                    "Path": source
                })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
        
        # Document actions
        st.subheader("🔧 Document Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Initialize confirmation state
            if 'clear_confirmed' not in st.session_state:
                st.session_state.clear_confirmed = False
            
            # Confirmation checkbox
            st.session_state.clear_confirmed = st.checkbox(
                "⚠️ I understand this will delete all documents",
                value=st.session_state.clear_confirmed
            )
            
            # Clear button (only enabled if confirmed)
            if st.button("🧹 Clear All Documents", 
                        type="secondary", 
                        disabled=not st.session_state.clear_confirmed):
                with st.spinner("Clearing documents..."):
                    success = st.session_state.vector_store.delete_collection()
                    if success:
                        st.success("✅ All documents cleared!")
                        st.session_state.chat_history = []
                        st.session_state.clear_confirmed = False  # Reset confirmation
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear documents")
        
        with col2:
            if st.button("💾 Backup Collection"):
                backup_path = f"./backups/"
                success = st.session_state.vector_store.backup_collection(backup_path)
                if success:
                    st.success("✅ Collection backed up successfully!")
                else:
                    st.error("❌ Backup failed")
    
    else:
        st.info("📭 No documents indexed yet. Upload some documents to get started!")

def main():
    """Main application function with modern UI."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        display_system_status()
        
        # Navigation with modern styling
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        # Create navigation buttons instead of selectbox
        nav_options = {
            "💬 Q&A Interface": "Chat with your documents using AI",
            "📄 Upload Documents": "Add new documents to your knowledge base", 
            "🗂️ Manage Documents": "View and manage your document collection",
            "📊 Analytics": "View system statistics and performance"
        }
        
        # Initialize page selection in session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "💬 Q&A Interface"
        
        for page_name, description in nav_options.items():
            is_selected = st.session_state.current_page == page_name
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(
                page_name, 
                type=button_type, 
                use_container_width=True,
                help=description,
                key=f"nav_{page_name}"
            ):
                st.session_state.current_page = page_name
                st.rerun()
        
        page = st.session_state.current_page
    
    # Main content based on navigation
    if page == "💬 Q&A Interface":
        display_qa_interface()
    elif page == "📄 Upload Documents":
        handle_file_upload()
    elif page == "🗂️ Manage Documents":
        display_document_manager()
    elif page == "📊 Analytics":
        display_system_stats()
    
    # Simple Footer
    st.write("")
    st.write("---")
    st.write("🤖 RAG ChatBot - Simple Document Q&A")

if __name__ == "__main__":
    main()