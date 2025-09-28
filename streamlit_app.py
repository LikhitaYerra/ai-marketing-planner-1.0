import os
import uuid
from openai import OpenAI
from newsapi import NewsApiClient
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st
import json
from datetime import datetime, timedelta
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import requests
import base64
from streamlit_oauth import OAuth2Component, StreamlitOauthError
from typing import Dict, List, Optional
import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from marketing_pipeline import MarketingPipeline

class RAGManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.initialize_vectorstore()
    
    def initialize_vectorstore(self):
        try:
            self.vectorstore = FAISS.load_local("knowledge_base", self.embeddings)
        except:
            self.vectorstore = FAISS.from_texts([""], self.embeddings)
    
    def add_to_knowledge_base(self, text: str):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self.vectorstore.save_local("knowledge_base")

    def get_context(self, query: str, k: int = 5) -> str:
        try:
            if not self.vectorstore:
                return ""
            docs = self.vectorstore.similarity_search(query or "", k=k)
            return "\n\n".join([d.page_content for d in docs if getattr(d, 'page_content', '')])
        except Exception:
            return ""

# Configure Streamlit page
st.set_page_config(
    page_title="AI Marketing Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    /* Content cards */
    .content-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
    
    /* Progress steps */
    .step-complete {
        color: #28a745;
        font-weight: bold;
    }
    
    .step-current {
        color: #007bff;
        font-weight: bold;
        background: #e3f2fd;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }
    
    .step-pending {
        color: #6c757d;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Copy button */
    .copy-button {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .copy-button:hover {
        background: #e9ecef;
    }
    
    /* Variant selection */
    .variant-card {
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .variant-card:hover {
        border-color: #007bff;
        box-shadow: 0 2px 8px rgba(0,123,255,0.15);
    }
    
    .variant-card.selected {
        border-color: #007bff;
        background: #f8f9ff;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Helpers: persisted KB guidelines and SEO reports
def _kb_log_path():
    return os.path.abspath(os.path.join('knowledge_base', 'guidelines.json'))

def load_guidelines_log():
    try:
        path = _kb_log_path()
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def append_guideline_entry(text: str):
    try:
        os.makedirs(os.path.abspath('knowledge_base'), exist_ok=True)
        entries = load_guidelines_log()
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'length': len(text or ''),
            'text': text or ''
        }
        entries.append(entry)
        with open(_kb_log_path(), 'w') as f:
            json.dump(entries, f, indent=2)
        return entry
    except Exception:
        return None

def clear_guidelines_log():
    try:
        path = _kb_log_path()
        with open(path, 'w') as f:
            json.dump([], f)
        return True
    except Exception:
        return False

def _seo_reports_path():
    return os.path.abspath('seo_reports.json')

def load_seo_reports():
    try:
        path = _seo_reports_path()
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_seo_report(text: str):
    try:
        if not text or not text.strip():
            return False
        reports = load_seo_reports()
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'length': len(text or ''),
            'text': text
        }
        # Avoid duplicate immediate saves
        if reports and reports[-1].get('text') == text:
            return True
        reports.append(entry)
        with open(_seo_reports_path(), 'w') as f:
            json.dump(reports, f, indent=2)
        return True
    except Exception:
        return False

# Initialize session state variables
INITIAL_SESSION_STATE = {
    'current_step': 1,
    'feedback_data': [],
    'pipeline_result': None,
    'content_plan_result': None,
    'headlines_result': [],
    'user_inputs': {},
    'show_article_config': False,
    'scheduled_articles': [],
    'selected_variant_index': 0,
    'feedback_text': '',
    'memory_context': '',
    'candidate_topics': [],
    'research_source': 'newsapi',  # newsapi | google | both
    'require_citations': False,
    'strict_grounding': False
}

# Initialize session state
for key, value in INITIAL_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Load scheduled articles from persistent storage
try:
    file_path = os.path.abspath('articles_data.json')
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            content = f.read()
            if content.strip():
                try:
                    articles = json.loads(content)
                    st.session_state.scheduled_articles = articles
                    print(f"[STARTUP] Loaded {len(articles)} articles from persistent storage")
                except json.JSONDecodeError:
                    print("[STARTUP] Error parsing articles_data.json")
except Exception as e:
    print(f"[STARTUP] Error loading articles: {str(e)}")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

def init_api_clients():
    """Initialize OpenAI and NewsAPI clients."""
    client = None
    newsapi = None
    
    # Initialize OpenAI (required)
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            client = OpenAI(api_key=openai_key)
        else:
            print("[ERROR] OpenAI API key not found")
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}")
    
    # Initialize NewsAPI (optional)
    try:
        news_key = os.getenv('NEWS_API_KEY')
        if news_key:
            newsapi = NewsApiClient(api_key=news_key)
            print("[INFO] NewsAPI client initialized successfully")
        else:
            print("[WARNING] NewsAPI key not found - news research will be limited")
    except Exception as e:
        print(f"[WARNING] Failed to initialize NewsAPI client: {e}")
    
    return client, newsapi

# Initialize API clients
client, newsapi = init_api_clients()
if not client:
    st.error("❌ OpenAI API key is required. Please check your .env configuration.")
    st.stop()

# Show NewsAPI status
if not newsapi:
    st.warning("⚠️ NewsAPI not configured - external research will be limited. Add NEWS_API_KEY to .env for full functionality.")

# Application Header with Status
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🎯 AI Marketing Planner")
with col2:
    # Quick status indicators
    openai_status = "🟢" if os.getenv('OPENAI_API_KEY') else "🔴"
    news_status = "🟢" if os.getenv('NEWS_API_KEY') else "🟡"
    
    st.markdown(f"""
    <div style="text-align: right; padding-top: 1rem;">
        <small>
        API Status:<br>
        {openai_status} OpenAI<br>
        {news_status} NewsAPI
        </small>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Progress Tracker
steps = ["Goals & Keywords", "Content Strategy", "Content Generation", "Review & Feedback", "Schedule & Publish"]
current_step = st.session_state.current_step

st.markdown("### 📈 Progress Overview")
progress_cols = st.columns(len(steps))
for idx, step in enumerate(steps, 1):
    with progress_cols[idx-1]:
        if idx < current_step:
            st.markdown(f'<div class="step-complete">✅ {step}</div>', unsafe_allow_html=True)
        elif idx == current_step:
            st.markdown(f'<div class="step-current">🔵 {step}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-pending">⚪ {step}</div>', unsafe_allow_html=True)

# Progress percentage
progress_percent = ((current_step - 1) / (len(steps) - 1)) * 100
st.progress(progress_percent / 100)
st.caption(f"Progress: {progress_percent:.0f}% complete")

st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings & Tools")
    
    # Knowledge Base Management
    with st.expander("📚 Knowledge Base", expanded=False):
        st.write("Add content guidelines and rules")
        new_guidelines = st.text_area(
            "New Guidelines", 
            placeholder="Enter your content guidelines here..."
        )
        if st.button("Add Guidelines"):
            if new_guidelines:
                rag_manager = RAGManager()
                rag_manager.add_to_knowledge_base(new_guidelines)
                append_guideline_entry(new_guidelines)
                st.success("Guidelines added successfully!")
            else:
                st.warning("Please enter guidelines first")

        # Previously added guidelines
        existing_entries = load_guidelines_log()
        if existing_entries:
            st.markdown("---")
            st.caption("Previously added (most recent last):")
            selected = st.selectbox(
                "View previous entries",
                options=[f"{e['timestamp']} — {e['length']} chars" for e in existing_entries],
                index=len(existing_entries)-1
            )
            idx = [f"{e['timestamp']} — {e['length']} chars" for e in existing_entries].index(selected)
            with st.expander("Preview selected entry"):
                st.text_area("Entry", value=existing_entries[idx]['text'], height=150, disabled=True)
            if st.button("Re-add selected to KB"):
                rag_manager = RAGManager()
                rag_manager.add_to_knowledge_base(existing_entries[idx]['text'])
                st.success("Re-added to knowledge base.")
            if st.button("Clear KB log"):
                if clear_guidelines_log():
                    st.success("Cleared KB log entries.")
                    st.rerun()
    
    # Content Settings
    with st.expander("🎯 Content Settings", expanded=False):
        st.selectbox(
            "Language", 
            list(MarketingPipeline.SUPPORTED_LANGUAGES.keys()),
            key="content_language"
        )
        st.selectbox(
            "Content Tone", 
            ["Professional", "Casual", "Technical", "Friendly", "Formal"],
            key="content_tone"
        )
        llm_choice = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-4o-mini", "Custom…"],
            index=0,
            key="llm_model_choice",
            help="Use a stronger model for better keyword adherence and quality."
        )
        if llm_choice == "Custom…":
            st.session_state.llm_model = st.text_input("Custom model name", value=st.session_state.get('llm_model', 'gpt-4o'))
        else:
            st.session_state.llm_model = llm_choice
        st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('llm_temperature', 0.4),
            step=0.05,
            key="llm_temperature",
            help="Lower = more accurate; Higher = more creative."
        )
        st.selectbox(
            "Keyword Strategy",
            ["Merge (SEO first)", "SEO-only", "Manual-only"],
            index=0,
            key="keyword_strategy",
            help="Control which keywords are used: only from SEO report, only manual, or merged (SEO prioritized)."
        )
        st.toggle(
            "Use AI to extract keywords from SEO report",
            key="ai_extract_keywords",
            help="When enabled, the model will parse your SEO report to extract keywords."
        )
        st.toggle(
            "Enforce keyword coverage (revise pass)",
            value=True,
            key="enforce_keyword_coverage",
            help="Ensures all required keywords appear at least once."
        )
        st.number_input(
            "Word Count Target", 
            min_value=300, 
            max_value=3000, 
            value=800,
            step=100,
            key="word_count"
        )

    # Research & Grounding
    with st.expander("🔎 Research & Grounding", expanded=False):
        st.selectbox(
            "External Research Source",
            ["NewsAPI", "Google", "Both", "None"],
            key="research_source_select",
            help="Use Google (SerpAPI/Custom Search) to reduce hallucinations with real web results."
        )
        st.toggle("Require citations in article", key="require_citations")
        st.toggle("Strict grounding (lower creativity)", key="strict_grounding")
        st.caption("Set SERPAPI_API_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID in .env for Google.")

# Main Content Area
if st.session_state.current_step == 1:
    st.header("Step 1: Define Your Marketing Goals")
    
    with st.form("goals_form"):
        user_goal = st.text_input(
            "What is your marketing goal?",
            placeholder="e.g., Increase brand awareness for eco-friendly products",
            help="Be specific about what you want to achieve"
        )
        
        keywords_input = st.text_input(
            "Target Keywords",
            placeholder="e.g., sustainable products, eco-friendly brands",
            help="Separate keywords with commas"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            target_audience = st.selectbox(
                "Target Audience",
                ["General", "B2B", "B2C", "Technical", "Enterprise", "Small Business"]
            )
        with col2:
            content_tone = st.selectbox(
                "Content Tone",
                ["Professional", "Casual", "Technical", "Conversational", "Formal"]
            )
        
        if st.form_submit_button("Next Step →"):
            if user_goal and keywords_input:
                st.session_state.user_inputs = {
                    'goal': user_goal,
                    'keywords': keywords_input,
                    'audience': target_audience,
                    'tone': content_tone
                }
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("Please fill in both the goal and keywords")

elif st.session_state.current_step == 2:
    st.header("Step 2: Content Strategy")
    
    strategy_tab, seo_tab = st.tabs(["📝 Content Plan", "🔍 SEO Analysis"])
    
    with strategy_tab:
        st.subheader("Content Planning")
        with st.form("strategy_form"):
            col1, col2 = st.columns(2)
            with col1:
                content_types = st.multiselect(
                    "Content Types",
                    ["Blog Posts", "Social Media", "Email Newsletter", "Video Scripts"],
                    default=["Blog Posts", "Social Media"]
                )
            with col2:
                platforms = st.multiselect(
                    "Target Platforms",
                    ["Website", "LinkedIn", "Instagram", "Facebook", "Twitter"],
                    default=["Website", "LinkedIn"]
                )
            st.markdown("---")
            # Topic extraction or manual topics
            existing_topics = st.session_state.get('candidate_topics', []) or []
            if not existing_topics:
                # Naive topic suggestions from goal/keywords
                goal = st.session_state.user_inputs.get('goal', '')
                keywords = st.session_state.user_inputs.get('keywords', '')
                simple_topics = []
                if goal:
                    simple_topics.append(goal)
                if keywords:
                    for kw in [k.strip() for k in keywords.split(',') if k.strip()]:
                        if len(simple_topics) < 5:
                            simple_topics.append(f"{goal} — {kw}" if goal else kw)
                st.session_state.candidate_topics = list(dict.fromkeys(simple_topics))
                existing_topics = st.session_state.candidate_topics

            st.write("Select one or more topics to develop:")
            selected_topics = st.pills(
                "Available Topics",
                options=existing_topics or [st.session_state.user_inputs.get('goal', 'Primary Topic')],
                selection_mode="multi",
                key="selected_topics_pills"
            ) if hasattr(st, 'pills') else st.multiselect(
                "Topics",
                options=existing_topics or [st.session_state.user_inputs.get('goal', 'Primary Topic')],
                default=(existing_topics[:1] if existing_topics else [])
            )

            custom_topic = st.text_input("Or add a custom topic")
            if custom_topic:
                if custom_topic not in st.session_state.candidate_topics:
                    st.session_state.candidate_topics.append(custom_topic)
                if custom_topic not in selected_topics:
                    selected_topics = list(selected_topics) + [custom_topic]

            st.session_state.user_inputs['selected_topics'] = selected_topics
            
            if st.form_submit_button("Generate Strategy"):
                st.session_state.user_inputs.update({
                    'content_types': content_types,
                    'platforms': platforms
                })
                st.session_state.current_step = 3
                st.rerun()
    
    with seo_tab:
        st.subheader("🔍 SEO Analysis")
        
        # SEO Report Status Indicator
        existing_report = st.session_state.user_inputs.get('seo_report', '')
        has_report = bool(existing_report and existing_report.strip())
        
        if has_report:
            st.success(f"✅ SEO Report Loaded ({len(existing_report)} characters)")
            with st.expander("📊 Current SEO Report Preview", expanded=False):
                st.text_area(
                    "Preview (first 500 characters)",
                    value=existing_report[:500] + ("..." if len(existing_report) > 500 else ""),
                    height=150,
                    disabled=True
                )
                if st.button("🗑️ Clear SEO Report"):
                    st.session_state.user_inputs['seo_report'] = ''
                    st.rerun()
        else:
            st.info("ℹ️ No SEO report uploaded yet")
        
        st.markdown("---")
        
        # Input Methods
        st.markdown("#### 📝 Method 1: Paste Text")
        seo_report = st.text_area(
            "Paste your SEO report or competitor analysis",
            height=200,
            value=existing_report,
            placeholder="Paste your SEO analysis, competitor research, or keyword data here..."
        )
        
        st.markdown("#### 📎 Method 2: Upload File")
        uploaded_file = st.file_uploader(
            "Upload SEO report", 
            type=["txt", "csv", "pdf"],
            help="Supported formats: TXT, CSV, PDF"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    st.warning("⚠️ PDF upload detected. Note: PDF text extraction may not be perfect.")
                    # For PDF, you might want to add PDF text extraction here
                    seo_report = "PDF content extraction not implemented yet. Please copy-paste the text manually."
                else:
                    seo_report = uploaded_file.read().decode('utf-8')
                
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                st.info(f"📊 File size: {len(seo_report)} characters")
                
            except UnicodeDecodeError:
                st.error("❌ Error: Could not decode file. Please ensure it's a valid text file.")
                seo_report = existing_report
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                seo_report = existing_report
        
        # Save SEO report and show confirmation
        if seo_report and seo_report.strip():
            if seo_report != existing_report:
                st.session_state.user_inputs['seo_report'] = seo_report
                if save_seo_report(seo_report):
                    st.success("💾 SEO report saved!")
                else:
                    st.warning("Could not persist report, but it is available in session.")
                st.rerun()
        elif not has_report:
            st.session_state.user_inputs['seo_report'] = ''
        
        # Previous reports picker
        st.markdown("---")
        prior_reports = load_seo_reports()
        if prior_reports:
            st.caption("Previously saved reports")
            choice = st.selectbox(
                "Select a report to load",
                options=[f"{r['timestamp']} — {r['length']} chars" for r in prior_reports],
                index=len(prior_reports)-1
            )
            rep_idx = [f"{r['timestamp']} — {r['length']} chars" for r in prior_reports].index(choice)
            if st.button("Load selected report"):
                st.session_state.user_inputs['seo_report'] = prior_reports[rep_idx]['text']
                st.success("Report loaded.")
                st.rerun()

        # Quick validation (show extracted keywords instead of generic terms)
        if has_report:
            st.markdown("#### 🔍 Extracted Keywords Preview")
            try:
                mp = MarketingPipeline(client, RAGManager(), newsapi)
                insights_preview = mp.analyze_seo_report(existing_report)
                kws = insights_preview.get('keywords', [])
                if st.session_state.get('ai_extract_keywords', False):
                    # Also show AI extraction for comparison
                    ai_kws = mp._ai_extract_keywords(existing_report, max_keywords=10, llm_model=st.session_state.get('llm_model', 'gpt-4o'))
                else:
                    ai_kws = []
                if kws:
                    st.success(f"Top keywords (report): {', '.join(kws[:10])}")
                if ai_kws:
                    st.info(f"AI extracted: {', '.join(ai_kws[:10])}")
                if not kws and not ai_kws:
                    st.warning("No keywords detected. Ensure your report includes 'keyword:' or 'keywords:' lines or enable AI extraction.")
            except Exception as e:
                st.warning(f"Could not parse keywords: {e}")

elif st.session_state.current_step == 3:
    st.header("Step 3: Content Generation")
    
    # Show current inputs for transparency
    with st.expander("📋 Current Settings", expanded=False):
        st.write("**Goal:** ", st.session_state.user_inputs.get('goal', 'Not set'))
        st.write("**Keywords:** ", st.session_state.user_inputs.get('keywords', 'Not set'))
        
        # Enhanced SEO report status
        seo_report = st.session_state.user_inputs.get('seo_report', '')
        if seo_report:
            char_count = len(seo_report)
            st.write("**SEO Report:** ", f"✅ Loaded ({char_count:,} characters)")
            if char_count > 1000:
                st.write("  └─ 🎯 Comprehensive report detected")
            elif char_count > 100:
                st.write("  └─ ⚠️ Basic report detected")
            else:
                st.write("  └─ ❌ Report too short, may not be useful")
        else:
            st.write("**SEO Report:** ", "❌ Not provided")
        
        st.write("**Language:** ", st.session_state.get('content_language', 'English'))
        st.write("**Tone:** ", st.session_state.get('content_tone', 'Professional'))
        st.write("**Word Count:** ", st.session_state.get('word_count', 800))
        st.write("**Feedback:** ", st.session_state.get('feedback_text', 'None'))
    
    # Regenerate when inputs change
    inputs_signature = (
        st.session_state.user_inputs.get('goal', ''),
        st.session_state.user_inputs.get('keywords', ''),
        st.session_state.user_inputs.get('seo_report', ''),
        st.session_state.get('content_language', 'English'),
        st.session_state.get('content_tone', 'Professional'),
        st.session_state.get('llm_model', 'gpt-4o'),
        float(st.session_state.get('llm_temperature', 0.4)),
        st.session_state.get('word_count', 800),
        st.session_state.get('feedback_text', ''),
        tuple(st.session_state.user_inputs.get('selected_topics', []) or []),
        st.session_state.get('research_source', 'newsapi'),
        st.session_state.get('require_citations', False),
        st.session_state.get('strict_grounding', False),
        st.session_state.get('keyword_strategy', 'Merge (SEO first)'),
        st.session_state.get('ai_extract_keywords', False),
        st.session_state.get('enforce_keyword_coverage', True),
    )
    last_signature = st.session_state.get('last_generation_signature')

    needs_generation = (
        st.session_state.pipeline_result is None or last_signature != inputs_signature
    )

    if needs_generation:
        # Validate required inputs first
        goal = st.session_state.user_inputs.get('goal', '')
        keywords = st.session_state.user_inputs.get('keywords', '')
        
        if not goal or not keywords:
            st.error("❌ Missing required inputs!")
            st.write("**Please complete Step 1 first:**")
            if not goal:
                st.write("- ❌ Marketing goal is not set")
            if not keywords:
                st.write("- ❌ Keywords are not set")
            st.write("Go back to Step 1 to set your goal and keywords.")
            st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔍 Retrieving knowledge base context...")
            progress_bar.progress(10)
            
            pipeline = MarketingPipeline(client, RAGManager(), newsapi)
            rag_manager = RAGManager()
            kb_context = rag_manager.get_context(goal)
            # Memory context accumulates from prior generations and feedback
            memory_context = st.session_state.get('memory_context', '')
            
            status_text.text(f"📚 Retrieved {len(kb_context.split())} words from knowledge base")
            progress_bar.progress(15)
            
            status_text.text("📰 Fetching relevant news and trends...")
            progress_bar.progress(25)
            
            status_text.text("🤖 Generating 4 article variants with AI...")
            progress_bar.progress(40)
            
            # Map research source selection
            source_map = {
                'NewsAPI': 'newsapi',
                'Google': 'google',
                'Both': 'both',
                'None': 'none'
            }
            selected_source = source_map.get(st.session_state.get('research_source_select', 'NewsAPI'), 'newsapi')
            st.session_state.research_source = selected_source

            result = pipeline.run_pipeline(
                report_text=st.session_state.user_inputs.get('seo_report', ''),
                topic=(st.session_state.user_inputs.get('selected_topics') or [goal])[0],
                manual_keywords=[k.strip() for k in keywords.split(',') if k.strip()],
                language_label=st.session_state.get('content_language', 'English'),
                tone=st.session_state.get('content_tone', 'Professional'),
                word_count=st.session_state.get('word_count', 800),
                kb_context=(kb_context + "\n\nPrevious session notes:\n" + memory_context if memory_context else kb_context),
                feedback=st.session_state.get('feedback_text', ''),
                variants=4,
                research_source=selected_source,
                require_citations=st.session_state.get('require_citations', False),
                strict_grounding=st.session_state.get('strict_grounding', False),
                keyword_strategy={
                    'Merge (SEO first)': 'merge',
                    'SEO-only': 'seo-only',
                    'Manual-only': 'manual-only',
                }.get(st.session_state.get('keyword_strategy', 'Merge (SEO first)'), 'merge'),
                ai_extract_keywords=st.session_state.get('ai_extract_keywords', False),
                llm_model=st.session_state.get('llm_model', 'gpt-4o'),
                llm_temperature=float(st.session_state.get('llm_temperature', 0.4)),
                enforce_keyword_coverage=st.session_state.get('enforce_keyword_coverage', True)
            )
            
            progress_bar.progress(80)
            status_text.text("✅ Generation complete!")
            
            st.session_state.pipeline_result = result
            # Update memory context with brief summary of the chosen variant and feedback applied
            try:
                variants_mem = result.get('variants', [])
                if variants_mem:
                    content_mem = variants_mem[0]['article']['content']
                    st.session_state.memory_context = (st.session_state.get('memory_context', '') + "\n\n" + content_mem[:1500]).strip()
            except Exception:
                pass
            st.session_state.last_generation_signature = inputs_signature
            
            progress_bar.progress(100)
            status_text.text(f"🎉 Generated {len(result.get('variants', []))} variants successfully!")
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"❌ Error generating content: {e}")
            st.write("**Debug info:**")
            st.write("- Check your OpenAI API key in the .env file")
            st.write("- Ensure your goal and keywords are set in Step 1")
            st.write("- Try refreshing the page")
            st.write(f"- Error details: {str(e)}")
    
    # Show generation results
    if st.session_state.pipeline_result:
        variants = st.session_state.pipeline_result.get('variants', [])
        seo_insights = st.session_state.pipeline_result.get('seo_insights', {})
        
        # Enhanced Success Metrics
        st.markdown("### 📊 Generation Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                metric_val = len(variants)
                st.metric("🎯 Variants Generated", metric_val)
                if metric_val > 0:
                    st.markdown('<span class="status-badge status-success">✓ Ready</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                seo_count = len(seo_insights.get('keywords', []))
                st.metric("🔍 SEO Keywords", seo_count)
                if seo_count > 0:
                    with st.popover("View Keywords"):
                        for kw in seo_insights.get('keywords', [])[:5]:
                            st.write(f"• {kw}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                gaps_count = len(seo_insights.get('content_gaps', []))
                st.metric("📈 Content Gaps", gaps_count)
                if gaps_count > 0:
                    with st.popover("View Gaps"):
                        for gap in seo_insights.get('content_gaps', [])[:3]:
                            st.write(f"• {gap}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                news_insights = st.session_state.pipeline_result.get('news_insights', {})
                news_count = news_insights.get('total_articles', 0)
                st.metric("📰 News Articles", news_count)
                if news_count > 0:
                    with st.popover("Recent Headlines"):
                        for headline in news_insights.get('sample_headlines', [])[:3]:
                            st.write(f"• {headline}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        if not variants:
            st.error("🚫 No Content Generated")
            with st.expander("🔧 Troubleshooting Guide", expanded=True):
                st.markdown("**Common issues and solutions:**")
                
                # Check API keys
                st.markdown("**1. API Key Issues**")
                openai_key = bool(os.getenv('OPENAI_API_KEY'))
                news_key = bool(os.getenv('NEWS_API_KEY'))
                
                col1, col2 = st.columns(2)
                with col1:
                    if openai_key:
                        st.success("✅ OpenAI API Key: Found")
                    else:
                        st.error("❌ OpenAI API Key: Missing")
                with col2:
                    if news_key:
                        st.success("✅ News API Key: Found") 
                    else:
                        st.warning("⚠️ News API Key: Missing (optional)")
                
                # Check inputs
                st.markdown("**2. Input Validation**")
                goal_check = bool(st.session_state.user_inputs.get('goal'))
                keywords_check = bool(st.session_state.user_inputs.get('keywords'))
                
                col1, col2 = st.columns(2)
                with col1:
                    if goal_check:
                        st.success("✅ Marketing Goal: Set")
                    else:
                        st.error("❌ Marketing Goal: Missing")
                with col2:
                    if keywords_check:
                        st.success("✅ Keywords: Set")
                    else:
                        st.error("❌ Keywords: Missing")
                
                # Action buttons
                st.markdown("**3. Quick Actions**")
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button("🔄 Retry Generation"):
                        st.session_state.pipeline_result = None
                        st.rerun()
                with action_col2:
                    if st.button("⬅️ Go to Step 1"):
                        st.session_state.current_step = 1
                        st.rerun()
                with action_col3:
                    if st.button("📋 Check Settings"):
                        st.info("Check your .env file and sidebar settings")
    else:
        st.info("🔄 Content will be generated automatically...")

    # Content display tabs
    if st.session_state.pipeline_result:
        try:
            tabs = st.tabs(["📄 Article Variants", "📱 Social Posts", "📰 News Insights", "✍️ Feedback", "📅 Calendar"])
            
            with tabs[0]:
                st.subheader("📝 Select an Article Variant")
                variants = st.session_state.pipeline_result.get('variants', [])
                if not variants:
                    st.warning("No variants generated")
                else:
                    # Visual variant selection
                    st.markdown("#### Choose Your Preferred Variant:")
                    variant_cols = st.columns(min(4, len(variants)))
                    
                    for i, variant in enumerate(variants):
                        with variant_cols[i % 4]:
                            article_content = variant['article']['content']
                            preview_text = article_content[:150] + "..." if len(article_content) > 150 else article_content
                            word_count = len(article_content.split())
                            
                            # Create variant card
                            card_class = "variant-card selected" if i == st.session_state.get('selected_variant_index', 0) else "variant-card"
                            
                            if st.button(
                                f"**Variant {i+1}**\n\n{word_count} words\n\n{preview_text}", 
                                key=f"variant_btn_{i}",
                                help=f"Click to select Variant {i+1}"
                            ):
                                st.session_state.selected_variant_index = i
                                st.rerun()
                    
                    # Show selected variant details
                    selected_idx = st.session_state.get('selected_variant_index', 0)
                    chosen_variant = variants[selected_idx]
                    
                    st.markdown("---")
                    st.markdown(f"### 📄 Variant {selected_idx + 1} - Full Content")
                    
                    # Article metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Word Count", len(chosen_variant['article']['content'].split()))
                    with col2:
                        st.metric("Characters", len(chosen_variant['article']['content']))
                    with col3:
                        st.metric("Language", chosen_variant['article'].get('language', 'English'))
                    with col4:
                        st.metric("Tone", chosen_variant['article'].get('tone', 'Professional'))
                    
                    # Content display with copy button
                    content_container = st.container()
                    with content_container:
                        st.text_area(
                            "Article Content",
                            value=chosen_variant['article']['content'],
                            height=450,
                            disabled=True,
                            key="selected_article_content"
                        )
                        
                        # Copy button (using Streamlit's built-in functionality)
                        if st.button("📋 Copy to Clipboard", key="copy_article"):
                            st.code(chosen_variant['article']['content'], language=None)
                            st.success("Content copied! Use Ctrl+A, Ctrl+C to copy from the code block above.")
        
            with tabs[1]:
                st.subheader("📱 Social Media Posts for Selected Variant")
                variants = st.session_state.pipeline_result.get('variants', [])
                if variants:
                    chosen_variant = variants[st.session_state.get('selected_variant_index', 0)]
                    st.info(f"📱 Posts for **Variant {st.session_state.get('selected_variant_index', 0) + 1}**")
                    
                    # Platform icons
                    platform_icons = {
                        'linkedin': '💼',
                        'twitter': '🐦', 
                        'instagram': '📸',
                        'facebook': '👥'
                    }
                    
                    # Display posts in a grid
                    social_cols = st.columns(2)
                    for idx, (platform, post) in enumerate(chosen_variant['social_posts'].items()):
                        with social_cols[idx % 2]:
                            with st.container():
                                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                                
                                # Platform header with icon and stats
                                icon = platform_icons.get(platform.lower(), '📱')
                                char_count = len(post['content'])
                                st.markdown(f"### {icon} {platform.title()}")
                                
                                # Character count with color coding
                                if platform.lower() == 'twitter' and char_count > 280:
                                    char_color = "🔴"
                                elif platform.lower() == 'linkedin' and char_count > 3000:
                                    char_color = "🔴" 
                                else:
                                    char_color = "🟢"
                                
                                st.caption(f"{char_color} {char_count} characters")
                                
                                # Content display
                                st.text_area(
                                    "Content",
                                    value=post['content'],
                                    height=150,
                                    disabled=True,
                                    key=f"social_post_{platform}",
                                    label_visibility="collapsed"
                                )
                                
                                # Copy button for each post
                                if st.button(f"📋 Copy {platform.title()} Post", key=f"copy_{platform}"):
                                    st.code(post['content'], language=None)
                                    st.success(f"{platform.title()} post copied!")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No posts available.")

            with tabs[2]:
                st.subheader("News & Trends Analysis")
                news_insights = st.session_state.pipeline_result.get('news_insights', {})
                
                if news_insights.get('total_articles', 0) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📈 Trending Topics:**")
                        if news_insights.get('trending_topics'):
                            for topic in news_insights['trending_topics'][:5]:
                                st.write(f"• {topic.title()}")
                        else:
                            st.write("No trending topics found")
                        
                        st.write("**📰 Key Sources:**")
                        if news_insights.get('key_sources'):
                            for source in news_insights['key_sources'][:3]:
                                st.write(f"• {source}")
                        else:
                            st.write("No sources identified")
                    
                    with col2:
                        st.write("**🔥 Recent Headlines:**")
                        if news_insights.get('sample_headlines'):
                            for headline in news_insights['sample_headlines'][:5]:
                                st.write(f"• {headline}")
                        else:
                            st.write("No headlines available")
                    
                    # References / Citations
                    citations = news_insights.get('citations') or []
                    if citations:
                        st.markdown("**🔗 References:**")
                        for c in citations:
                            title = c.get('title', 'Untitled')
                            url = c.get('url', '')
                            source = c.get('source', '')
                            st.write(f"[{c.get('index', '?')}] {title} — {source}: {url}")

                    st.write("**📊 Recent Developments:**")
                    if news_insights.get('recent_developments'):
                        for dev in news_insights['recent_developments'][:3]:
                            st.info(dev)
                    else:
                        st.write("No recent developments found")
                else:
                    st.info("No recent news found for your topic. This could be due to:")
                    st.write("- Very specific or niche topic")
                    st.write("- NewsAPI rate limits")
                    st.write("- Network connectivity issues")
                    st.write("- Invalid NewsAPI key")

            with tabs[3]:
                st.subheader("Feedback")
                st.info("Provide suggestions to refine the article and posts. We'll regenerate with your feedback.")
                feedback_text = st.text_area("Your suggestions", value=st.session_state.get('feedback_text', ''), height=150)
                if st.button("Regenerate with Feedback"):
                    st.session_state.feedback_text = feedback_text
                    st.session_state.pipeline_result = None
                    st.rerun()
            
            with tabs[4]:
                st.subheader("📅 Content Calendar")
                calendar_data = st.session_state.pipeline_result.get('content_calendar', {})
                if calendar_data and calendar_data.get('calendar'):
                    st.write(f"📋 **{len(calendar_data['calendar'])} content pieces** planned based on your topic and keywords:")
                    
                    for i, entry in enumerate(calendar_data.get('calendar', []), 1):
                        # Create columns for better layout
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {entry['topic']}**")
                            st.write(f"📝 {entry['description']}")
                            
                            # Show keywords if available
                            if entry.get('keywords'):
                                keywords_text = ", ".join(entry['keywords'])
                                st.caption(f"🎯 Keywords: {keywords_text}")
                        
                        with col2:
                            st.markdown(f"**{entry['format']}**")
                            st.write(f"📍 {entry['platform']}")
                            st.write(f"👥 {entry['target_audience']}")
                            if entry.get('estimated_time'):
                                st.caption(f"⏱️ {entry['estimated_time']}")
                        
                        if i < len(calendar_data['calendar']):
                            st.markdown("---")
                else:
                    st.info("📅 Content calendar will be generated based on your article variants.")
        except Exception as e:
            st.error(f"Error displaying content: {e}")

elif st.session_state.current_step == 4:
    st.header("Step 4: Review & Feedback")
    
    if not st.session_state.pipeline_result:
        st.warning("Please generate content in Step 3 first.")
        st.stop()
    
    # Display generated content for review
    st.subheader("📖 Generated Content Review")
    
    variants = st.session_state.pipeline_result.get('variants', [])
    if variants:
        # Variant selection
        variant_options = [f"Variant {i+1}" for i in range(len(variants))]
        selected_variant_index = st.selectbox(
            "Choose content variant to review:",
            range(len(variant_options)),
            format_func=lambda x: variant_options[x],
            key="review_variant_selector"
        )
        
        # Update session state
        st.session_state.selected_variant_index = selected_variant_index
        
        # Show selected variant
        selected_variant = variants[selected_variant_index]
        
        # Article review
        with st.expander("📝 Article Content", expanded=True):
            article = selected_variant['article']
            st.markdown(f"**Topic:** {article.get('topic', 'N/A')}")
            st.markdown(f"**Keywords:** {', '.join(article.get('keywords', []))}")
            st.markdown(f"**Word Count:** {article.get('word_count', 0)} words")
            st.markdown("**Content:**")
            st.write(article.get('content', ''))
        
        # Social media posts review
        with st.expander("📱 Social Media Posts", expanded=True):
            social_posts = selected_variant.get('social_posts', {})
            for platform, post in social_posts.items():
                st.markdown(f"**{platform.title()}:**")
                st.write(post.get('content', ''))
                st.markdown("---")
        
        # Content calendar review
        calendar_data = st.session_state.pipeline_result.get('content_calendar', {})
        if calendar_data and calendar_data.get('calendar'):
            with st.expander("📅 Content Calendar Preview", expanded=False):
                st.write(f"📋 **{len(calendar_data['calendar'])} content pieces** planned:")
                
                for i, entry in enumerate(calendar_data.get('calendar', []), 1):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {entry['topic']}**")
                        st.write(f"📝 {entry['description']}")
                        
                        if entry.get('keywords'):
                            keywords_text = ", ".join(entry['keywords'])
                            st.caption(f"🎯 Keywords: {keywords_text}")
                    
                    with col2:
                        st.markdown(f"**{entry['format']}**")
                        st.write(f"📍 {entry['platform']}")
                        st.write(f"👥 {entry['target_audience']}")
                        if entry.get('estimated_time'):
                            st.caption(f"⏱️ {entry['estimated_time']}")
                    
                    if i < len(calendar_data['calendar']):
                        st.markdown("---")
    
    # Feedback section
    st.markdown("---")
    st.subheader("💬 Provide Feedback")
    
    feedback_text = st.text_area(
        "Share your thoughts on the generated content:",
        value=st.session_state.get('feedback_text', ''),
        height=150,
        placeholder="Example: The tone is too formal, make it more conversational. Add more technical details about the product features. Focus more on benefits rather than features.",
        key="feedback_input"
    )
    
    if st.button("💾 Save Feedback & Continue", type="primary"):
        st.session_state.feedback_text = feedback_text
        if feedback_text.strip():
            # Update memory context with feedback
            memory_update = f"User feedback: {feedback_text.strip()}"
            st.session_state.memory_context = (st.session_state.get('memory_context', '') + "\n" + memory_update).strip()
        
        st.success("✅ Feedback saved! You can now proceed to scheduling.")
        st.info("💡 Use the navigation buttons below to go to Step 5 for scheduling, or go back to Step 3 to regenerate content with your feedback.")

elif st.session_state.current_step == 5:
    st.header("Step 5: Schedule & Publish")
    
    # Check if we just scheduled content - show a refresh button after balloons
    if st.session_state.get('just_scheduled', False):
        st.info("🎉 Content scheduled successfully! Click below to refresh and see your updated schedule.")
        if st.button("🔄 Refresh to View Updated Schedule", type="primary"):
            st.session_state.just_scheduled = False
            st.rerun()
    
    # Add a section to view all scheduled content at the top
    with st.expander("📋 View All Scheduled Content", expanded=False):
        st.subheader("All Scheduled Content")
        try:
            # Always load fresh from file
            file_path = os.path.abspath('articles_data.json')
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as f:
                    all_scheduled = json.load(f)
                    st.session_state.scheduled_articles = all_scheduled  # Update session state
                    
                if all_scheduled:
                    st.write(f"📊 **Total scheduled items:** {len(all_scheduled)}")
                    
                    # Group by date
                    from collections import defaultdict
                    by_date = defaultdict(list)
                    for item in all_scheduled:
                        date = item.get('scheduled_date', 'Unknown')
                        by_date[date].append(item)
                    
                    # Display by date
                    for date in sorted(by_date.keys()):
                        st.markdown(f"**📅 {date}**")
                        items = by_date[date]
                        for item in items:
                            icon = "📝" if item.get('type') == 'article' else "📱"
                            platform = item.get('platform', 'website')
                            preview = (item.get('content', '') or '')[:100] + ('...' if len(item.get('content', '')) > 100 else '')
                            st.write(f"{icon} {platform.title()}: {preview}")
                        st.markdown("---")
                    
                    # Clear all scheduled content button
                    if st.button("🗑️ Clear All Scheduled Content", type="secondary"):
                        try:
                            with open(file_path, 'w') as f:
                                json.dump([], f)
                            st.session_state.scheduled_articles = []
                            st.success("All scheduled content cleared!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing content: {e}")
                else:
                    st.info("No content scheduled yet.")
            else:
                st.info("No scheduled content file found.")
        except Exception as e:
            st.error(f"Error loading scheduled content: {e}")
    
    # Show current settings for transparency
    with st.expander("📋 Current Settings for Scheduling", expanded=False):
        st.write("**Goal:** ", st.session_state.user_inputs.get('goal', 'Not set'))
        st.write("**Language:** ", st.session_state.get('content_language', 'English'))
        
        # SEO report status
        seo_report = st.session_state.user_inputs.get('seo_report', '')
        if seo_report:
            st.write("**SEO Report:** ", f"✅ Used ({len(seo_report):,} chars)")
        else:
            st.write("**SEO Report:** ", "❌ Not used")
            
        st.write("**Selected Variant:** ", f"Variant {st.session_state.get('selected_variant_index', 0) + 1}")
        if st.session_state.pipeline_result:
            variants = st.session_state.pipeline_result.get('variants', [])
            if variants:
                st.write("**Variants Available:** ", len(variants))
    
    # Ensure latest generation is used even when user jumps here
    inputs_signature = (
        st.session_state.user_inputs.get('goal', ''),
        st.session_state.user_inputs.get('keywords', ''),
        st.session_state.user_inputs.get('seo_report', ''),
        st.session_state.get('content_language', 'English'),
        st.session_state.get('content_tone', 'Professional'),
        st.session_state.get('word_count', 800),
        st.session_state.get('feedback_text', ''),
        tuple(st.session_state.user_inputs.get('selected_topics', []) or []),
        st.session_state.get('research_source', 'newsapi'),
        st.session_state.get('require_citations', False),
        st.session_state.get('strict_grounding', False),
    )
    last_signature = st.session_state.get('last_generation_signature')
    if st.session_state.pipeline_result is None or last_signature != inputs_signature:
        # Validate required inputs
        goal = st.session_state.user_inputs.get('goal', '')
        keywords = st.session_state.user_inputs.get('keywords', '')
        
        if not goal or not keywords:
            st.error("❌ Missing required inputs! Please complete Step 1 first.")
            st.write("Go back to Step 1 to set your goal and keywords, then return here.")
        else:
            with st.spinner("Refreshing content for current settings..."):
                try:
                    pipeline = MarketingPipeline(client, RAGManager(), newsapi)
                    rag_manager = RAGManager()
                    kb_context = rag_manager.get_context(goal)
                    
                    result = pipeline.run_pipeline(
                        report_text=st.session_state.user_inputs.get('seo_report', ''),
                        topic=(st.session_state.user_inputs.get('selected_topics') or [goal])[0],
                        manual_keywords=[k.strip() for k in keywords.split(',') if k.strip()],
                        language_label=st.session_state.get('content_language', 'English'),
                        tone=st.session_state.get('content_tone', 'Professional'),
                        word_count=st.session_state.get('word_count', 800),
                        kb_context=(kb_context + "\n\nPrevious session notes:\n" + st.session_state.get('memory_context', '')),
                        feedback=st.session_state.get('feedback_text', ''),
                        variants=4,
                        research_source=st.session_state.get('research_source', 'newsapi'),
                        require_citations=st.session_state.get('require_citations', False),
                        strict_grounding=st.session_state.get('strict_grounding', False),
                        keyword_strategy={
                            'Merge (SEO first)': 'merge',
                            'SEO-only': 'seo-only',
                            'Manual-only': 'manual-only',
                        }.get(st.session_state.get('keyword_strategy', 'Merge (SEO first)'), 'merge'),
                        ai_extract_keywords=st.session_state.get('ai_extract_keywords', False),
                        llm_model=st.session_state.get('llm_model', 'gpt-4o'),
                        llm_temperature=float(st.session_state.get('llm_temperature', 0.4)),
                        enforce_keyword_coverage=st.session_state.get('enforce_keyword_coverage', True)
                    )
                    st.session_state.pipeline_result = result
                    st.session_state.last_generation_signature = inputs_signature
                except Exception as e:
                    st.error(f"Error generating content: {e}")
    
    if st.session_state.pipeline_result:
        with st.form("schedule_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                publish_date = st.date_input(
                    "Select Publication Date",
                    min_value=datetime.now().date()
                )
            
            with col2:
                platforms = st.multiselect(
                    "Select Platforms",
                    ["Website", "LinkedIn", "Instagram", "Facebook"],
                    default=["Website", "LinkedIn"]
                )
            
            if st.form_submit_button("Schedule Content"):
                try:
                    content_items = []
                    
                    # Add selected variant article
                    variants = st.session_state.pipeline_result.get('variants', [])
                    if not variants:
                        st.warning("No variant available to schedule")
                        st.stop()
                    chosen_variant = variants[st.session_state.get('selected_variant_index', 0)]
                    article = chosen_variant['article']
                    content_items.append({
                        'id': article['id'],
                        'type': 'article',
                        'content': article['content']
                    })
                    
                    # Add selected social posts from chosen variant
                    for platform in platforms:
                        if platform.lower() in chosen_variant['social_posts']:
                            post = chosen_variant['social_posts'][platform.lower()]
                            content_items.append({
                                'id': post['id'],
                                'type': 'social',
                                'platform': platform.lower(),
                                'content': post['content']
                            })
                    
                    # Schedule content
                    if content_items:
                        pipeline = MarketingPipeline(client, RAGManager(), newsapi)
                        scheduled = pipeline.content_manager().schedule_content_group(
                            content_items,
                            publish_date.strftime('%Y-%m-%d')
                        )
                        if scheduled:
                            st.success(f"✅ Successfully scheduled {len(scheduled)} content items for {publish_date}!")
                            st.balloons()
                            # Refresh scheduled articles in session
                            try:
                                file_path = os.path.abspath('articles_data.json')
                                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                    with open(file_path, 'r') as f:
                                        st.session_state.scheduled_articles = json.load(f)
                            except Exception:
                                pass
                            st.info("💡 Expand 'View All Scheduled Content' above to see your complete schedule!")
                            # Set a flag to show we just scheduled content
                            st.session_state.just_scheduled = True
                        else:
                            st.error("❌ Failed to schedule content")
                    else:
                        st.warning("No content selected for scheduling")
                        
                except Exception as e:
                    st.error(f"❌ Error scheduling content: {e}")
                    import traceback
                    with st.expander("🔍 Debug Information", expanded=False):
                        st.code(traceback.format_exc())

        # Show scheduled content list
        st.markdown("---")
        st.subheader("Scheduled Content")
        scheduled = st.session_state.get('scheduled_articles', []) or []
        if scheduled:
            # Normalize for display
            rows = []
            for item in scheduled:
                rows.append({
                    'Scheduled Date': item.get('scheduled_date', ''),
                    'Type': item.get('type', ''),
                    'Platform': item.get('platform', 'website' if item.get('type') == 'article' else ''),
                    'Content Preview': (item.get('content', '') or '')[:160] + ('…' if len(item.get('content', '')) > 160 else '')
                })
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No content scheduled yet.")

# Navigation Buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.session_state.current_step > 1:
        if st.button("← Previous Step"):
            st.session_state.current_step -= 1
            st.rerun()
with col3:
    # Only show Next Step button for steps that don't have form-based navigation
    if st.session_state.current_step < len(steps) and st.session_state.current_step not in [1, 2]:
        if st.button("Next Step →"):
            st.session_state.current_step += 1
            st.rerun()
    elif st.session_state.current_step == 2:
        # Step 2 has its own form navigation, but add a manual next button for convenience
        if st.button("Skip to Content Generation →"):
            st.session_state.current_step = 3
            st.rerun()