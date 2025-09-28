# 🎯 AI Marketing Planner

A comprehensive, AI-powered marketing content generation platform that creates professional articles and social media posts using real-time data, SEO insights, knowledge base integration, and advanced grounding techniques.

## ✨ Key Features

### 🤖 **Advanced Content Generation**
- **4 Article Variants**: Generate multiple versions with A/B testing support
- **Platform-Specific Social Posts**: Tailored content for LinkedIn, Twitter, Instagram, and Facebook
- **Multi-Language Support**: English, Spanish, and French with automatic translation
- **Tone Customization**: Professional, Casual, Technical, Friendly, or Formal
- **Word Count Control**: Precise targeting from 300-3000 words
- **LLM Model Selection**: Choose between GPT-4o, GPT-4o-mini, or custom models
- **Creativity Control**: Adjustable temperature slider for accuracy vs creativity balance

### 🎯 **Enhanced Keyword Management**
- **Smart SEO Parsing**: Extract keywords from `keyword:` and `keywords:` lines in reports
- **AI Keyword Extraction**: LLM-powered extraction from unstructured SEO text
- **Keyword Strategy Control**: SEO-only, Manual-only, or Merge (SEO first)
- **Verbatim Enforcement**: Ensures all required keywords appear at least once
- **Revision Pass**: Automatically adds missing keywords with secondary LLM call
- **Real-time Preview**: Shows extracted keywords instead of generic terms

### 🔍 **Research & Grounding System**
- **Multi-Source Research**: NewsAPI, Google Search (SerpAPI/Custom Search), or both
- **Citation Support**: Bracketed [1], [2] style citations with References section
- **Strict Grounding Mode**: Lower temperature + "avoid unsupported claims" instruction
- **Source Verification**: Links citations to actual web sources and news articles
- **Hallucination Reduction**: Fact-checking with external data validation

### 💾 **Memory & Persistence**
- **Session Memory**: Carries context between generations for consistency
- **Knowledge Base History**: View and reuse previously added guidelines
- **SEO Report Library**: Save and select from past SEO analysis reports
- **Topic Selection**: Choose from extracted topics or add custom ones
- **Persistent Settings**: Maintains preferences across sessions

### 📊 **Data-Driven Intelligence**
- **Real-Time News Integration**: Incorporates current trends via NewsAPI
- **Google Search Integration**: Web research with SerpAPI or Custom Search
- **SEO Analysis**: Upload and analyze reports for keyword optimization
- **Knowledge Base**: FAISS-powered vector storage for brand guidelines
- **Competitive Intelligence**: Extract insights from competitor analysis

### 🎨 **Professional User Interface**
- **Interactive Progress Tracking**: Visual step-by-step workflow
- **Enhanced Content Preview**: Variant comparison with metrics and copy buttons
- **Smart Error Handling**: Comprehensive troubleshooting with API status
- **Research Preview**: Shows extracted keywords and citation count
- **Content Calendar**: Visual scheduling with multiple content types

### 📅 **Enhanced Content Calendar**
- **Multi-Content Planning**: Blog posts, social media, and follow-up articles
- **Keyword-Based Suggestions**: Additional content ideas from SEO keywords
- **Reading Time Estimates**: Calculated based on word count
- **Platform-Specific Metrics**: Character counts and engagement estimates
- **Strategic Layout**: Two-column design with descriptions and metadata

## 🛠 **Technical Architecture**

- **Frontend**: Streamlit with custom CSS styling and enhanced UX
- **AI Engine**: Configurable OpenAI models (GPT-4o, GPT-4o-mini, custom)
- **Research Sources**: NewsAPI + Google Search (SerpAPI/Custom Search)
- **Vector Storage**: FAISS with LangChain for persistent knowledge management
- **Memory System**: Session-based context retention across generations
- **Citation Engine**: Automatic reference linking and fact verification

## 📋 **Prerequisites**

- Python 3.8+
- OpenAI API Key (Required)
- NewsAPI Key (Optional but recommended for news grounding)
- SerpAPI Key or Google Custom Search (Optional for web research)

## 🚀 **Quick Start**

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/ai-marketing-planner.git
cd ai-marketing-planner
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```bash
# Required
OPENAI_API_KEY=sk-your_openai_api_key_here

# Optional (for enhanced research and grounding)
NEWS_API_KEY=your_newsapi_key_here
SERPAPI_API_KEY=your_serpapi_key_here

# Alternative to SerpAPI (requires both keys)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id
```

### 3. Launch Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📖 **Complete Workflow Guide**

### Step 1: Goals & Keywords 🎯
- **Marketing Goal**: Define your specific objective
- **Target Keywords**: Enter comma-separated keywords (or leave empty for SEO-only mode)
- **Audience**: Select target demographic
- **Tone**: Choose content voice and style

### Step 2: Content Strategy 📝
- **Content Types**: Select Blog Posts, Social Media, etc.
- **Platforms**: Choose LinkedIn, Instagram, Facebook, Twitter
- **Topic Selection**: Pick from auto-generated topics or add custom ones
- **SEO Analysis**: 
  - Upload reports or paste text with `keyword:` lines
  - AI keyword extraction available
  - Real-time preview of extracted keywords
  - Load from previously saved reports

### Step 3: Content Generation 🤖
- **Enhanced Settings**:
  - LLM Model: GPT-4o (default), GPT-4o-mini, or custom
  - Creativity slider: 0.0-1.0 (lower = more accurate)
  - Keyword enforcement with revision pass
  - Research source: NewsAPI, Google, Both, or None
  - Citation requirements and strict grounding
- **4 Variant System**: Choose from multiple article versions
- **Platform-Specific Posts**: Optimized for each social network
- **Research Integration**: Citations and references included
- **Interactive Metrics**: SEO keywords, news articles, citations count
- **Feedback Loop**: Refine content with custom suggestions

### Step 4: Schedule & Publish 📅
- **Enhanced Calendar**: Multiple content types with metadata
- **Platform Selection**: Choose publication channels
- **Batch Scheduling**: Articles and social posts together
- **Content Suggestions**: Follow-up articles based on keywords

## 🔧 **Advanced Configuration**

### Model & Creativity Settings
```
Sidebar → Content Settings:
- LLM Model: gpt-4o (recommended), gpt-4o-mini, or custom
- Creativity: 0.2-0.4 for accuracy, 0.6-0.8 for creativity
- Keyword Strategy: SEO-only, Manual-only, or Merge
- AI Keyword Extraction: Enable for unstructured SEO text
- Enforce Keyword Coverage: Ensures verbatim keyword inclusion
```

### Research & Grounding
```
Sidebar → Research & Grounding:
- External Research Source: NewsAPI, Google, Both, or None
- Require Citations: Adds [1], [2] style references
- Strict Grounding: Lower creativity + fact-checking prompts
```

### Knowledge Base Management 📚
- Add brand guidelines and content rules
- FAISS vector search for relevant context
- View and reuse previously added guidelines
- Persistent storage across sessions

## 🔑 **API Key Setup Guide**

### OpenAI API Key (Required)
1. Visit [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Create account and generate API key
3. Copy key (starts with `sk-`)
4. Add to `.env` file

### NewsAPI Key (Optional - News Research)
1. Go to [NewsAPI.org](https://newsapi.org/)
2. Register for free account (100 requests/day)
3. Get API key from dashboard
4. Add to `.env` file

### SerpAPI Key (Optional - Google Search)
1. Visit [SerpAPI](https://serpapi.com/)
2. Sign up for account (100 searches/month free)
3. Get API key from dashboard
4. Add `SERPAPI_API_KEY` to `.env`

### Google Custom Search (Alternative to SerpAPI)
1. Create project at [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Custom Search API
3. Create Custom Search Engine at [cse.google.com](https://cse.google.com/)
4. Add both `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` to `.env`

## 🎯 **Usage Examples**

### SEO-First Workflow
```
1. Set Keyword Strategy: "SEO-only"
2. Enable "AI keyword extraction"
3. Paste SEO report with keyword: lines
4. Set model to gpt-4o, creativity 0.3
5. Enable "Require citations" + "Strict grounding"
6. Generate with external research
```

### Creative Content Workflow
```
1. Set Keyword Strategy: "Manual-only" 
2. Add creative keywords in Step 1
3. Set creativity to 0.7-0.8
4. Use NewsAPI for trending topics
5. Add brand voice to Knowledge Base
6. Generate variants and refine with feedback
```

## 🐛 **Troubleshooting**

### Common Issues
- **No Keywords Extracted**: Use explicit `keyword:` format in SEO reports
- **Hallucinated Content**: Enable strict grounding + citations
- **Missing Citations**: Check Google/SerpAPI keys and research source setting
- **Keyword Not Included**: Enable "Enforce keyword coverage"
- **Generic Content**: Lower creativity, add Knowledge Base guidelines

### Debug Features
- **API Status Indicators**: Shows OpenAI/NewsAPI/Google connectivity
- **Keyword Preview**: Displays extracted keywords in Step 2
- **Citation Count**: Shows research sources found
- **Error Messages**: Specific guidance for each issue

## 📊 **Performance & Limits**

### Content Generation
- **Article Length**: 300-3000 words with revision pass
- **Variants**: 4 per generation with keyword enforcement
- **Social Platforms**: 4 with platform-specific optimization
- **Research Sources**: Up to 10 articles per source
- **Citations**: Up to 10 references per article

### API Usage (per generation)
- **OpenAI**: ~3000-6000 tokens (article + revision + social posts)
- **NewsAPI**: 10 articles per search
- **Google Search**: 10 results per search
- **Combined Research**: Merged and deduplicated results

## 🤝 **Contributing**

### Development Setup
```bash
# Clone and setup
git clone <repo-url>
cd ai-marketing-planner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env

# Run in development mode
streamlit run streamlit_app.py --logger.level=debug
```

### Feature Development
1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Update README if needed
5. Submit pull request

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🔮 **Recent Updates**

### v2.0 - Enhanced Intelligence
- ✅ Advanced keyword management with AI extraction
- ✅ Multi-source research with citations
- ✅ Session memory and persistent storage
- ✅ Enhanced content calendar
- ✅ Strict grounding and fact-checking
- ✅ Custom model support with creativity control

### Roadmap
- [ ] PDF text extraction for SEO reports
- [ ] Direct social media publishing integration
- [ ] Content performance analytics
- [ ] Team collaboration features
- [ ] Advanced brand voice training
- [ ] CMS platform integrations

---

**Built with ❤️ using OpenAI GPT-4o, Streamlit, FAISS, and modern Python technologies.**

*Reduces hallucinations. Enforces keywords. Grounds content with real research.*