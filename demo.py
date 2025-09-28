import os
from dotenv import load_dotenv
from openai import OpenAI
from newsapi import NewsApiClient
from marketing_pipeline import MarketingPipeline
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

class MarketingPlannerDemo:
    def __init__(self):
        # Initialize clients
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.newsapi_client = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        
        # Initialize RAG Manager
        self.embeddings = OpenAIEmbeddings()
        self.rag_manager = self._initialize_rag_manager()

    def _initialize_rag_manager(self):
        """Initialize or load existing knowledge base"""
        try:
            # Try to load existing knowledge base
            vectorstore = FAISS.load_local("knowledge_base", self.embeddings)
        except:
            # If no existing knowledge base, create a new one
            vectorstore = FAISS.from_texts(["Initial marketing knowledge base"], self.embeddings)
            vectorstore.save_local("knowledge_base")
        return vectorstore

    def demonstrate_workflow(self):
        """
        Demonstrate the full marketing content generation workflow
        """
        print("🚀 AI Marketing Planner Demonstration")
        print("=====================================")

        # Scenario: Eco-friendly product marketing campaign
        marketing_goal = "Increase brand awareness for eco-friendly tech products"
        keywords = ["sustainable technology", "green innovation", "eco-tech"]
        
        # Optional SEO report
        seo_report = """
        gap: Lack of content on sustainable tech innovations
        competitor: Tech companies not focusing on environmental impact
        keyword: green technology trends
        """

        # Initialize Marketing Pipeline
        pipeline = MarketingPipeline(
            client=self.openai_client, 
            rag_manager=self.rag_manager, 
            newsapi=self.newsapi_client
        )

        # Run full content generation pipeline
        print("\n📊 Analyzing SEO Insights...")
        result = pipeline.run_pipeline(
            report_text=seo_report,
            topic=marketing_goal,
            manual_keywords=keywords,
            language_label='English'
        )

        # Display Results
        print("\n📝 Generated Article:")
        print(result['article']['content'][:500] + "...\n")

        print("🌐 Social Media Posts:")
        for platform, post in result['social_posts'].items():
            print(f"{platform.upper()} Post: {post['content']}\n")

        print("📅 Content Calendar:")
        for entry in result['content_calendar']['calendar']:
            print(f"- {entry['topic']} on {entry['platform']}")

        print("\n🔍 SEO Insights:")
        print(f"Keywords: {result['seo_insights'].get('keywords', [])}")
        print(f"Content Gaps: {result['seo_insights'].get('content_gaps', [])}")

if __name__ == "__main__":
    demo = MarketingPlannerDemo()
    demo.demonstrate_workflow()
