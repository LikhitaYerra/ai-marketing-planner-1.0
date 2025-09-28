import os
import json
from datetime import datetime, timedelta
import uuid
import re
import requests

class MarketingPipeline:
    SUPPORTED_LANGUAGES = {
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr'
    }

    def __init__(self, client, rag_manager, newsapi):
        """
        Initialize the Marketing Pipeline with AI and data services
        
        :param client: OpenAI client
        :param rag_manager: Retrieval Augmented Generation manager
        :param newsapi: NewsAPI client
        """
        self.client = client
        self.rag_manager = rag_manager
        self.newsapi = newsapi
        self.content_storage_path = 'articles_data.json'

    def analyze_seo_report(self, report_text):
        """
        Extract key SEO insights from the report
        
        :param report_text: SEO report or competitor analysis text
        :return: Dictionary of SEO insights
        """
        if not report_text:
            return {
                'keywords': [],
                'content_gaps': [],
                'competitor_insights': []
            }
        
        # Prefer explicit lines like: keyword: ..., keywords: ... (comma-separated allowed)
        explicit_keywords = []
        for line in report_text.splitlines():
            m = re.match(r"\s*(keywords?|keyphrases?)\s*:\s*(.+)", line, flags=re.IGNORECASE)
            if m:
                raw = m.group(2).strip()
                # split by comma/semicolon; if none, treat entire raw as one keyword
                parts = re.split(r"[,;]", raw) if ("," in raw or ";" in raw) else [raw]
                for p in parts:
                    kw = p.strip().strip("-•·: ")
                    if kw:
                        explicit_keywords.append(kw)
        
        # Fallback light extraction of capitalized or 2-3 word phrases (kept after explicit ones)
        fallback_candidates = re.findall(r"[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,2}", report_text)
        fallback_keywords = []
        for cand in fallback_candidates:
            token_count = len(cand.split())
            if 1 <= token_count <= 4 and len(cand) >= 4:
                fallback_keywords.append(cand.strip())
        
        # Merge while preserving order and uniqueness, prefer explicit
        keywords = self._merge_keywords(explicit_keywords, fallback_keywords)

        content_gaps = re.findall(r'gap:\s*(.+)', report_text, re.IGNORECASE)
        competitor_insights = re.findall(r'competitor:\s*(.+)', report_text, re.IGNORECASE)
        
        return {
            'keywords': (keywords[:10]),  # keep order, limit to 10
            'content_gaps': content_gaps[:5],     # Top content gaps
            'competitor_insights': competitor_insights[:5]  # Top competitor insights
        }

    def _merge_keywords(self, primary_list, secondary_list):
        """Merge two keyword lists preserving order and uniqueness, preferring primary order."""
        seen = set()
        merged = []
        for kw in (primary_list or []):
            norm = kw.strip()
            if norm and norm.lower() not in seen:
                merged.append(norm)
                seen.add(norm.lower())
        for kw in (secondary_list or []):
            norm = kw.strip()
            if norm and norm.lower() not in seen:
                merged.append(norm)
                seen.add(norm.lower())
        return merged

    def _ai_extract_keywords(self, report_text: str, max_keywords: int = 10, llm_model: str = "gpt-4o", llm_temperature: float = 0.2):
        """
        Use the LLM to extract prioritized keywords from the SEO report.
        Returns a list of strings; falls back to [] on failure.
        """
        if not report_text or not isinstance(report_text, str):
            return []
        try:
            system_prompt = (
                "You extract SEO keywords and keyphrases from reports. "
                "Return strict JSON: {\"keywords\":[\"...\"]}. No commentary."
            )
            user_prompt = (
                "From the following SEO analysis, extract up to " + str(max_keywords) +
                " prioritized search keywords or keyphrases (1-4 words each). \n\n" + report_text
            )
            resp = self.client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(0.0, min(1.0, llm_temperature)),
                max_tokens=300,
            )
            content = resp.choices[0].message.content or ""
            # Strip code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("` ")
                # If language tag included, remove first line
                lines = content.splitlines()
                if lines:
                    if ':' in lines[0] or lines[0].strip() and not lines[0].strip().startswith('{'):
                        content = "\n".join(lines[1:])
            # Parse JSON
            data = json.loads(content)
            kws = [k.strip() for k in (data.get('keywords') or []) if isinstance(k, str) and k.strip()]
            return kws[:max_keywords]
        except Exception:
            return []

    def fetch_relevant_news(self, topic, keywords=None, language='en', days_back=7):
        """
        Fetch relevant news articles using NewsAPI
        
        :param topic: Main topic to search for
        :param keywords: Additional keywords to include
        :param language: Language code (en, es, fr, etc.)
        :param days_back: How many days back to search
        :return: Dictionary with news insights
        """
        try:
            from datetime import datetime, timedelta
            
            # Build search query
            search_terms = [topic]
            if keywords:
                search_terms.extend(keywords[:3])  # Add top 3 keywords
            
            query = ' OR '.join([f'"{term}"' for term in search_terms])
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Fetch news
            news_response = self.newsapi.get_everything(
                q=query,
                language=language,
                sort_by='relevancy',
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                page_size=10  # Limit to top 10 articles
            )
            
            if news_response['status'] == 'ok' and news_response['articles']:
                articles = news_response['articles']
                
                # Extract key insights
                news_insights = {
                    'total_articles': len(articles),
                    'trending_topics': [],
                    'recent_developments': [],
                    'key_sources': list(set([article.get('source', {}).get('name', 'Unknown') for article in articles[:5]])),
                    'sample_headlines': [article['title'] for article in articles[:5]],
                    'citations': []
                }
                
                # Extract trending topics from titles and descriptions
                all_text = ' '.join([
                    (article.get('title', '') + ' ' + article.get('description', ''))
                    for article in articles
                ])
                
                # Simple trending topic extraction
                words = all_text.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 4 and word.isalpha():  # Filter meaningful words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top trending words
                trending = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                news_insights['trending_topics'] = [word for word, count in trending]
                
                # Recent developments from descriptions
                news_insights['recent_developments'] = [
                    article.get('description', '')[:100] + '...' 
                    for article in articles[:3] 
                    if article.get('description')
                ]
                
                return news_insights
            
        except Exception as e:
            print(f"NewsAPI fetch failed: {e}")
        
        # Return empty insights if failed
        return {
            'total_articles': 0,
            'trending_topics': [],
            'recent_developments': [],
            'key_sources': [],
            'sample_headlines': [],
            'citations': []
        }

    def fetch_google_research(self, topic, keywords=None, language='en', num_results=10):
        """
        Fetch web research via SerpAPI (preferred) or Google Custom Search as fallback.
        Returns a structure compatible with news_insights and includes citations.
        """
        query_terms = [topic]
        if keywords:
            query_terms.extend(keywords[:3])
        query = ' '.join(query_terms)

        serpapi_key = os.getenv('SERPAPI_API_KEY')
        google_key = os.getenv('GOOGLE_API_KEY')
        google_cx = os.getenv('GOOGLE_CSE_ID')

        try:
            results = []
            if serpapi_key:
                params = {
                    'engine': 'google',
                    'q': query,
                    'hl': language,
                    'num': num_results,
                    'api_key': serpapi_key,
                }
                resp = requests.get('https://serpapi.com/search.json', params=params, timeout=20)
                data = resp.json()
                for item in data.get('organic_results', [])[:num_results]:
                    results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', ''),
                        'source': item.get('displayed_link', ''),
                    })
            elif google_key and google_cx:
                params = {
                    'key': google_key,
                    'cx': google_cx,
                    'q': query,
                    'num': min(10, num_results),
                }
                resp = requests.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=20)
                data = resp.json()
                for item in data.get('items', [])[:num_results]:
                    results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', ''),
                        'source': item.get('displayLink', ''),
                    })
            else:
                return {
                    'total_articles': 0,
                    'trending_topics': [],
                    'recent_developments': [],
                    'key_sources': [],
                    'sample_headlines': [],
                    'citations': []
                }

            if not results:
                return {
                    'total_articles': 0,
                    'trending_topics': [],
                    'recent_developments': [],
                    'key_sources': [],
                    'sample_headlines': [],
                    'citations': []
                }

            # Aggregate insights
            all_text = ' '.join([(r.get('title', '') + ' ' + r.get('snippet', '')) for r in results])
            words = all_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            trending = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

            citations = []
            key_sources = []
            sample_headlines = []
            recent_developments = []
            for idx, r in enumerate(results[:5]):
                citations.append({'index': idx + 1, 'title': r['title'], 'url': r['link'], 'source': r['source']})
                key_sources.append(r['source'])
                sample_headlines.append(r['title'])
                if r.get('snippet'):
                    recent_developments.append(r['snippet'][:100] + '...')

            return {
                'total_articles': len(results),
                'trending_topics': [w for w, _ in trending],
                'recent_developments': recent_developments,
                'key_sources': list(dict.fromkeys(key_sources)),
                'sample_headlines': sample_headlines,
                'citations': citations,
            }
        except Exception as e:
            print(f"Google research fetch failed: {e}")
            return {
                'total_articles': 0,
                'trending_topics': [],
                'recent_developments': [],
                'key_sources': [],
                'sample_headlines': [],
                'citations': []
            }

    def _merge_research_insights(self, a, b):
        """Merge two research insight dicts, deduplicating values and concatenating citations."""
        if not a:
            return b or {}
        if not b:
            return a or {}
        merged = {
            'total_articles': (a.get('total_articles', 0) + b.get('total_articles', 0)),
            'trending_topics': list(dict.fromkeys((a.get('trending_topics') or []) + (b.get('trending_topics') or [])))[:5],
            'recent_developments': ((a.get('recent_developments') or []) + (b.get('recent_developments') or []))[:6],
            'key_sources': list(dict.fromkeys((a.get('key_sources') or []) + (b.get('key_sources') or [])))[:6],
            'sample_headlines': ((a.get('sample_headlines') or []) + (b.get('sample_headlines') or []))[:6],
            'citations': ((a.get('citations') or []) + (b.get('citations') or []))[:10],
        }
        return merged

    def generate_article(self, topic, keywords, seo_insights=None, language='English', tone='Professional', word_count=800, kb_context: str = "", feedback: str = "", news_insights=None, research_prefs=None, llm_model: str = "gpt-4o", enforce_keyword_coverage: bool = True, llm_temperature: float = 0.4):
        """
        Generate a marketing article using LLM, incorporating SEO insights and KB context
        
        :param topic: Marketing goal or topic
        :param keywords: List of target keywords
        :param seo_insights: SEO analysis insights
        :param language: Content language
        :param tone: Content tone
        :param word_count: Target word count
        :return: Dictionary with article details
        """
        # Build prompts with better structure. Keep order: prefer SEO keywords, then provided keywords
        merged_keywords = self._merge_keywords((seo_insights.get('keywords', []) if seo_insights else []), (keywords or []))
        
        system_prompt = (
            f"You are a senior marketing content writer. Create a comprehensive article in {language} "
            f"with a {tone} tone. Target approximately {word_count} words. "
            f"Structure the article with clear headings, subheadings, and actionable insights. "
            f"Incorporate SEO best practices and use the provided context effectively."
        )
        
        context_parts = []
        if seo_insights:
            seo_formatted = self._format_seo_insights(seo_insights)
            if seo_formatted.strip():
                context_parts.append(f"SEO Analysis:\n{seo_formatted}")
        
        if news_insights and news_insights.get('total_articles', 0) > 0:
            news_formatted = self._format_news_insights(news_insights)
            if news_formatted.strip():
                context_parts.append(f"Current News & Trends:\n{news_formatted}")
        
        if kb_context and kb_context.strip():
            context_parts.append(f"Knowledge Base Guidelines:\n{kb_context}")
        
        if feedback and feedback.strip():
            context_parts.append(f"User Feedback to Address:\n{feedback}")
        
        context_section = "\n\n".join(context_parts) if context_parts else "No additional context provided."
        
        # Prepare required keywords section for stronger guidance
        required_keywords_block = "\n".join([f"- {kw}" for kw in merged_keywords]) if merged_keywords else "- (none)"

        user_prompt = (
            f"Write a marketing article about: {topic}\n\n"
            f"Target Keywords: {', '.join(merged_keywords)}\n\n"
            f"Context Information:\n{context_section}\n\n"
            f"Requirements:\n"
            f"- Write approximately {word_count} words\n"
            f"- Use {tone.lower()} tone throughout\n"
            f"- Include relevant headings and subheadings\n"
            f"- Incorporate the target keywords naturally\n"
            f"- Provide actionable insights and practical advice\n"
            f"- Make it engaging and valuable for readers\n"
            f"- Do not fabricate facts. When uncertain, state that information is not available.\n\n"
            f"Required Keywords (verbatim; include EACH at least once, use as subheading/bullet if needed):\n"
            f"{required_keywords_block}"
        )

        try:
            completion = self.client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt + (" Always avoid unsupported claims. If unsure, say you are unsure." if (research_prefs or {}).get('strict_grounding') else "")},
                    {"role": "user", "content": user_prompt + ("\n\nIf you cite external facts, include bracketed numeric citations like [1], [2] that correspond to the References list provided in Context Information. Do not fabricate citations." if (research_prefs or {}).get('require_citations') and (news_insights or {}).get('citations') else "")},
                ],
                temperature=0.2 if (research_prefs or {}).get('strict_grounding') else max(0.0, min(1.0, llm_temperature)),
                max_tokens=min(4000, word_count * 2),  # Reasonable token limit
            )
            article_content = completion.choices[0].message.content
        except Exception as e:
            # Better fallback with more structure
            print(f"LLM article generation failed: {e}")
            article_content = self._create_fallback_article(topic, merged_keywords, seo_insights, tone, word_count, kb_context, feedback, news_insights)
        
        # Optional revise pass to enforce keyword coverage verbatim
        if enforce_keyword_coverage and merged_keywords:
            missing = [kw for kw in merged_keywords if kw.lower() not in article_content.lower()]
            if missing:
                try:
                    revise_system = (
                        "You are an editor. Revise the user's article to include ALL required keywords verbatim. "
                        "Keep tone, structure, and meaning. Do not add new claims. Return only the revised article."
                    )
                    revise_user = (
                        "Required keywords (must appear verbatim at least once each):\n- " + "\n- ".join(missing) +
                        "\n\nArticle to revise:\n" + article_content
                    )
                    rev = self.client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": revise_system},
                            {"role": "user", "content": revise_user},
                        ],
                        temperature=0.2,
                        max_tokens=min(4000, word_count * 2),
                    )
                    revised = rev.choices[0].message.content
                    if isinstance(revised, str) and revised.strip():
                        article_content = revised
                except Exception:
                    pass
        
        return {
            'id': str(uuid.uuid4()),
            'content': article_content.strip(),
            'topic': topic,
            'keywords': merged_keywords,  # Use merged keywords
            'language': language,
            'tone': tone,
            'seo_insights': seo_insights or {},
            'word_count': len(article_content.split()),
            'kb_context_used': bool(kb_context and kb_context.strip()),
            'feedback_applied': bool(feedback and feedback.strip())
        }

    def _format_seo_insights(self, seo_insights):
        """
        Format SEO insights for the article
        
        :param seo_insights: Dictionary of SEO insights
        :return: Formatted insights string
        """
        insights_str = ""
        
        if seo_insights.get('keywords'):
            insights_str += f"### Target Keywords\n{', '.join(seo_insights['keywords'])}\n\n"
        
        if seo_insights.get('content_gaps'):
            insights_str += "### Content Gaps\n"
            for gap in seo_insights['content_gaps']:
                insights_str += f"- {gap}\n"
            insights_str += "\n"
        
        if seo_insights.get('competitor_insights'):
            insights_str += "### Competitor Insights\n"
            for insight in seo_insights['competitor_insights']:
                insights_str += f"- {insight}\n"
        
        return insights_str

    def _format_news_insights(self, news_insights):
        """
        Format news insights for the article
        
        :param news_insights: Dictionary of news insights
        :return: Formatted news string
        """
        if not news_insights or news_insights.get('total_articles', 0) == 0:
            return ""
        
        insights_str = ""
        
        if news_insights.get('trending_topics'):
            insights_str += f"### Trending Topics\n{', '.join(news_insights['trending_topics'][:5])}\n\n"
        
        if news_insights.get('recent_developments'):
            insights_str += "### Recent Developments\n"
            for dev in news_insights['recent_developments'][:3]:
                insights_str += f"- {dev}\n"
            insights_str += "\n"
        
        if news_insights.get('key_sources'):
            insights_str += f"### Key News Sources\n{', '.join(news_insights['key_sources'][:3])}\n\n"
        
        if news_insights.get('sample_headlines'):
            insights_str += "### Recent Headlines\n"
            for headline in news_insights['sample_headlines'][:3]:
                insights_str += f"- {headline}\n"
        
        # Optional references for grounding
        citations = news_insights.get('citations') or []
        if citations:
            insights_str += "\n### References\n"
            for idx, c in enumerate(citations, 1):
                title = c.get('title', 'Untitled')
                url = c.get('url', '')
                source = c.get('source', '')
                insights_str += f"[{idx}] {title} — {source} — {url}\n"
        
        return insights_str

    def _generate_content_sections(self, topic, keywords, word_count):
        """
        Generate content sections based on topic and keywords
        
        :param topic: Main topic
        :param keywords: List of keywords
        :param word_count: Target word count
        :return: Generated content sections
        """
        # Simple content generation logic
        sections = [
            f"### {keyword.title()} Exploration" 
            for keyword in keywords[:3]  # Create sections for top 3 keywords
        ]
        
        # Distribute word count across sections
        section_word_count = word_count // len(sections) if sections else word_count
        
        content = "\n\n".join([
            f"{section}\n\n"
            f"This section delves into the nuances of {topic} "
            f"with a focus on {keyword}. We explore its significance, "
            f"impact, and strategic implications in the current market landscape."
            for section, keyword in zip(sections, keywords[:3])
        ])
        
        return content

    def _create_fallback_article(self, topic, keywords, seo_insights, tone, word_count, kb_context, feedback, news_insights=None):
        """Create a structured fallback article when LLM fails"""
        sections = []
        sections.append(f"# {topic}")
        
        if seo_insights:
            sections.append("## SEO Insights")
            sections.append(self._format_seo_insights(seo_insights))
        
        if news_insights and news_insights.get('total_articles', 0) > 0:
            sections.append("## Current News & Trends")
            sections.append(self._format_news_insights(news_insights))
        
        sections.append("## Overview")
        sections.append(f"This comprehensive guide explores {topic} with a focus on {tone.lower()} insights and practical applications.")
        
        if keywords:
            sections.append("## Key Areas of Focus")
            for i, keyword in enumerate(keywords[:5], 1):
                sections.append(f"### {i}. {keyword.title()}")
                sections.append(f"Understanding {keyword} is crucial for {topic}. This section provides detailed insights and actionable strategies.")
        
        if kb_context and kb_context.strip():
            sections.append("## Guidelines and Best Practices")
            sections.append(kb_context[:500] + ("..." if len(kb_context) > 500 else ""))
        
        if feedback and feedback.strip():
            sections.append("## Addressing Your Requirements")
            sections.append(f"Based on your feedback: {feedback}")
        
        sections.append("## Conclusion")
        sections.append(f"This guide provides a comprehensive overview of {topic}. Apply these insights to achieve your marketing goals effectively.")
        
        return "\n\n".join(sections)

    def generate_social_posts(self, article, language_label='English', tone='Professional', platforms=['LinkedIn', 'Twitter', 'Instagram', 'Facebook'], llm_model: str = "gpt-4o", llm_temperature: float = 0.6):
        """
        Generate social media posts based on the article
        
        :param article: Generated article dictionary
        :param language_label: Target language label
        :param tone: Desired tone
        :param platforms: List of social media platforms
        :return: Dictionary of social media posts
        """
        social_posts = {}
        for platform in platforms:
            try:
                generated = self._generate_social_post(
                    platform=platform,
                    topic=article['topic'],
                    keywords=article.get('keywords', []),
                    tone=tone,
                    language_label=language_label,
                    llm_model=llm_model,
                    llm_temperature=llm_temperature,
                )
            except Exception:
                generated = None

            if not generated:
                # Fallback simple content
                fallback_hashtags = {
                    'linkedin': '#Marketing #ContentStrategy',
                    'twitter': '#Marketing #ContentStrategy',
                    'instagram': '#marketing #contentstrategy #growth',
                    'facebook': '#Marketing #ContentStrategy'
                }
                generated = (
                    f"🚀 {article['topic']} — key takeaways. "
                    f"Keywords: {', '.join(article.get('keywords', [])[:5])}. "
                    f"{fallback_hashtags.get(platform.lower(), '#Marketing')}"
                )

            social_posts[platform.lower()] = {
                'id': str(uuid.uuid4()),
                'content': generated,
                'platform': platform.lower()
            }
        
        return social_posts

    def run_pipeline(self, report_text='', topic='', manual_keywords=None, language_label='English', tone='Professional', word_count=800, kb_context: str = "", feedback: str = "", variants: int = 1, research_source: str = 'newsapi', require_citations: bool = False, strict_grounding: bool = False, keyword_strategy: str = 'merge', ai_extract_keywords: bool = False, llm_model: str = 'gpt-4o', llm_temperature: float = 0.4, enforce_keyword_coverage: bool = True):
        """
        Full marketing content generation pipeline
        
        :param report_text: Optional SEO or competitor report
        :param topic: Marketing goal or topic
        :param manual_keywords: List of manual keywords
        :param language_label: Content language
        :return: Comprehensive content generation result
        """
        # Analyze SEO report
        seo_insights = self.analyze_seo_report(report_text)
        
        # Optionally enhance SEO keywords with AI extraction
        if ai_extract_keywords:
            ai_kws = self._ai_extract_keywords(report_text or "", max_keywords=15, llm_model=llm_model, llm_temperature=max(0.0, min(1.0, llm_temperature)))
            if ai_kws:
                seo_insights['keywords'] = ai_kws

        # Merge keywords (SEO preferred) for consistent downstream usage according to strategy
        strategy = (keyword_strategy or 'merge').lower()
        if strategy == 'seo-only':
            merged_keywords = (seo_insights.get('keywords', []) or [])
        elif strategy == 'manual-only':
            merged_keywords = (manual_keywords or [])
        else:
            merged_keywords = self._merge_keywords(seo_insights.get('keywords', []), (manual_keywords or []))

        # Fetch external research insights based on selected source
        lang_code = self.SUPPORTED_LANGUAGES.get(language_label, 'en')
        research_insights = {'total_articles': 0, 'trending_topics': [], 'recent_developments': [], 'key_sources': [], 'sample_headlines': [], 'citations': []}
        source = (research_source or '').lower()
        try:
            if source in ('newsapi', 'both'):
                news_data = self.fetch_relevant_news(topic=topic, keywords=merged_keywords, language=lang_code)
                research_insights = self._merge_research_insights(research_insights, news_data)
            if source in ('google', 'both'):
                google_data = self.fetch_google_research(topic=topic, keywords=merged_keywords, language=lang_code)
                research_insights = self._merge_research_insights(research_insights, google_data)
        except Exception:
            pass

        # Build variants
        variant_results = []
        for _ in range(max(1, variants)):
            article = self.generate_article(
                topic=topic,
                keywords=merged_keywords,
                seo_insights=seo_insights,
                language=language_label,
                tone=tone,
                word_count=word_count,
                kb_context=kb_context,
                feedback=feedback,
                news_insights=research_insights,
                research_prefs={'require_citations': require_citations, 'strict_grounding': strict_grounding},
                llm_model=llm_model,
                enforce_keyword_coverage=enforce_keyword_coverage,
                llm_temperature=max(0.0, min(1.0, llm_temperature)),
            )

            # Generate social posts directly in target language and tone
            social_posts = self.generate_social_posts(article, language_label=language_label, tone=tone, llm_model=llm_model, llm_temperature=max(0.0, min(1.0, llm_temperature + 0.2)))

            # Add to variants list
            variant_results.append({
                'article': article,
                'social_posts': social_posts
            })

        # Translate article to selected language if needed
        try:
            if language_label != 'English':
                target_lang = self.SUPPORTED_LANGUAGES.get(language_label, 'en')
                translator_prompt = (
                    "Translate the following content to " + language_label + 
                    ". Keep formatting and tone. Only return translated text."
                )
                # Translate article only for all variants (posts were generated in target language)
                for variant in variant_results:
                    translated_article = self._translate_text(variant['article']['content'], translator_prompt, llm_model=llm_model)
                    if translated_article:
                        variant['article']['content'] = translated_article
                        variant['article']['language'] = language_label
        except Exception:
            # Fallback: keep original English content
            pass
        
        # Generate enhanced content calendar
        calendar_entries = []
        
        # Main article entry
        if variant_results:
            main_article = variant_results[0]['article']
            calendar_entries.append({
                'topic': main_article['topic'],
                'format': 'Blog Post',
                'platform': 'Website',
                'target_audience': 'Marketing Professionals',
                'description': f"Comprehensive {main_article['word_count']}-word article covering {', '.join(main_article['keywords'][:3])}",
                'estimated_time': f"{main_article['word_count'] // 200}-{main_article['word_count'] // 150} min read",
                'keywords': main_article['keywords'][:5]
            })
            
            # Social media entries from first variant
            social_posts = variant_results[0].get('social_posts', {})
            for platform, post in social_posts.items():
                calendar_entries.append({
                    'topic': f"{main_article['topic']} - {platform.title()} Promotion",
                    'format': 'Social Media Post',
                    'platform': platform.title(),
                    'target_audience': f"{platform.title()} Audience",
                    'description': f"Engaging {platform} post ({len(post['content'])} chars) promoting the main article",
                    'estimated_time': "1-2 min engagement",
                    'keywords': main_article['keywords'][:3]
                })
        
        # Additional content suggestions based on keywords
        if merged_keywords:
            for i, keyword in enumerate(merged_keywords[:2], 1):
                calendar_entries.append({
                    'topic': f"{keyword} Deep Dive",
                    'format': 'Follow-up Article',
                    'platform': 'Website',
                    'target_audience': 'Marketing Professionals',
                    'description': f"Detailed exploration of {keyword} as a follow-up to the main article",
                    'estimated_time': f"{word_count//2 // 200}-{word_count//2 // 150} min read",
                    'keywords': [keyword] + [kw for kw in merged_keywords if kw != keyword][:2]
                })
        
        content_calendar = {'calendar': calendar_entries}

        return {
            'variants': variant_results,
            'content_calendar': content_calendar,
            'seo_insights': seo_insights,
            'news_insights': research_insights
        }

    def _generate_social_post(self, platform, topic, keywords, tone, language_label, llm_model: str = "gpt-4o", llm_temperature: float = 0.6):
        """
        Use OpenAI to generate a platform-specific post with sufficient length and variation.
        """
        target_lengths = {
            'twitter': 240,    # near max
            'linkedin': 600,   # richer post
            'instagram': 1500, # caption-style
            'facebook': 400,
        }
        stylistic_tips = {
            'twitter': 'Use concise sentences, 1-2 emojis, 2-3 hashtags. No links.',
            'linkedin': 'Use a hook, 2-3 short paragraphs, 3-5 hashtags at the end.',
            'instagram': 'Use an engaging caption, emojis, and 5-8 hashtags at the end.',
            'facebook': 'Friendly tone, 2 short paragraphs, 2-4 hashtags.',
        }

        platform_key = platform.lower()
        target_length = target_lengths.get(platform_key, 400)
        style = stylistic_tips.get(platform_key, 'Keep it engaging and useful.')

        keywords_text = ', '.join(keywords[:8]) if keywords else topic
        required_keywords_block = "\n".join([f"- {kw}" for kw in (keywords[:5] if keywords else [])])
        
        system_prompt = (
            f"You are an expert social media copywriter specializing in {platform} content. "
            f"Create engaging posts in {language_label} with a {tone} tone that drive engagement and clicks."
        )
        
        user_prompt = (
            f"Create a {platform} post about: {topic}\n\n"
            f"Key themes/keywords to include: {keywords_text}\n\n"
            f"Platform Guidelines: {style}\n\n"
            f"Target length: approximately {target_length} characters\n\n"
            f"Make it engaging, valuable, and optimized for {platform}'s audience. "
            f"Include relevant emojis and hashtags as specified in the guidelines.\n\n"
            f"Required keywords (verbatim). If length-constrained, prioritize the FIRST keywords in this list and include at least two: \n"
            f"{required_keywords_block if required_keywords_block else '- (none)'}"
        )

        try:
            response = self.client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(0.0, min(1.0, llm_temperature)),  # Creativity
                max_tokens=800,  # Reasonable limit for social posts
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Social post generation failed for {platform}: {e}")
            # Return None to trigger fallback
            return None

    def _translate_text(self, text, system_prompt, llm_model: str = "gpt-4o"):
        """
        Translate text using the OpenAI client; return original on failure.
        """
        try:
            # Use responses API compatible with installed SDKs; keep simple
            response = self.client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception:
            return text

    def content_manager(self):
        """
        Content management utility
        
        :return: ContentManager instance
        """
        class ContentManager:
            def __init__(self, storage_path):
                self.storage_path = storage_path
            
            def schedule_content_group(self, content_items, publish_date):
                """
                Schedule a group of content items
                
                :param content_items: List of content to schedule
                :param publish_date: Scheduled publication date
                :return: List of scheduled content
                """
                try:
                    # Load existing scheduled articles
                    try:
                        with open(self.storage_path, 'r') as f:
                            scheduled_articles = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        scheduled_articles = []
                
                    # Add new scheduled items
                    for item in content_items:
                        item['scheduled_date'] = publish_date
                        scheduled_articles.append(item)
                
                    # Save updated schedule
                    with open(self.storage_path, 'w') as f:
                        json.dump(scheduled_articles, f, indent=2)
                
                    return content_items
                except Exception as e:
                    print(f"Error scheduling content: {e}")
                    return []
        
        return ContentManager(self.content_storage_path)
