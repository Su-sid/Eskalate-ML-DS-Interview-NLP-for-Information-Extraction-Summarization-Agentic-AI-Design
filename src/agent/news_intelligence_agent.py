# Agno agent for news intelligence
from agno.agent import Agent
from agno.models.groq import Groq
from typing import List, Dict

from src.tools.agno_tools import NEWS_AGENT_TOOLS
from src.preprocessing.text_processor import TextPreprocessor
from src.extraction.entity_extractor import extract_entities_ner_func, extract_entities_regex_func
from src.extraction.keyword_extractor import get_top_keywords_func
from src.summarization.extractive_summarizer import extractive_summarize_func
from src.summarization.abstractive_summarizer import simple_abstractive_summarize_func
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from src.utils.document_searcher import search_articles_func

class NewsIntelligenceAgent:
    """News Intelligence Agent using Agno framework"""
    
    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile"):
        self.preprocessor = TextPreprocessor()
        self.evaluator = ComprehensiveEvaluator()
        self.agent = self._create_agent(api_key, model_id)
    
    def _create_agent(self, api_key: str, model_id: str) -> Agent:
        """Create and configure the Agno agent"""
        return Agent(
            name="News Intelligence Agent",
            role="Advanced news analysis and information extraction specialist",
            instructions="""You are a sophisticated news intelligence agent that helps users understand current events and extract insights from news articles.
Your capabilities include:
- Searching and retrieving relevant news articles
- Extracting key entities (people, organizations, locations, dates, etc.)
- Generating both extractive and abstractive summaries
- Evaluating the quality of information extraction and summarization
- Providing comprehensive analysis reports
When users ask about news topics, search for relevant articles first, then provide comprehensive analysis including entity extraction, summarization, and quality evaluation. Always include specific examples and metrics in your responses.""",
            tools=NEWS_AGENT_TOOLS,
            model=Groq(id=model_id, api_key=api_key),
            show_tool_calls=True,
            markdown=True
        )
    
    def analyze_topic(self, topic: str, num_articles: int = 3) -> str:
        """Analyze a news topic comprehensively"""
        articles = search_articles_func(topic, num_articles)
        
        if not articles:
            return f"❌ No relevant articles found for topic: '{topic}'"
        
        combined_text = ' '.join([article['combined_text'] for article in articles])
        
        entities_ner = extract_entities_ner_func(combined_text)
        entities_regex = extract_entities_regex_func(combined_text)
        keywords = get_top_keywords_func(combined_text, 8)
        
        extractive_summary = extractive_summarize_func(combined_text, 3)
        abstractive_summary = simple_abstractive_summarize_func(combined_text, 60)
        
        evaluation = self.evaluator.evaluate_entity_extraction(combined_text, entities_ner, entities_regex)
        summary_eval = self.evaluator.evaluate_summarization(combined_text, extractive_summary, abstractive_summary)
        
        report_parts = [
            f"📰 NEWS ANALYSIS REPORT: {topic.upper()}",
            "=" * 50,
            f"📊 Articles analyzed: {len(articles)}",
            f"📂 Categories: {', '.join(set(article['category'] for article in articles))}",
            "",
            "🔍 KEY FINDINGS:",
            f"• Total entities found: {evaluation['quality_assessment']['total_entities_ner'] + evaluation['quality_assessment']['total_entities_regex']}",
            f"• Entity coverage: {evaluation['quality_assessment']['entity_coverage']}",
            f"• Content diversity: {evaluation['quality_assessment']['diversity_score']} types",
            "",
            "👥 KEY ENTITIES:",
        ]
        
        # Add entity information
        if entities_ner:
            for entity_type, entities in list(entities_ner.items())[:3]:
                if entities:
                    report_parts.append(f"• {entity_type}: {', '.join(entities[:3])}")
        
        if entities_regex:
            for pattern_type, patterns in entities_regex.items():
                if patterns:
                    report_parts.append(f"• {pattern_type}: {', '.join(patterns[:2])}")
        
        report_parts.extend([
            "",
            "🔑 TOP KEYWORDS:",
        ])
        
        for word, score in keywords[:5]:
            report_parts.append(f"• {word}: {score:.3f}")
        
        report_parts.extend([
            "",
            "📋 EXTRACTIVE SUMMARY:",
            f"{extractive_summary}",
            "",
            "🎯 ABSTRACTIVE SUMMARY:",
            f"{abstractive_summary}",
            "",
            "📊 SUMMARY QUALITY METRICS:",
            f"• Extractive compression: {summary_eval['extractive_evaluation']['compression']['compression_percentage']:.1f}%",
            f"• Abstractive compression: {summary_eval['abstractive_evaluation']['compression']['compression_percentage']:.1f}%",
            f"• Content preservation: {summary_eval['extractive_evaluation']['content_preservation']['preservation_quality']}",
            f"• Coherence score: {summary_eval['extractive_evaluation']['coherence']['coherence_score']:.2f}",
            "",
            "📑 TOP ARTICLES ANALYZED:",
        ])
        
        for i, article in enumerate(articles[:3], 1):
            report_parts.append(f"{i}. {article['headline']} ({article['category']})")
        
        return '\n'.join(report_parts)
    
    def evaluate_quality(self, text: str) -> str:
        """Evaluate NLP processing quality on a text sample"""
        return self.agent.run(f"Evaluate the NLP processing quality on this text: '{text[:200]}...'").content
    
    def run(self, query: str) -> str:
        """Run a general query through the agent"""
        return self.agent.run(query).content
