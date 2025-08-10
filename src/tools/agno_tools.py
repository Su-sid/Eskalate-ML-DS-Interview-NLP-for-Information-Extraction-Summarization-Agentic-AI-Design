# Agno Tools

from agno.tools import tool
from typing import List, Dict

# Import local modules
from src.preprocessing.text_processor import TextPreprocessor
from src.extraction.entity_extractor import extract_entities_ner_func, extract_entities_regex_func
from src.extraction.keyword_extractor import get_top_keywords_func
from src.summarization.extractive_summarizer import extractive_summarize_func
from src.summarization.abstractive_summarizer import simple_abstractive_summarize_func
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from src.utils.document_searcher import search_articles_func

# Initialize shared objects
preprocessor = TextPreprocessor()
evaluator = ComprehensiveEvaluator()

# Tool definitions
@tool(
    name="extract_named_entities",
    description="Extract named entities (persons, organizations, locations) from text using spaCy NER",
    show_result=False,
    cache_results=True,
    cache_ttl=3600
)
def extract_entities_ner(text: str) -> Dict[str, List[str]]:
    return extract_entities_ner_func(text)

@tool(
    name="extract_patterns_regex",
    description="Extract structured patterns (dates, URLs, emails, numbers) from text using regex",
    show_result=False,
    cache_results=True
)
def extract_entities_regex(text: str) -> Dict[str, List[str]]:
    return extract_entities_regex_func(text)

@tool(
    name="extract_keywords_tfidf",
    description="Extract top keywords from text using TF-IDF scoring",
    show_result=False
)
def get_top_keywords(text: str, n: int = 10) -> List[tuple]:
    return get_top_keywords_func(text, n)

@tool(
    name="extractive_summarization",
    description="Generate extractive summary using TF-IDF sentence ranking",
    show_result=False,
    cache_results=True
)
def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    return extractive_summarize_func(text, num_sentences)

@tool(
    name="abstractive_summarization",
    description="Generate abstractive summary using language model",
    show_result=False,
    requires_confirmation=False
)
def simple_abstractive_summarize(text: str, max_words: int = 50) -> str:
    """Tool version"""
    return simple_abstractive_summarize_func(text, max_words)

@tool(
    name="search_news_articles",
    description="Search for relevant news articles based on query terms",
    show_result=False,
    cache_results=True
)
def search_articles(query: str, top_k: int = 5) -> List[Dict]:
    return search_articles_func(query, top_k)

@tool(
    name="evaluate_nlp_quality",
    description="Evaluate the quality of entity extraction and summarization on sample text",
    show_result=True,
    requires_confirmation=False
)
def evaluate_sample_quality(text: str) -> str:
    entities_ner = extract_entities_ner_func(text)
    entities_regex = extract_entities_regex_func(text)
    extractive_summary = extractive_summarize_func(text, 2)
    abstractive_summary = simple_abstractive_summarize_func(text, 40)
    
    entity_eval = evaluator.evaluate_entity_extraction(text, entities_ner, entities_regex)
    summary_eval = evaluator.evaluate_summarization(text, extractive_summary, abstractive_summary)
    
    report = [
        "🔍 NLP QUALITY EVALUATION REPORT",
        "=" * 40,
        f"📝 Text length: {len(text.split())} words",
        "",
        "👥 ENTITY EXTRACTION EVALUATION:",
        f"• NER entities found: {entity_eval['quality_assessment']['total_entities_ner']}",
        f"• Regex patterns found: {entity_eval['quality_assessment']['total_entities_regex']}",
        f"• Overall coverage: {entity_eval['quality_assessment']['entity_coverage']}",
        "",
        "📋 ENTITY EXAMPLES:",
    ]
    
    # Add entity examples
    for entity_type, examples in entity_eval['examples']['ner_samples'].items():
        if examples:
            report.append(f"• {entity_type}: {', '.join(examples)}")
    
    for pattern_type, examples in entity_eval['examples']['regex_samples'].items():
        if examples:
            report.append(f"• {pattern_type}: {', '.join(examples)}")
    
    report.extend([
        "",
        "📊 SUMMARIZATION EVALUATION:",
        f"• Extractive quality: {summary_eval['extractive_evaluation']['content_preservation']['preservation_quality']}",
        f"• Abstractive quality: {summary_eval['abstractive_evaluation']['content_preservation']['preservation_quality']}",
        f"• Best approach: {summary_eval['comparative_analysis']['better_content_preservation']}",
        "",
        "📝 SUMMARY EXAMPLES:",
        f"Extractive: {extractive_summary}",
        "",
        f"Abstractive: {abstractive_summary}",
        "",
        "📈 QUALITY METRICS:",
        f"• Content preservation: {summary_eval['extractive_evaluation']['content_preservation']['similarity_score']:.3f}",
        f"• Compression ratio: {summary_eval['extractive_evaluation']['compression']['ratio']:.2f}",
        f"• Coherence score: {summary_eval['extractive_evaluation']['coherence']['coherence_score']:.2f}",
    ])
    
    return '\n'.join(report)

# Tool list for agent initialization
NEWS_AGENT_TOOLS = [
    search_articles,
    extract_entities_ner,
    extract_entities_regex,
    get_top_keywords,
    extractive_summarize,
    simple_abstractive_summarize,
    evaluate_sample_quality
]
