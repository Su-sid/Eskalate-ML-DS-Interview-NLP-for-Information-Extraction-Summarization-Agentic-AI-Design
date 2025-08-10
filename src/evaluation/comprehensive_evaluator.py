# Quality metrics
from typing import Dict, List, Optional
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.text_processor import TextPreprocessor

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

preprocessor = TextPreprocessor()

class ComprehensiveEvaluator:
    """Advanced evaluation system for NLP tasks"""

    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_entity_extraction(self, text: str, entities_ner: Dict, entities_regex: Dict) -> Dict:
        """Comprehensive entity extraction evaluation with qualitative examples"""

        evaluation = {
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'ner_analysis': self._analyze_entities(entities_ner, "Named Entity Recognition"),
            'regex_analysis': self._analyze_entities(entities_regex, "Regex Pattern Extraction"),
            'quality_assessment': {},
            'examples': {}
        }

        # Quality indicators
        total_ner = sum(len(ents) for ents in entities_ner.values())
        total_regex = sum(len(ents) for ents in entities_regex.values())

        evaluation['quality_assessment'] = {
            'total_entities_ner': total_ner,
            'total_entities_regex': total_regex,
            'entity_coverage': 'High' if (total_ner + total_regex) >= 5 else 'Medium' if (total_ner + total_regex) >= 2 else 'Low',
            'diversity_score': len(entities_ner.keys()) + len(entities_regex.keys()),
            'extraction_success': total_ner > 0 or total_regex > 0
        }

        # Qualitative examples
        evaluation['examples']['ner_samples'] = self._get_entity_samples(entities_ner)
        evaluation['examples']['regex_samples'] = self._get_entity_samples(entities_regex)

        return evaluation

    def evaluate_summarization(self, original_text: str, extractive_summary: str,
                             abstractive_summary: str, reference_summary: str = None) -> Dict:
        """Comprehensive summarization evaluation with qualitative analysis"""

        evaluation = {
            'text_info': {
                'original_length_words': len(original_text.split()),
                'original_length_chars': len(original_text),
                'original_sentences': len(nltk.sent_tokenize(original_text))
            },
            'extractive_evaluation': self._evaluate_single_summary(original_text, extractive_summary, "Extractive"),
            'abstractive_evaluation': self._evaluate_single_summary(original_text, abstractive_summary, "Abstractive"),
            'comparative_analysis': {},
            'qualitative_examples': {}
        }

        # Comparative analysis
        ext_score = evaluation['extractive_evaluation']['content_preservation']['similarity_score']
        abs_score = evaluation['abstractive_evaluation']['content_preservation']['similarity_score']

        evaluation['comparative_analysis'] = {
            'better_content_preservation': 'Extractive' if ext_score > abs_score else 'Abstractive',
            'extractive_compression': evaluation['extractive_evaluation']['compression']['ratio'],
            'abstractive_compression': evaluation['abstractive_evaluation']['compression']['ratio'],
            'coherence_comparison': {
                'extractive': evaluation['extractive_evaluation']['coherence']['coherence_score'],
                'abstractive': evaluation['abstractive_evaluation']['coherence']['coherence_score']
            }
        }

        # Qualitative examples
        evaluation['qualitative_examples'] = {
            'original_excerpt': original_text[:300] + "..." if len(original_text) > 300 else original_text,
            'extractive_summary': extractive_summary,
            'abstractive_summary': abstractive_summary,
            'summary_comparison': self._compare_summaries(extractive_summary, abstractive_summary)
        }

        # ROUGE evaluation if reference available
        if reference_summary and ROUGE_AVAILABLE:
            evaluation['rouge_scores'] = self._calculate_rouge_scores(
                extractive_summary, abstractive_summary, reference_summary
            )

        return evaluation

    def _analyze_entities(self, entities: Dict, method_name: str) -> Dict:
        """Analyze entity extraction results"""
        analysis = {
            'method': method_name,
            'entity_types_found': len(entities),
            'total_entities': sum(len(ents) for ents in entities.values()),
            'entity_breakdown': {}
        }

        for entity_type, entity_list in entities.items():
            if entity_list:
                analysis['entity_breakdown'][entity_type] = {
                    'count': len(entity_list),
                    'unique_count': len(set(entity_list)),
                    'samples': entity_list[:3]  # First 3 examples
                }

        return analysis

    def _get_entity_samples(self, entities: Dict) -> Dict:
        """Get sample entities for qualitative review"""
        samples = {}
        for entity_type, entity_list in entities.items():
            if entity_list:
                samples[entity_type] = entity_list[:3]  # Top 3 examples
        return samples

    def _evaluate_single_summary(self, original: str, summary: str, method: str) -> Dict:
        """Evaluate a single summary comprehensively"""
        orig_words = len(original.split())
        summ_words = len(summary.split())

        evaluation = {
            'method': method,
            'compression': {
                'ratio': summ_words / orig_words if orig_words > 0 else 0,
                'original_words': orig_words,
                'summary_words': summ_words,
                'compression_percentage': (1 - summ_words / orig_words) * 100 if orig_words > 0 else 0
            },
            'coherence': self._assess_coherence(summary),
            'content_preservation': self._assess_content_preservation(original, summary)
        }

        return evaluation

    def _assess_coherence(self, text: str) -> Dict:
        """Assess text coherence using linguistic indicators"""
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return {'coherence_score': 0, 'sentence_count': 0, 'avg_sentence_length': 0}

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Check for discourse markers
        discourse_markers = ['however', 'therefore', 'furthermore', 'moreover', 'additionally',
                           'consequently', 'meanwhile', 'similarly', 'in contrast', 'as a result',
                           'first', 'second', 'finally', 'also', 'furthermore']

        marker_count = sum(1 for marker in discourse_markers if marker in text.lower())

        # Simple coherence score
        coherence_score = min(marker_count / len(sentences), 1.0) if sentences else 0

        return {
            'coherence_score': coherence_score,
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'discourse_markers_found': marker_count,
            'readability_indicator': 'Good' if 10 <= avg_sentence_length <= 20 else 'Needs improvement'
        }

    def _assess_content_preservation(self, original: str, summary: str) -> Dict:
        """Assess content preservation using multiple metrics"""
        # TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english')

        try:
            tfidf_matrix = vectorizer.fit_transform([original.lower(), summary.lower()])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0.0

        # Keyword overlap
        orig_tokens = set(preprocessor.tokenize_and_filter(original))
        summ_tokens = set(preprocessor.tokenize_and_filter(summary))

        if orig_tokens:
            overlap_ratio = len(orig_tokens.intersection(summ_tokens)) / len(orig_tokens)
        else:
            overlap_ratio = 0

        return {
            'similarity_score': similarity,
            'keyword_overlap_ratio': overlap_ratio,
            'shared_keywords_count': len(orig_tokens.intersection(summ_tokens)),
            'preservation_quality': 'High' if similarity > 0.3 else 'Medium' if similarity > 0.1 else 'Low'
        }

    def _compare_summaries(self, extractive: str, abstractive: str) -> Dict:
        """Compare extractive and abstractive summaries"""
        return {
            'length_difference': abs(len(extractive.split()) - len(abstractive.split())),
            'extractive_length': len(extractive.split()),
            'abstractive_length': len(abstractive.split()),
            'style_difference': 'Abstractive appears more concise' if len(abstractive.split()) < len(extractive.split()) else 'Similar length'
        }

    def _calculate_rouge_scores(self, extractive: str, abstractive: str, reference: str) -> Dict:
        """Calculate ROUGE scores against reference summary"""
        scores = {}

        if ROUGE_AVAILABLE:
            ext_scores = self.rouge_scorer.score(reference, extractive)
            abs_scores = self.rouge_scorer.score(reference, abstractive)

            scores = {
                'extractive_rouge': {
                    'rouge1_f1': ext_scores['rouge1'].fmeasure,
                    'rouge2_f1': ext_scores['rouge2'].fmeasure,
                    'rougeL_f1': ext_scores['rougeL'].fmeasure
                },
                'abstractive_rouge': {
                    'rouge1_f1': abs_scores['rouge1'].fmeasure,
                    'rouge2_f1': abs_scores['rouge2'].fmeasure,
                    'rougeL_f1': abs_scores['rougeL'].fmeasure
                }
            }

        return scores

# Initialize evaluator
evaluator = ComprehensiveEvaluator()
print("✅ Evaluation system created!")