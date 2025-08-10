# Keyword-driven
import nltk
from src.extraction.keyword_extractor import get_top_keywords_func

def simple_abstractive_summarize_func(text: str, max_words: int = 50) -> str:
	"""
	Generate a simple abstractive summary by combining key sentences and phrases.
	"""
	sentences = nltk.sent_tokenize(text)
	if len(sentences) <= 2:
		return text
	keywords = get_top_keywords_func(text, 8)
	key_terms = [kw[0] for kw in keywords]
	sentence_scores = []
	for sent in sentences:
		sent_lower = sent.lower()
		score = sum(1 for term in key_terms if term in sent_lower)
		sentence_scores.append((sent, score))
	sentence_scores.sort(key=lambda x: x[1], reverse=True)
	summary_parts = []
	word_count = 0
	for sent, score in sentence_scores:
		sent_words = len(sent.split())
		if word_count + sent_words <= max_words:
			summary_parts.append(sent)
			word_count += sent_words
		else:
			break
	if not summary_parts:
		summary_parts = [sentences[0]]
	return ' '.join(summary_parts)
