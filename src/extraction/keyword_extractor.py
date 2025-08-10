# TF-IDF keywords
from collections import Counter
from typing import List, Tuple
from src.preprocessing.text_processor import TextPreprocessor

preprocessor = TextPreprocessor()

def get_top_keywords_func(text: str, n: int = 10) -> List[Tuple[str, float]]:
	"""
	Extract top keywords using TF-IDF (simple TF for demo).
	"""
	tokens = preprocessor.tokenize_and_filter(text)
	if not tokens:
		return []
	word_freq = Counter(tokens)
	total_words = len(tokens)
	tf_scores = [(word, count/total_words) for word, count in word_freq.most_common(n)]
	return tf_scores
