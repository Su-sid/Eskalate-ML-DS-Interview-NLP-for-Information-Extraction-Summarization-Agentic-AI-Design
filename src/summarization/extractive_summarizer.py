# TF-IDF ranking
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import nltk
from src.preprocessing.text_processor import TextPreprocessor

preprocessor = TextPreprocessor()

def extractive_summarize_func(text: str, num_sentences: int = 3) -> str:
	"""
	Generate extractive summary using TF-IDF sentence ranking.
	"""
	sentences = nltk.sent_tokenize(text)
	if len(sentences) <= num_sentences:
		return text
	processed_sentences = [preprocessor.clean_text(sent) for sent in sentences]
	try:
		vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
		tfidf_matrix = vectorizer.fit_transform(processed_sentences)
		sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
		top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
		top_indices = sorted(top_indices)
		summary_sentences = [sentences[i] for i in top_indices]
		return ' '.join(summary_sentences)
	except:
		return ' '.join(sentences[:num_sentences])
