# Text cleaning utilities
import re
import nltk
from typing import List

class TextPreprocessor:
	"""Advanced text preprocessing utility"""

	def __init__(self):
		self.stop_words = set(nltk.corpus.stopwords.words('english'))

	def clean_text(self, text: str) -> str:
		"""Clean and preprocess text"""
		if not isinstance(text, str):
			return ""
		# Remove special characters and extra whitespace
		text = re.sub(r'[^\w\s]', ' ', text)
		text = re.sub(r'\s+', ' ', text.strip())
		# Convert to lowercase
		text = text.lower()
		return text

	def tokenize_and_filter(self, text: str) -> List[str]:
		"""Tokenize and remove stop words"""
		tokens = nltk.word_tokenize(self.clean_text(text))
		return [token for token in tokens if token not in self.stop_words and len(token) > 2]
