# NER + Regex extraction
import spacy
import re
from collections import defaultdict
from typing import Dict, List

nlp = spacy.load('en_core_web_sm')

def extract_entities_ner_func(text: str) -> Dict[str, List[str]]:
	"""
	Extract named entities from text using spaCy NER.
	"""
	doc = nlp(text)
	entities = defaultdict(list)
	for ent in doc.ents:
		entities[ent.label_].append(ent.text)
	result = {}
	for label, ents in entities.items():
		result[label] = list(set(ents))
	return result

def extract_entities_regex_func(text: str) -> Dict[str, List[str]]:
	"""
	Extract entities using regex patterns.
	"""
	patterns = {
		'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
		'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
		'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
		'phone_numbers': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
		'currencies': r'\$\d+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR)\b',
		'percentages': r'\b\d+(?:\.\d+)?%\b'
	}
	extracted = {}
	for pattern_name, pattern in patterns.items():
		matches = re.findall(pattern, text, re.IGNORECASE)
		extracted[pattern_name] = list(set(matches)) if matches else []
	return extracted
