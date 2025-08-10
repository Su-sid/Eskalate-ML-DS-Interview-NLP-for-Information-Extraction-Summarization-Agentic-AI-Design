# TF-IDF search system
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class DocumentSearcher:
    """TF-IDF based document search system"""
    
    def __init__(self, documents_df: pd.DataFrame):
        self.documents_df = documents_df
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF search index"""
        print("🔍 Building document search index...")
        texts = self.documents_df['processed_text'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print("✅ Search index ready")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum relevance threshold
                doc = self.documents_df.iloc[idx]
                results.append({
                    'headline': doc['headline'],
                    'short_description': doc['short_description'],
                    'category': doc['category'],
                    'date': doc.get('date', 'N/A'),
                    'link': doc.get('link', ''),
                    'similarity_score': float(similarities[idx]),
                    'combined_text': doc['combined_text']
                })
        return results


    # Initialize document searcher

doc_searcher = DocumentSearcher(documents_df)


def search_articles_func(query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant articles based on query.

        Args:
            query (str): Search query
            top_k (int): Number of articles to return

        Returns:
            List[Dict]: List of relevant articles with metadata
        """
    return doc_searcher.search_documents(query, top_k)