"""Text summarization utilities."""
from .extractive_summarizer import extractive_summarize_func
from .abstractive_summarizer import simple_abstractive_summarize_func

__all__ = ['extractive_summarize_func', 'simple_abstractive_summarize_func']
