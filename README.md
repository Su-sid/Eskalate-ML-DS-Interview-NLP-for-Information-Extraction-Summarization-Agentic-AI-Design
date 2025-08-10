# News Intelligence Agent

A comprehensive AI agent for news analysis, entity extraction, and intelligent summarization with quality evaluation.

## Project Overview

The News Intelligence Agent transforms unstructured news data into actionable insights through:

- **Multi-modal Entity Extraction** (NER + Regex patterns)
- **Dual Summarization** (Extractive + Abstractive approaches)
- **Comprehensive Quality Evaluation** with ROUGE metrics
- **Interactive Intelligence** via Agno agent framework

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Su-sid/Eskalate-ML-DS-Interview-NLP-for-Information-Extraction-Summarization-Agentic-AI-Design.git
cd news-intelligence-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy English model
python -m spacy download en_core_web_sm
```

### 2. API Configuration 

Set up your GROQ API key:

```bash
export GROQ_API_KEY="your-api-key-here"
```

### 3. Jupyter Setup

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open news_intelligence_system.ipynb
```

## Dataset Source and Preprocessing

### Dataset

- **Source**: [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset) via KaggleHub
- **Size**: 200,000+ news articles from HuffPost (2012-2022)
- **Format**: JSON lines with headline, short_description, category, date, link

### Preprocessing Pipeline

```python
# 1. Data Loading
df = load_news_dataset()  # Automatic download via kagglehub

# 2. Text Preprocessing
preprocessor = TextPreprocessor()
- Clean special characters and normalize whitespace
- Convert to lowercase
- Tokenization with stop word removal
- Combine headline + description for analysis

# 3. Feature Engineering
- Document length analysis
- Category distribution mapping
- TF-IDF vectorization for search
- Temporal analysis (when dates available)

# 4. Sampling Strategy
# Uses stratified sampling for demo (500 articles)
# Maintains category distribution
# Ensures processing efficiency
```

## How to Run Each Module

### 1. Basic Data Analysis

```python
# Run exploratory data analysis
perform_eda(df_sample)

# View preprocessing results
sample_text = df_sample['combined_text'].iloc[0]
processed = preprocessor.clean_text(sample_text)
```

### 2. Entity Extraction

```python
# Named Entity Recognition
entities_ner = extract_entities_ner_func(text)
print("People:", entities_ner.get('PERSON', []))
print("Organizations:", entities_ner.get('ORG', []))

# Pattern-based Extraction  
entities_regex = extract_entities_regex_func(text)
print("Dates:", entities_regex.get('dates', []))
print("URLs:", entities_regex.get('urls', []))
```

### 3. Summarization

```python
# Extractive Summarization (TF-IDF based)
extractive_summary = extractive_summarize_func(text, num_sentences=3)

# Abstractive Summarization (keyword-driven)
abstractive_summary = simple_abstractive_summarize_func(text, max_words=50)
```

### 4. Quality Evaluation

```python
# Comprehensive evaluation with metrics
evaluator = ComprehensiveEvaluator()
evaluation = evaluator.evaluate_summarization(
    original_text, extractive_summary, abstractive_summary
)

print("Content Preservation:", evaluation['extractive_evaluation']['content_preservation']['similarity_score'])
print("Compression Ratio:", evaluation['extractive_evaluation']['compression']['ratio'])
```

### 5. Agent Interaction

```python
# Initialize the intelligent agent
agent = news_intelligence_agent

# Comprehensive topic analysis
response = agent.run("Analyze artificial intelligence news trends")

# Interactive mode
interactive_news_interface()
```

### 6. Demonstration Workflows

```python
# Run full system demo
run_comprehensive_demo()

# Detailed evaluation examples
run_detailed_evaluation_examples()

# System performance analysis
performance_metrics = analyze_system_performance()
```

## Agent Design Explanation

### Architecture Components

#### 1. **Core Processing Modules**

- **TextPreprocessor**: Advanced text cleaning and normalization
- **Entity Extraction**: Dual approach (spaCy NER + Regex patterns)
- **Summarization**: Extractive (TF-IDF) + Abstractive (keyword-driven)
- **Quality Evaluation**: Multi-metric assessment with ROUGE integration

#### 2. **Agent Framework Integration**

- **Agno-based Architecture**: Tool-driven agent with structured workflows
- **Tool Chaining**: Sequential and parallel processing pipelines
- **Quality Assurance**: Built-in evaluation at every processing stage

#### 3. **Intelligence Layer**

```python
# Query Processing Flow
User Query → Intent Classification → Document Retrieval → 
Multi-Tool Processing → Quality Evaluation → Report Synthesis
```

### Workflow Strategy

#### Document Selection

1. **TF-IDF Vector Search**: Semantic similarity matching
2. **Relevance Filtering**: Minimum threshold + diversity optimization
3. **Quality Gates**: Content completeness validation

#### Tool Orchestration

1. **Parallel Extraction**: Simultaneous NER, regex, and keyword extraction
2. **Dual Summarization**: Comparative extractive vs. abstractive analysis
3. **Cross-Validation**: Quality metrics ensure output reliability

#### Response Generation

1. **Structured Reporting**: Combines quantitative metrics with qualitative examples
2. **Evidence-Based**: Provides specific examples and confidence scores
3. **Actionable Insights**: Formatted for decision-making

### Key Design Decisions

#### Why Dual Summarization?

- **Extractive**: Preserves original content fidelity
- **Abstractive**: Provides concise, readable summaries
- **Evaluation**: Quality metrics determine best approach per document

#### Why Multi-Modal Entity Extraction?

- **NER**: Captures semantic entities (people, organizations, locations)
- **Regex**: Extracts structured patterns (dates, URLs, currencies)
- **Complementary**: Comprehensive coverage of information types

#### Why Quality-First Design?

- **Transparency**: Users see confidence scores and evaluation metrics
- **Reliability**: Built-in validation prevents hallucination
- **Improvement**: Performance tracking enables system optimization

## Project Structure

```
news-intelligence-agent/
├── notebooks/
│   └── news_intelligence_system.ipynb    # Main analysis notebook
├── src/
│   ├── preprocessing/
│   │   └── text_processor.py             # Text cleaning utilities
│   ├── extraction/
│   │   ├── entity_extractor.py           # NER + Regex extraction
│   │   └── keyword_extractor.py          # TF-IDF keywords
│   ├── summarization/
│   │   ├── extractive_summarizer.py      # TF-IDF ranking
│   │   └── abstractive_summarizer.py     # Keyword-driven
│   ├── evaluation/
│   │   └── comprehensive_evaluator.py    # Quality metrics
│   ├── tools/
│   │   └── agno_tools.py    # Agno Tools
│   ├── agent/
│   │   └── news_intelligence_agent.py    # Agno agent
│   └── utils/
│       └── document_searcher.py          # TF-IDF search
├── requirements.txt                       # Dependencies
├── README.md                             # This file
└── System_report.md                           # Technical report
```

## Troubleshooting

### Common Issues

1. **spaCy Model Missing**

   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Memory Issues with Large Datasets**

   ```python
   # Reduce sample size
   SAMPLE_SIZE = 100  # Instead of 500
   ```

3. **API Key Issues**

   - Ensure you sign up for a [free groq key](https://console.groq.com/keys)
 

## Performance Metrics

Based on 500-article evaluation:

- **Entity Extraction Success Rate**: ~85%
- **Summarization Success Rate**: ~90%
- **Average Content Preservation**: 0.35 (TF-IDF similarity)
- **Average Compression Ratio**: 0.15 (85% reduction)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Contact

For questions or support, please open an issue on GitHub.