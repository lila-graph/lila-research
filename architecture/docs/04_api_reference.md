# API Reference

**Project:** lila-research - RAG Evaluation Framework
**Version:** 0.1.0
**Python Requirement:** >=3.13
**Generated:** 2025-10-04

---

## Overview

This API reference documents the public interfaces for the RAG Evaluation Framework, a comprehensive system for evaluating retrieval-augmented generation (RAG) strategies. The framework combines three major components:

1. **RAG Evaluation Pipeline** - Core evaluation framework with 6 retrieval strategies
2. **Repository Analyzer Framework** - Portable multi-domain orchestration system
3. **Validation & Analysis Tools** - Data analysis and telemetry validation utilities

**Key Features:**
- 6 retrieval strategies (naive, semantic, BM25, compression, multiquery, ensemble)
- RAGAS-based golden test set generation
- Phoenix observability integration for LLM tracing
- PostgreSQL vector store with PGVector
- Automated experiments with QA correctness and RAG relevance metrics
- Portable orchestrator framework for multi-domain analysis

---

## Core Components

### RAG Evaluation Module

#### Configuration Management

##### `Config` (Dataclass)

**Purpose:** Centralized configuration for RAG evaluation pipeline
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:47-96`

**Attributes:**

```python
@dataclass
class Config:
    # API Keys
    openai_api_key: str
    cohere_api_key: str

    # Phoenix settings
    phoenix_endpoint: str = "http://localhost:6006"
    project_name: str = f"retrieval-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Database settings
    postgres_user: str = "langchain"
    postgres_password: str = "langchain"
    postgres_host: str = "localhost"
    postgres_port: str = "6024"
    postgres_db: str = "langchain"
    vector_size: int = 1536
    table_baseline: str = "mixed_baseline_documents"
    table_semantic: str = "mixed_semantic_documents"
    overwrite_existing_tables: bool = True

    # Model settings
    model_name: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    # Data settings
    data_urls: List[tuple] = None
    load_pdfs: bool = True
    load_csvs: bool = False
    load_markdowns: bool = True

    # Golden test set settings
    golden_testset_size: int = 10

    # Research domain configuration
    research_domain: str = "theory_of_mind"
```

**Properties:**
- `async_url` (str): Returns PostgreSQL async connection string for asyncpg driver

**Example:**
```python
from src.langchain_eval_foundations_e2e import Config, setup_environment

# Initialize from environment variables
config = setup_environment()

# Access configuration
print(f"Using model: {config.model_name}")
print(f"Database URL: {config.async_url}")
print(f"Phoenix endpoint: {config.phoenix_endpoint}")
```

**Usage Notes:**
- Loads API keys from `.env` file via `setup_environment()`
- All database settings use sensible defaults to avoid port conflicts
- `async_url` property automatically constructs the asyncpg connection string
- `project_name` includes timestamp for unique Phoenix project identification

---

#### Environment Setup

##### `setup_environment() -> Config`

**Purpose:** Initialize environment and load configuration from .env file
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:99-113`

**Parameters:** None

**Returns:**
- `Config`: Configured instance with API keys loaded from environment

**Example:**
```python
from src.langchain_eval_foundations_e2e import setup_environment

config = setup_environment()
# Environment variables are automatically set:
# - OPENAI_API_KEY
# - COHERE_API_KEY
# - PHOENIX_COLLECTOR_ENDPOINT
```

**Usage Notes:**
- Calls `load_dotenv()` to read `.env` file
- Sets environment variables for downstream libraries
- Required API keys: `OPENAI_API_KEY`, `COHERE_API_KEY`
- Safe to call multiple times (idempotent)

---

##### `setup_phoenix_tracing(config: Config)`

**Purpose:** Configure Phoenix observability with auto-instrumentation disabled to prevent resource exhaustion
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:116-124`

**Parameters:**
- `config` (Config): Configuration object with Phoenix settings

**Returns:**
- Tracer provider for Phoenix observability

**Example:**
```python
from src.langchain_eval_foundations_e2e import setup_environment, setup_phoenix_tracing

config = setup_environment()
tracer_provider = setup_phoenix_tracing(config)

# Phoenix UI now available at http://localhost:6006
# Auto-instrumentation is disabled to handle large documents
```

**Usage Notes:**
- `auto_instrument=False` prevents RESOURCE_EXHAUSTED errors with large markdown documents
- Batching enabled for efficient trace transmission
- View traces at configured `phoenix_endpoint` (default: http://localhost:6006)
- Must be called before creating LangChain components for proper tracing

---

#### Vector Store Management

##### `async setup_vector_store(config: Config, table_name: str, embeddings) -> PGVectorStore`

**Purpose:** Reusable vector store initialization for PostgreSQL
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:127-141`

**Parameters:**
- `config` (Config): Configuration with database settings
- `table_name` (str): Name of PostgreSQL table for vector storage
- `embeddings`: LangChain embeddings instance (e.g., OpenAIEmbeddings)

**Returns:**
- `PGVectorStore`: Initialized vector store ready for document ingestion

**Example:**
```python
from langchain_openai import OpenAIEmbeddings
from src.langchain_eval_foundations_e2e import setup_environment, setup_vector_store

config = setup_environment()
embeddings = OpenAIEmbeddings(model=config.embedding_model)

# Create baseline vector store
baseline_vectorstore = await setup_vector_store(
    config,
    config.table_baseline,
    embeddings
)

# Add documents
await baseline_vectorstore.aadd_documents(documents)
```

**Usage Notes:**
- Creates table if it doesn't exist
- Vector size automatically matches embedding model (1536 for text-embedding-3-small)
- Set `overwrite_existing_tables=True` in config to reset tables on each run
- Uses async PostgreSQL connection for performance

---

#### Retrieval Strategies

##### `create_retrievers(baseline_vectorstore, semantic_vectorstore, all_docs, llm) -> Dict[str, Any]`

**Purpose:** Create all 6 retrieval strategies for evaluation
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:143-174`

**Parameters:**
- `baseline_vectorstore` (PGVectorStore): Vector store with basic chunking
- `semantic_vectorstore` (PGVectorStore): Vector store with semantic chunking
- `all_docs` (List[Document]): All documents for BM25 retriever
- `llm`: LangChain LLM instance for multi-query retriever

**Returns:**
- `Dict[str, Any]`: Dictionary mapping strategy names to retriever instances
  - `"naive"`: Basic vector similarity search
  - `"semantic"`: Semantic chunking-based vector search
  - `"bm25"`: BM25 keyword-based retrieval
  - `"compression"`: Contextual compression with Cohere reranking
  - `"multiquery"`: Multi-query retriever with query expansion
  - `"ensemble"`: Ensemble combining all strategies

**Example:**
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.langchain_eval_foundations_e2e import (
    setup_environment,
    setup_vector_store,
    create_retrievers
)

config = setup_environment()
llm = ChatOpenAI(model=config.model_name)
embeddings = OpenAIEmbeddings(model=config.embedding_model)

# Setup vector stores
baseline_vs = await setup_vector_store(config, "baseline", embeddings)
semantic_vs = await setup_vector_store(config, "semantic", embeddings)

# Create all retrievers
retrievers = create_retrievers(baseline_vs, semantic_vs, all_docs, llm)

# Use individual retrievers
naive_results = retrievers["naive"].get_relevant_documents("query")
ensemble_results = retrievers["ensemble"].get_relevant_documents("query")
```

**Retrieval Strategy Details:**

| Strategy | Type | Description | Key Parameters |
|----------|------|-------------|----------------|
| **naive** | Vector | Basic similarity search on baseline chunks | k=10 |
| **semantic** | Vector | Similarity search on semantic chunks | k=10 |
| **bm25** | Keyword | BM25 ranking algorithm | k=10 (default) |
| **compression** | Hybrid | Cohere reranking on top of naive | model=rerank-english-v3.0 |
| **multiquery** | Enhanced | Query expansion with LLM | base=naive |
| **ensemble** | Ensemble | Weighted combination of all | weights=[0.25, 0.25, 0.25, 0.25] |

**Usage Notes:**
- All retrievers return k=10 documents by default
- Compression strategy requires Cohere API key
- Multiquery strategy uses LLM to generate query variations
- Ensemble weights can be adjusted based on strategy performance
- Each strategy has different latency and cost characteristics

---

#### Document Loading

##### `async load_pdf_documents(data_dir: Path) -> List[Document]`

**Purpose:** Load PDF documents with metadata for theory of mind research
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:176-208`

**Parameters:**
- `data_dir` (Path): Directory containing PDF files

**Returns:**
- `List[Document]`: List of LangChain Document objects with page-level granularity

**Example:**
```python
from pathlib import Path
from src.langchain_eval_foundations_e2e import load_pdf_documents

data_dir = Path.cwd() / "data"
pdf_docs = await load_pdf_documents(data_dir)

print(f"Loaded {len(pdf_docs)} PDF pages")
for doc in pdf_docs[:3]:
    print(f"Source: {doc.metadata['document_name']}")
    print(f"Page content length: {len(doc.page_content)} chars")
```

**Metadata Added:**
- `source_type`: "pdf"
- `document_name`: PDF filename without extension
- `research_domain`: "theory_of_mind"
- `last_accessed_at`: ISO 8601 timestamp

**Usage Notes:**
- Automatically finds all `*.pdf` files in directory
- Each PDF page becomes a separate Document
- Gracefully handles errors and continues with remaining files
- Returns empty list if no PDFs found

---

##### `async load_markdown_documents(data_dir: Path) -> List[Document]`

**Purpose:** Load Markdown documents with H2 header-based semantic splitting
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:211-286`

**Parameters:**
- `data_dir` (Path): Directory containing Markdown files

**Returns:**
- `List[Document]`: List of Documents split by H2 sections with metadata

**Example:**
```python
from pathlib import Path
from src.langchain_eval_foundations_e2e import load_markdown_documents

data_dir = Path.cwd() / "data"
md_docs = await load_markdown_documents(data_dir)

print(f"Loaded {len(md_docs)} Markdown chunks")
for doc in md_docs[:3]:
    print(f"Document: {doc.metadata['Header_1']}")
    print(f"Section: {doc.metadata['Header_2']}")
    print(f"Chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']}")
```

**Splitting Strategy:**
- **Stage 1**: Split on H2 headings (`##`) to preserve semantic sections
- **Stage 2**: Further split oversized sections (chunk_size=2000, overlap=200)

**Metadata Added:**
- `source_type`: "markdown"
- `document_name`: Filename without extension
- `Header_1`: H1 title extracted from content
- `Header_2`: H2 section heading (from MarkdownHeaderTextSplitter)
- `Header_3`: Empty (reserved for future use)
- `chunk_index`: Index of chunk within document
- `total_chunks`: Total chunks for this document
- `research_domain`: "theory_of_mind"
- `last_accessed_at`: ISO 8601 timestamp

**Usage Notes:**
- Preserves H2 sections as complete semantic units
- H3 subsections remain intact within H2 sections
- Headers not stripped from content for better context
- Oversized sections automatically split for embedding model constraints
- Returns empty list if no Markdown files found

---

##### `async load_and_process_data(config: Config) -> List[Document]`

**Purpose:** Orchestrate loading of CSV, PDF, and Markdown data sources
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:289-357`

**Parameters:**
- `config` (Config): Configuration with data loading flags

**Returns:**
- `List[Document]`: Combined list of all loaded documents

**Example:**
```python
from src.langchain_eval_foundations_e2e import setup_environment, load_and_process_data

config = setup_environment()
all_docs = await load_and_process_data(config)

print(f"Total documents: {len(all_docs)}")
# Output: Total documents: 245 (CSV: 0, PDF: 127, Markdown: 118)

# Filter by source type
pdf_docs = [d for d in all_docs if d.metadata.get('source_type') == 'pdf']
md_docs = [d for d in all_docs if d.metadata.get('source_type') == 'markdown']
```

**Data Loading Flags in Config:**
- `load_pdfs` (bool): Enable/disable PDF loading (default: True)
- `load_csvs` (bool): Enable/disable CSV loading (default: False)
- `load_markdowns` (bool): Enable/disable Markdown loading (default: True)

**Usage Notes:**
- Creates `data/` directory if it doesn't exist
- CSV data downloads from URLs if not cached locally
- Provides detailed logging of loading progress
- Gracefully handles missing or corrupt files
- Returns combined list for easy filtering by metadata

---

#### RAG Chain Creation

##### `create_rag_chain(retriever, llm, method_name: str)`

**Purpose:** Create RAG chain with Phoenix auto-tracing and method tagging
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:359-371`

**Parameters:**
- `retriever`: LangChain retriever instance
- `llm`: LangChain LLM instance
- `method_name` (str): Name of retrieval strategy for tracing

**Returns:**
- Configured RAG chain with Phoenix instrumentation

**Example:**
```python
from langchain_openai import ChatOpenAI
from src.langchain_eval_foundations_e2e import (
    create_retrievers,
    create_rag_chain
)

llm = ChatOpenAI(model="gpt-4.1-mini")
retrievers = create_retrievers(baseline_vs, semantic_vs, all_docs, llm)

# Create chains for each strategy
chains = {
    name: create_rag_chain(retriever, llm, name)
    for name, retriever in retrievers.items()
}

# Use chain
result = await chains["naive"].ainvoke({"question": "What is Theory of Mind?"})
print(result["response"].content)
```

**Chain Structure:**
1. Extract question from input
2. Retrieve relevant documents using retriever
3. Pass question and context to LLM via RAG_PROMPT
4. Return response and context

**Phoenix Tracing:**
- Run name: `rag_chain_{method_name}`
- Span attributes: `{"retriever": method_name}`
- Enables filtering by retrieval strategy in Phoenix UI

**Usage Notes:**
- Uses centralized `RAG_PROMPT` template for consistency
- Returns both response and retrieved context
- Traces automatically sent to Phoenix for observability
- Async-compatible with `ainvoke()`

---

#### Evaluation Execution

##### `async run_evaluation(question: str, chains: Dict[str, Any]) -> Dict[str, str]`

**Purpose:** Execute evaluation across all retrieval strategies
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:374-387`

**Parameters:**
- `question` (str): Query to evaluate
- `chains` (Dict[str, Any]): Dictionary of RAG chains by strategy name

**Returns:**
- `Dict[str, str]`: Dictionary mapping strategy names to response strings

**Example:**
```python
from src.langchain_eval_foundations_e2e import run_evaluation

question = "What is the role of Theory of Mind in self-reflective AI agents?"
results = await run_evaluation(question, chains)

# Compare responses
for strategy, response in results.items():
    print(f"\n{strategy.upper()}:")
    print(f"{response[:200]}...")

# Output example:
# NAIVE:
# Theory of Mind (ToM) plays a crucial role in self-reflective AI agents...
#
# ENSEMBLE:
# Self-reflective AI agents leverage Theory of Mind capabilities to...
```

**Usage Notes:**
- Executes all strategies concurrently for fair comparison
- Gracefully handles errors (returns error message for failed strategies)
- Logs errors for debugging
- Useful for qualitative comparison of retrieval strategies

---

### Golden Test Set Generation

#### RAGAS Test Set Generation

##### `generate_testset(docs: list, llm, embeddings, testset_size: int = 10)`

**Purpose:** Generate golden test set using RAGAS TestsetGenerator
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_golden_testset.py:16-42`

**Parameters:**
- `docs` (list): List of LangChain Document objects
- `llm`: LangChain LLM instance wrapped with `LangchainLLMWrapper`
- `embeddings`: LangChain embeddings wrapped with `LangchainEmbeddingsWrapper`
- `testset_size` (int): Number of test examples to generate (default: 10)

**Returns:**
- RAGAS testset with synthetic questions, reference answers, and contexts

**Example:**
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.langchain_eval_golden_testset import generate_testset
from src.data_loader import load_docs_from_postgres

# Load documents
docs = load_docs_from_postgres("mixed_baseline_documents")

# Setup RAGAS models
llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Generate testset
testset = generate_testset(docs, generator_llm, generator_embeddings, testset_size=20)
print(f"Generated {len(testset.samples)} test examples")
```

**Usage Notes:**
- Requires at least 3 documents, recommended 10+ for quality
- Works best with 20-50 documents (avoid timeouts)
- Automatically samples if too many documents provided
- Generation takes several minutes depending on testset size
- Creates diverse question types (simple, reasoning, multi-context)

---

##### `upload_to_phoenix(golden_testset, dataset_name: str = "mixed_golden_testset") -> dict`

**Purpose:** Convert RAGAS testset to Phoenix format and upload
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_golden_testset.py:45-128`

**Parameters:**
- `golden_testset`: RAGAS testset object
- `dataset_name` (str): Name for Phoenix dataset (default: "mixed_golden_testset")

**Returns:**
- `dict`: Upload result with dataset metadata
  - `dataset_name`: Name of uploaded dataset
  - `num_samples`: Number of samples uploaded
  - `status`: "success" or error message
  - `dataset`: Phoenix dataset object

**Example:**
```python
from src.langchain_eval_golden_testset import generate_testset, upload_to_phoenix

testset = generate_testset(docs, generator_llm, generator_embeddings, testset_size=15)
result = upload_to_phoenix(testset, dataset_name="my_golden_testset")

print(f"Status: {result['status']}")
print(f"Uploaded {result['num_samples']} samples")
print(f"Dataset ID: {result['dataset'].id}")
```

**Phoenix Dataset Schema:**
- `input`: Question/query (from RAGAS `user_input`)
- `output`: Reference answer (from RAGAS `reference`)
- `contexts`: Retrieved contexts as string
- `synthesizer`: RAGAS synthesizer name
- `question_type`: Type of question generated
- `dataset_source`: "ragas_golden_testset"

**Usage Notes:**
- Handles both old and new RAGAS API formats
- Validates testset is not empty before upload
- Provides detailed error messages for troubleshooting
- Dataset available immediately in Phoenix UI for experiments
- Uses Phoenix Client API for reliable uploads

---

### Automated Experiments

#### Phoenix Evaluators

##### `@create_evaluator(name="qa_correctness_score")`

**Purpose:** Evaluate answer correctness against ground truth using QAEvaluator
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_experiments.py:24-59`

**Function Signature:**
```python
def qa_correctness_evaluator(output, reference, input) -> float
```

**Parameters:**
- `output` (str): Generated answer from RAG system
- `reference` (str): Ground truth reference answer
- `input` (str): Original question/query

**Returns:**
- `float`: Correctness score (0.0 to 1.0)

**Example:**
```python
from phoenix.experiments import run_experiment
from src.langchain_eval_experiments import qa_correctness_evaluator

# Used automatically in experiments
experiment = run_experiment(
    dataset=dataset,
    task=task_function,
    evaluators=[qa_correctness_evaluator],
    experiment_name="naive_rag_eval"
)
```

**Evaluation Criteria:**
- Semantic similarity to reference answer
- Factual accuracy
- Completeness of response
- Uses GPT-4.1-mini as evaluation model

**Usage Notes:**
- Based on Phoenix official QAEvaluator pattern
- Requires reference answers in dataset
- Returns 0.0 on evaluation errors (with error logging)
- Higher scores indicate better answer quality

---

##### `@create_evaluator(name="rag_relevance_score")`

**Purpose:** Evaluate retrieved context relevance using RelevanceEvaluator
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_experiments.py:61-92`

**Function Signature:**
```python
def rag_relevance_evaluator(output, input, metadata) -> float
```

**Parameters:**
- `output` (str): Generated answer (not used, included for API compatibility)
- `input` (str): Original question/query
- `metadata` (dict): Must contain `retrieved_context` key

**Returns:**
- `float`: Relevance score (0.0 to 1.0)

**Example:**
```python
from phoenix.experiments import run_experiment
from src.langchain_eval_experiments import (
    create_enhanced_task_function,
    rag_relevance_evaluator
)

# Task function must return metadata with retrieved_context
task = create_enhanced_task_function(chain, "naive")

experiment = run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[rag_relevance_evaluator],
    experiment_name="naive_rag_relevance"
)
```

**Evaluation Criteria:**
- Relevance of retrieved documents to query
- Context quality assessment
- Uses GPT-4.1-mini as evaluation model

**Usage Notes:**
- Requires task function to return metadata with `retrieved_context`
- Use `create_enhanced_task_function()` to ensure proper metadata
- Returns 0.0 on evaluation errors (with error logging)
- Evaluates retrieval quality independent of generation

---

##### `create_enhanced_task_function(strategy_chain, strategy)`

**Purpose:** Factory for task functions that capture retrieval context for evaluators
**Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_experiments.py:96-111`

**Parameters:**
- `strategy_chain`: RAG chain for specific retrieval strategy
- `strategy` (str): Name of retrieval strategy

**Returns:**
- Task function compatible with Phoenix experiments that returns dict with metadata

**Example:**
```python
from phoenix.experiments import run_experiment
from src.langchain_eval_experiments import create_enhanced_task_function

# Create task function for each strategy
naive_task = create_enhanced_task_function(chains["naive"], "naive")
ensemble_task = create_enhanced_task_function(chains["ensemble"], "ensemble")

# Use in experiments
experiment = run_experiment(
    dataset=dataset,
    task=naive_task,
    evaluators=[qa_correctness_evaluator, rag_relevance_evaluator],
    experiment_name="naive_full_eval"
)
```

**Returned Task Function Signature:**
```python
def task(example: Example) -> dict:
    return {
        "output": str,  # Generated response
        "metadata": {
            "retrieved_context": List[Document],
            "strategy": str
        }
    }
```

**Usage Notes:**
- Required for RAG relevance evaluation
- Captures both response and retrieval context
- Metadata includes strategy name for result analysis
- Compatible with all Phoenix evaluators

---

### Data Loading Utilities

##### `load_docs_from_postgres(table_name: str = "mixed_baseline_documents") -> List[Document]`

**Purpose:** Load documents from PostgreSQL table into LangChain Document objects
**Source:** `/home/donbr/lila-graph/lila-research/src/data_loader.py:10-46`

**Parameters:**
- `table_name` (str): Name of PostgreSQL table (default: "mixed_baseline_documents")

**Returns:**
- `List[Document]`: List of LangChain Document objects with content and metadata

**Example:**
```python
from src.data_loader import load_docs_from_postgres

# Load baseline documents
docs = load_docs_from_postgres("mixed_baseline_documents")
print(f"Loaded {len(docs)} documents")

# Access document content and metadata
for doc in docs[:3]:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

**Database Connection:**
- Uses environment variables or defaults:
  - `POSTGRES_USER` (default: "langchain")
  - `POSTGRES_PASSWORD` (default: "langchain")
  - `POSTGRES_HOST` (default: "localhost")
  - `POSTGRES_PORT` (default: "6024")
  - `POSTGRES_DB` (default: "langchain")

**Usage Notes:**
- Returns empty list on errors (with error logging)
- Loads both content and langchain_metadata columns
- Synchronous function (use in async contexts with `asyncio.to_thread`)
- Useful for experiment setup and RAGAS test generation

---

## Repository Analyzer Framework

### Base Orchestrator

##### `BaseOrchestrator` (Abstract Base Class)

**Purpose:** Base framework for domain-specific orchestrators
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:30-309`

**Constructor:**
```python
def __init__(
    self,
    domain_name: str,
    output_base_dir: Path = Path("ra_output"),
    show_tool_details: bool = True,
    use_timestamp: bool = True,
)
```

**Parameters:**
- `domain_name` (str): Name of domain (e.g., 'architecture', 'ux', 'devops')
- `output_base_dir` (Path): Base directory for outputs (default: ra_output)
- `show_tool_details` (bool): Display detailed tool usage (default: True)
- `use_timestamp` (bool): Append timestamp to output directory (default: True)

**Attributes:**
- `domain_name` (str): Domain name
- `output_dir` (Path): Timestamped output directory
- `output_base_dir` (Path): Base output directory
- `show_tool_details` (bool): Tool detail visibility flag
- `total_cost` (float): Total cost tracking
- `phase_costs` (Dict[str, float]): Per-phase cost breakdown
- `completed_phases` (List[str]): List of completed phase names

**Abstract Methods (Must Implement):**
```python
@abstractmethod
def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
    """Return agent definitions for this orchestrator."""
    pass

@abstractmethod
def get_allowed_tools(self) -> List[str]:
    """Return list of allowed tools."""
    pass

@abstractmethod
async def run(self):
    """Run the orchestrator workflow."""
    pass
```

**Example:**
```python
from pathlib import Path
from typing import Dict, List
from claude_agent_sdk import AgentDefinition
from ra_orchestrators.base_orchestrator import BaseOrchestrator

class CustomOrchestrator(BaseOrchestrator):
    def __init__(self):
        super().__init__(domain_name="custom")
        self.create_output_structure(["phase1", "phase2"])

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        return {
            "analyzer": AgentDefinition(
                description="Code analyzer",
                prompt="You are an expert code analyzer...",
                tools=["Read", "Write", "Grep"],
                model="sonnet"
            )
        }

    def get_allowed_tools(self) -> List[str]:
        return ["Read", "Write", "Grep", "Glob", "Bash"]

    async def run(self):
        self.display_phase_header(1, "Analysis Phase", "🔍")
        await self.execute_phase(
            "analysis",
            "analyzer",
            "Analyze the codebase...",
            self.client
        )

# Usage
orchestrator = CustomOrchestrator()
await orchestrator.run_with_client()
```

---

#### BaseOrchestrator Methods

##### `create_output_structure(subdirs: Optional[List[str]] = None)`

**Purpose:** Create output directory structure with optional subdirectories
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:75-85`

**Parameters:**
- `subdirs` (Optional[List[str]]): List of subdirectory names to create

**Example:**
```python
orchestrator = CustomOrchestrator()
orchestrator.create_output_structure(["docs", "diagrams", "reports"])
# Creates: ra_output/custom_20251004_120000/docs/
#          ra_output/custom_20251004_120000/diagrams/
#          ra_output/custom_20251004_120000/reports/
```

---

##### `display_message(msg, show_tools: bool = True)`

**Purpose:** Display message content with full visibility into tool usage
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:87-121`

**Parameters:**
- `msg`: Message to display (AssistantMessage, UserMessage, or ResultMessage)
- `show_tools` (bool): Whether to show tool usage details

**Example:**
```python
async for msg in client.receive_response():
    orchestrator.display_message(msg)
    # Output:
    # 🤖 Agent: Analyzing codebase structure...
    # 🔧 Using tool: Read
    #    Reading: /path/to/file.py
    # 🔧 Using tool: Write
    #    ✍️  Writing: ra_output/analysis.md
```

---

##### `display_phase_header(phase_number: int, phase_name: str, emoji: str = "📋")`

**Purpose:** Display formatted phase header for progress tracking
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:130-141`

**Parameters:**
- `phase_number` (int): Phase number (1-indexed)
- `phase_name` (str): Name of the phase
- `emoji` (str): Emoji to display (default: "📋")

**Example:**
```python
orchestrator.display_phase_header(1, "Component Inventory", "🔍")
# Output:
# ======================================================================
# 🔍 PHASE 1: Component Inventory
# ======================================================================
```

---

##### `track_phase_cost(phase_name: str, cost: float)`

**Purpose:** Track cost for specific phase execution
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:142-150`

**Parameters:**
- `phase_name` (str): Name of the phase
- `cost` (float): Cost in USD

**Example:**
```python
orchestrator.track_phase_cost("analysis", 0.45)
orchestrator.track_phase_cost("documentation", 0.32)
# Total cost: $0.77
```

---

##### `mark_phase_complete(phase_name: str)`

**Purpose:** Mark phase as completed in tracking system
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:152-158`

**Parameters:**
- `phase_name` (str): Name of the completed phase

**Example:**
```python
orchestrator.mark_phase_complete("analysis")
orchestrator.mark_phase_complete("documentation")
print(orchestrator.completed_phases)
# Output: ['analysis', 'documentation']
```

---

##### `async verify_outputs(expected_files: List[Path]) -> bool`

**Purpose:** Verify all expected outputs were created
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:160-183`

**Parameters:**
- `expected_files` (List[Path]): List of expected file paths

**Returns:**
- `bool`: True if all files exist, False otherwise

**Example:**
```python
expected = [
    orchestrator.output_dir / "docs" / "component_inventory.md",
    orchestrator.output_dir / "diagrams" / "architecture.md"
]

all_exist = await orchestrator.verify_outputs(expected)
if all_exist:
    print("All outputs verified!")
else:
    print("Some outputs missing!")
```

---

##### `display_summary()`

**Purpose:** Display orchestrator run summary with costs and completion status
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:185-200`

**Example:**
```python
orchestrator.display_summary()
# Output:
# ======================================================================
# 📊 ARCHITECTURE ORCHESTRATOR SUMMARY
# ======================================================================
# Domain: architecture
# Output Directory: /path/to/ra_output/architecture_20251004_120000
# Completed Phases: 5
# Total Cost: $2.34
#
# Cost Breakdown:
#   - component_inventory: $0.45
#   - architecture_diagrams: $0.67
#   - data_flows: $0.52
#   - api_documentation: $0.38
#   - synthesis: $0.32
# ======================================================================
```

---

##### `async execute_phase(phase_name: str, agent_name: str, prompt: str, client: ClaudeSDKClient)`

**Purpose:** Execute single phase of workflow with cost tracking
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:229-253`

**Parameters:**
- `phase_name` (str): Name of the phase
- `agent_name` (str): Name of the agent to use
- `prompt` (str): Prompt for the agent
- `client` (ClaudeSDKClient): Claude SDK client

**Example:**
```python
await orchestrator.execute_phase(
    "component_inventory",
    "analyzer",
    "Analyze the codebase and create a component inventory...",
    client
)
```

**Automatic Features:**
- Displays all agent messages
- Tracks phase cost
- Marks phase as complete
- Shows tool usage in real-time

---

##### `create_client_options(permission_mode: str = "acceptEdits", cwd: str = ".") -> ClaudeAgentOptions`

**Purpose:** Create Claude SDK client options for orchestrator
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:255-277`

**Parameters:**
- `permission_mode` (str): Permission mode for the client (default: "acceptEdits")
- `cwd` (str): Current working directory (default: ".")

**Returns:**
- `ClaudeAgentOptions`: Configured options for Claude SDK client

**Example:**
```python
options = orchestrator.create_client_options(
    permission_mode="acceptEdits",
    cwd="/path/to/project"
)
```

---

##### `async run_with_client()`

**Purpose:** Run orchestrator with automatic client setup and teardown
**Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:279-308`

**Returns:**
- `bool`: True on success

**Example:**
```python
orchestrator = ArchitectureOrchestrator()
success = await orchestrator.run_with_client()

if success:
    print(f"Results in: {orchestrator.output_dir}")
```

**Automatic Features:**
- Creates and configures Claude SDK client
- Calls `run()` method with client
- Displays summary on completion
- Handles errors gracefully
- Cleans up client on exit

---

### Agent Registry

##### `AgentRegistry`

**Purpose:** Discover and load agent definitions from JSON files
**Source:** `/home/donbr/lila-graph/lila-research/ra_agents/registry.py:10-100`

**Constructor:**
```python
def __init__(self, agents_dir: Path = Path(__file__).parent)
```

**Parameters:**
- `agents_dir` (Path): Base directory containing agent definitions

**Example:**
```python
from ra_agents.registry import AgentRegistry

registry = AgentRegistry()

# Discover all agents
agents = registry.discover_agents()
print(f"Found {len(agents)} agent definitions")

# Load specific agent
analyzer = registry.load_agent("analyzer", domain="architecture")
print(f"Agent: {analyzer.description}")
```

---

##### `discover_agents(domain: Optional[str] = None) -> Dict[str, str]`

**Purpose:** Discover all available agent definition files
**Source:** `/home/donbr/lila-graph/lila-research/ra_agents/registry.py:22-43`

**Parameters:**
- `domain` (Optional[str]): Optional domain filter (e.g., 'ux', 'architecture')

**Returns:**
- `Dict[str, str]`: Dictionary mapping agent names to file paths

**Example:**
```python
registry = AgentRegistry()

# All agents
all_agents = registry.discover_agents()

# Domain-specific agents
ux_agents = registry.discover_agents(domain="ux")
arch_agents = registry.discover_agents(domain="architecture")
```

---

##### `load_agent(agent_name: str, domain: Optional[str] = None) -> Optional[AgentDefinition]`

**Purpose:** Load agent definition from JSON file with caching
**Source:** `/home/donbr/lila-graph/lila-research/ra_agents/registry.py:45-80`

**Parameters:**
- `agent_name` (str): Name of the agent (without .json extension)
- `domain` (Optional[str]): Optional domain to search in

**Returns:**
- `Optional[AgentDefinition]`: AgentDefinition object or None if not found

**Example:**
```python
registry = AgentRegistry()

# Load agent
analyzer = registry.load_agent("analyzer", domain="architecture")

if analyzer:
    print(f"Description: {analyzer.description}")
    print(f"Tools: {analyzer.tools}")
    print(f"Model: {analyzer.model}")
```

---

##### `load_domain_agents(domain: str) -> Dict[str, AgentDefinition]`

**Purpose:** Load all agents for specific domain
**Source:** `/home/donbr/lila-graph/lila-research/ra_agents/registry.py:82-99`

**Parameters:**
- `domain` (str): Domain name (e.g., 'ux', 'architecture')

**Returns:**
- `Dict[str, AgentDefinition]`: Dictionary mapping agent names to AgentDefinition objects

**Example:**
```python
registry = AgentRegistry()
arch_agents = registry.load_domain_agents("architecture")

for name, agent in arch_agents.items():
    print(f"{name}: {agent.description}")
```

---

### MCP Registry

##### `MCPRegistry`

**Purpose:** MCP (Model Context Protocol) server discovery and management
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:8-153`

**Constructor:**
```python
def __init__(self)
```

**Example:**
```python
from ra_tools.mcp_registry import MCPRegistry

registry = MCPRegistry()

# Check server availability
if registry.is_server_available("figma"):
    print("Figma MCP server is available")
    tools = registry.get_server_tools("figma")
    print(f"Available tools: {tools}")
```

---

##### `discover_mcp_servers() -> Dict[str, Dict[str, Any]]`

**Purpose:** Auto-discover available MCP servers
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:16-53`

**Returns:**
- `Dict[str, Dict[str, Any]]`: Dictionary of MCP servers with capabilities

**Supported Servers:**
- **figma**: Figma MCP Server for design context
- **v0**: Vercel v0 MCP Server for UI generation
- **sequential-thinking**: Advanced reasoning MCP tool
- **playwright**: Browser automation MCP tool

**Example:**
```python
registry = MCPRegistry()
servers = registry.discover_mcp_servers()

for name, info in servers.items():
    print(f"{name}: {info['description']}")
    print(f"  Available: {info['available']}")
    print(f"  Tools: {info['tools']}")
```

---

##### `is_server_available(server_name: str) -> bool`

**Purpose:** Check if MCP server is available
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:55-67`

**Parameters:**
- `server_name` (str): Name of the MCP server

**Returns:**
- `bool`: True if server is available

**Example:**
```python
if registry.is_server_available("figma"):
    # Use Figma integration
    pass
else:
    # Use fallback approach
    fallbacks = registry.get_fallback_options("figma_get_file")
```

---

##### `get_server_tools(server_name: str) -> List[str]`

**Purpose:** Get list of tools provided by MCP server
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:69-81`

**Parameters:**
- `server_name` (str): Name of the MCP server

**Returns:**
- `List[str]`: List of tool names

**Example:**
```python
figma_tools = registry.get_server_tools("figma")
# Returns: ["figma_get_file", "figma_get_components"]

v0_tools = registry.get_server_tools("v0")
# Returns: ["v0_generate_ui", "v0_generate_from_image", "v0_chat_complete"]
```

---

##### `validate_tool_availability(tool_name: str) -> bool`

**Purpose:** Validate if specific tool is available
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:83-96`

**Parameters:**
- `tool_name` (str): Name of the tool

**Returns:**
- `bool`: True if tool is available

**Example:**
```python
if registry.validate_tool_availability("figma_get_file"):
    # Tool is available
    pass
```

---

##### `get_configuration_requirements(server_name: str) -> Optional[Dict[str, Any]]`

**Purpose:** Get configuration requirements for MCP server
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:98-128`

**Parameters:**
- `server_name` (str): Name of the MCP server

**Returns:**
- `Optional[Dict[str, Any]]`: Configuration requirements or None

**Example:**
```python
figma_config = registry.get_configuration_requirements("figma")
print(f"Required env vars: {figma_config['required_env']}")
print(f"Setup: {figma_config['setup_instructions']}")

# Output:
# Required env vars: ['FIGMA_ACCESS_TOKEN']
# Setup: Get access token from Figma Settings > Personal Access Tokens
```

---

##### `get_fallback_options(tool_name: str) -> List[str]`

**Purpose:** Get fallback options if tool is unavailable
**Source:** `/home/donbr/lila-graph/lila-research/ra_tools/mcp_registry.py:130-152`

**Parameters:**
- `tool_name` (str): Name of the requested tool

**Returns:**
- `List[str]`: List of alternative approaches

**Example:**
```python
fallbacks = registry.get_fallback_options("figma_get_file")
for option in fallbacks:
    print(f"- {option}")

# Output:
# - Create design specifications in markdown
# - Use Mermaid diagrams for wireframes
# - Document design manually with screenshots
```

---

## Configuration

### Environment Variables

All environment variables should be set in `.env` file at project root.

#### Required Variables

**OpenAI Configuration:**
```bash
# Required for embeddings and chat models
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Cohere Configuration:**
```bash
# Required for reranking in contextual compression
COHERE_API_KEY=your-cohere-api-key-here
```

#### Optional Variables

**Phoenix Observability:**
```bash
# Phoenix collector endpoint (default: http://localhost:6006)
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

# Phoenix authentication headers for cloud instances
# Format: 'key1=value1,key2=value2'
PHOENIX_CLIENT_HEADERS=

# Phoenix project name (defaults to timestamp-based name)
PHOENIX_PROJECT_NAME=my-rag-evaluation
```

**PostgreSQL Database:**
```bash
# PostgreSQL connection settings (defaults shown)
POSTGRES_USER=langchain
POSTGRES_PASSWORD=langchain
POSTGRES_HOST=localhost
POSTGRES_PORT=6024
POSTGRES_DB=langchain

# Alternative: Full connection string
POSTGRES_CONNECTION_STRING=postgresql://langchain:langchain@localhost:6024/langchain
```

**Model Configuration:**
```bash
# OpenAI model selection (defaults shown)
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Golden Test Set:**
```bash
# Number of examples to generate (default: 10)
GOLDEN_TESTSET_SIZE=20
```

**Hugging Face (Optional):**
```bash
# For additional models
HUGGINGFACE_TOKEN=hf_your_token_here
```

---

### Docker Configuration

#### Service Ports

Configure ports in `.env` to avoid conflicts:

```bash
# PostgreSQL port (default: 6024, standard PostgreSQL is 5432)
POSTGRES_PORT=6024

# Phoenix UI port (default: 6006, same as TensorBoard)
PHOENIX_UI_PORT=6006

# Phoenix OpenTelemetry collector port (default: 4317, standard OTLP gRPC)
PHOENIX_OTLP_PORT=4317
```

**Example for avoiding conflicts:**
```bash
# If default ports conflict with existing services
POSTGRES_PORT=6025
PHOENIX_UI_PORT=6007
PHOENIX_OTLP_PORT=4318
```

#### Docker Compose Services

**Source:** `/home/donbr/lila-graph/lila-research/docker-compose.yml`

**Services:**

1. **PostgreSQL with PGVector**
   - Image: `pgvector/pgvector:pg16`
   - Container: `rag-eval-pgvector`
   - Port: `${POSTGRES_PORT:-6024}:5432`
   - Volume: `rag_eval_postgres_data`
   - Network: `rag-eval-network`
   - Health check: `pg_isready -U langchain`

2. **Phoenix Observability**
   - Image: `arizephoenix/phoenix:latest`
   - Container: `rag-eval-phoenix`
   - Ports:
     - UI: `${PHOENIX_UI_PORT:-6006}:6006`
     - OTLP: `${PHOENIX_OTLP_PORT:-4317}:4317`
   - Network: `rag-eval-network`

**Common Commands:**

```bash
# Check if services are running
docker ps | grep rag-eval

# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# View logs
docker-compose logs -f
docker-compose logs -f postgres
docker-compose logs -f phoenix

# Restart a service
docker-compose restart postgres
```

**Access URLs:**
- PostgreSQL: `localhost:${POSTGRES_PORT:-6024}`
- Phoenix UI: `http://localhost:${PHOENIX_UI_PORT:-6006}`
- Phoenix OTLP: `localhost:${PHOENIX_OTLP_PORT:-4317}`

---

## Usage Patterns

### Pattern 1: End-to-End RAG Evaluation

**Scenario:** Evaluate all 6 retrieval strategies on Theory of Mind research documents

```python
import asyncio
from src.langchain_eval_foundations_e2e import (
    setup_environment,
    setup_phoenix_tracing,
    setup_vector_store,
    load_and_process_data,
    create_retrievers,
    create_rag_chain,
    run_evaluation
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

async def main():
    # 1. Setup
    config = setup_environment()
    tracer_provider = setup_phoenix_tracing(config)

    # 2. Initialize models
    llm = ChatOpenAI(model=config.model_name)
    embeddings = OpenAIEmbeddings(model=config.embedding_model)

    # 3. Load data
    all_docs = await load_and_process_data(config)
    print(f"Loaded {len(all_docs)} documents")

    # 4. Setup vector stores
    baseline_vs = await setup_vector_store(config, config.table_baseline, embeddings)
    semantic_vs = await setup_vector_store(config, config.table_semantic, embeddings)

    # 5. Ingest documents
    await baseline_vs.aadd_documents(all_docs)

    # Semantic chunking for non-markdown docs
    from langchain_experimental.text_splitter import SemanticChunker
    non_md_docs = [d for d in all_docs if d.metadata.get('source_type') != 'markdown']
    md_docs = [d for d in all_docs if d.metadata.get('source_type') == 'markdown']

    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    semantic_docs = semantic_chunker.split_documents(non_md_docs) + md_docs
    await semantic_vs.aadd_documents(semantic_docs)

    # 6. Create retrievers and chains
    retrievers = create_retrievers(baseline_vs, semantic_vs, all_docs, llm)
    chains = {
        name: create_rag_chain(retriever, llm, name)
        for name, retriever in retrievers.items()
    }

    # 7. Run evaluation
    question = "What is the role of Theory of Mind in self-reflective AI agents?"
    results = await run_evaluation(question, chains)

    # 8. Compare results
    for strategy, response in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"{response[:200]}...")

    print(f"\nView traces at: {config.phoenix_endpoint}")

asyncio.run(main())
```

**Expected Output:**
- 6 different responses from retrieval strategies
- Phoenix traces at http://localhost:6006
- Documents stored in PostgreSQL

---

### Pattern 2: Golden Test Set Generation

**Scenario:** Generate RAGAS golden test set from existing documents

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.langchain_eval_golden_testset import generate_testset, upload_to_phoenix
from src.langchain_eval_foundations_e2e import setup_environment
from src.data_loader import load_docs_from_postgres

# 1. Setup
config = setup_environment()

# 2. Load documents
docs = load_docs_from_postgres(config.table_baseline)
print(f"Loaded {len(docs)} documents")

# 3. Setup RAGAS models
llm = ChatOpenAI(model=config.model_name)
embeddings = OpenAIEmbeddings(model=config.embedding_model)
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# 4. Generate testset (sample for efficiency)
import random
random.seed(42)
sampled_docs = random.sample(docs, min(50, len(docs)))

testset = generate_testset(
    sampled_docs,
    generator_llm,
    generator_embeddings,
    testset_size=20
)
print(f"Generated {len(testset.samples)} test examples")

# 5. Upload to Phoenix
result = upload_to_phoenix(testset, dataset_name="theory_of_mind_testset")
print(f"Upload status: {result['status']}")
print(f"Dataset ID: {result['dataset'].id}")
```

**Expected Output:**
- 20 synthetic test examples
- Dataset available in Phoenix UI
- Ready for experiments

---

### Pattern 3: Automated Experiments

**Scenario:** Run Phoenix experiments to compare all retrieval strategies

```python
import asyncio
from src.langchain_eval_experiments import (
    qa_correctness_evaluator,
    rag_relevance_evaluator,
    create_enhanced_task_function
)
from phoenix.experiments import run_experiment
import phoenix as px

async def run_experiments():
    # 1. Setup (similar to Pattern 1)
    # ... setup code ...

    # 2. Connect to Phoenix
    px_client = px.Client()
    dataset = px_client.get_dataset(name="theory_of_mind_testset")

    # 3. Run experiments for each strategy
    results = []
    for strategy_name, chain in chains.items():
        print(f"Running experiment: {strategy_name}")

        # Create task function
        task = create_enhanced_task_function(chain, strategy_name)

        # Run experiment
        experiment = run_experiment(
            dataset=dataset,
            task=task,
            evaluators=[qa_correctness_evaluator, rag_relevance_evaluator],
            experiment_name=f"{strategy_name}_rag_eval",
            experiment_description=f"Evaluation for {strategy_name} retrieval"
        )

        results.append({
            "strategy": strategy_name,
            "experiment_id": experiment.id,
            "status": "SUCCESS"
        })

    # 4. View results
    print("\nExperiment Results:")
    for result in results:
        print(f"  {result['strategy']}: {result['experiment_id']}")

    print("\nView detailed results in Phoenix UI")

asyncio.run(run_experiments())
```

**Expected Output:**
- 6 experiments (one per strategy)
- QA correctness scores
- RAG relevance scores
- Comparative analysis in Phoenix UI

---

### Pattern 4: Custom Orchestrator

**Scenario:** Create a custom orchestrator for security analysis

```python
from pathlib import Path
from typing import Dict, List
from claude_agent_sdk import AgentDefinition
from ra_orchestrators.base_orchestrator import BaseOrchestrator

class SecurityOrchestrator(BaseOrchestrator):
    def __init__(self):
        super().__init__(domain_name="security")

        # Define output structure
        self.findings_dir = self.output_dir / "findings"
        self.reports_dir = self.output_dir / "reports"

        self.create_output_structure(["findings", "reports"])

    def get_agent_definitions(self) -> Dict[str, AgentDefinition]:
        return {
            "security_scanner": AgentDefinition(
                description="Scans code for security vulnerabilities",
                prompt="""You are a security expert. Analyze the codebase for:
                - SQL injection vulnerabilities
                - XSS vulnerabilities
                - Authentication issues
                - Hardcoded secrets
                - Insecure dependencies

                Write findings to the findings directory.""",
                tools=["Read", "Write", "Grep", "Bash"],
                model="sonnet"
            ),
            "report_writer": AgentDefinition(
                description="Creates security audit report",
                prompt="""You are a security report writer. Create a comprehensive
                security audit report from the findings.""",
                tools=["Read", "Write"],
                model="sonnet"
            )
        }

    def get_allowed_tools(self) -> List[str]:
        return ["Read", "Write", "Grep", "Glob", "Bash"]

    async def run(self):
        # Phase 1: Security Scan
        self.display_phase_header(1, "Security Vulnerability Scan", "🔒")
        await self.execute_phase(
            "vulnerability_scan",
            "security_scanner",
            f"Scan the codebase for security vulnerabilities. Write findings to {self.findings_dir}/",
            self.client
        )

        # Phase 2: Report Generation
        self.display_phase_header(2, "Security Audit Report", "📋")
        await self.execute_phase(
            "audit_report",
            "report_writer",
            f"Read findings from {self.findings_dir}/ and create audit report in {self.reports_dir}/",
            self.client
        )

        # Verify outputs
        expected_files = [
            self.findings_dir / "vulnerabilities.md",
            self.reports_dir / "security_audit.md"
        ]
        await self.verify_outputs(expected_files)

# Usage
async def main():
    orchestrator = SecurityOrchestrator()
    await orchestrator.run_with_client()

import asyncio
asyncio.run(main())
```

**Expected Output:**
- `ra_output/security_20251004_120000/findings/vulnerabilities.md`
- `ra_output/security_20251004_120000/reports/security_audit.md`
- Cost tracking and summary

---

### Pattern 5: Document Loading and Processing

**Scenario:** Load mixed document types with appropriate chunking strategies

```python
import asyncio
from pathlib import Path
from src.langchain_eval_foundations_e2e import (
    setup_environment,
    load_pdf_documents,
    load_markdown_documents
)

async def process_documents():
    config = setup_environment()
    data_dir = Path.cwd() / "data"

    # Load PDFs
    pdf_docs = await load_pdf_documents(data_dir)
    print(f"PDFs: {len(pdf_docs)} pages")

    # Load Markdowns
    md_docs = await load_markdown_documents(data_dir)
    print(f"Markdown: {len(md_docs)} sections")

    # Analyze metadata
    for doc in pdf_docs[:3]:
        print(f"\nPDF: {doc.metadata['document_name']}")
        print(f"Content: {len(doc.page_content)} chars")

    for doc in md_docs[:3]:
        print(f"\nMarkdown: {doc.metadata['Header_1']}")
        print(f"Section: {doc.metadata['Header_2']}")
        print(f"Chunk: {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']}")

    # Combine for vector store
    all_docs = pdf_docs + md_docs
    return all_docs

asyncio.run(process_documents())
```

**Expected Output:**
- PDF documents split by page
- Markdown documents split by H2 sections
- Rich metadata for filtering and analysis

---

## Best Practices

### RAG Evaluation

1. **Vector Store Management**
   - Use separate tables for baseline and semantic chunking strategies
   - Set `overwrite_existing_tables=True` during development
   - Use meaningful table names for multiple experiments
   - Monitor PostgreSQL disk usage with large document sets

2. **Phoenix Observability**
   - Disable auto-instrumentation for large documents (`auto_instrument=False`)
   - Use custom span names for filtering (`run_name`, `span_attributes`)
   - Batch traces for efficiency
   - Regularly check Phoenix UI for performance insights

3. **Retrieval Strategy Selection**
   - Start with naive and semantic for baseline comparison
   - Add BM25 for keyword-heavy queries
   - Use compression for quality over speed
   - Try ensemble for best overall performance
   - Benchmark strategies on your specific use case

4. **Document Loading**
   - Use PDF loading for research papers and books
   - Use Markdown loading for technical documentation
   - Prefer H2-based splitting for Markdown (preserves semantic structure)
   - Sample large document sets for RAGAS (20-50 docs optimal)

5. **Cost Optimization**
   - Use gpt-4.1-mini for cost-effective evaluation
   - Limit testset size during development (5-10 examples)
   - Monitor Phoenix cost tracking per phase
   - Cache embeddings in vector store (avoid re-computing)

### Orchestrator Framework

1. **Orchestrator Design**
   - Inherit from `BaseOrchestrator` for all custom orchestrators
   - Define clear phase boundaries
   - Use descriptive phase names and emojis
   - Implement output verification for critical files

2. **Agent Definition**
   - Create reusable agents in JSON files
   - Use specific, actionable prompts
   - Limit tools to what agent actually needs
   - Include examples in prompts for consistency

3. **Output Management**
   - Use timestamped directories by default
   - Create logical subdirectory structure
   - Verify all expected outputs were created
   - Include metadata files for reproducibility

4. **Error Handling**
   - Use try-except in phase execution
   - Preserve partial results on failures
   - Log errors for debugging
   - Provide clear error messages with solutions

5. **Cost Tracking**
   - Track costs per phase
   - Display summary with cost breakdown
   - Monitor total cost against budget
   - Use cheaper models for large-scale analysis

### Code Quality

1. **Type Hints**
   - Use type hints for all public functions
   - Import from `typing` module
   - Document complex types in docstrings

2. **Async/Await**
   - Use async for I/O-bound operations
   - Await all coroutines
   - Use `asyncio.run()` for top-level execution

3. **Configuration**
   - Use environment variables for secrets
   - Provide sensible defaults
   - Document all configuration options
   - Use dataclasses for structured config

4. **Documentation**
   - Write docstrings for all public APIs
   - Include examples in docstrings
   - Document parameters and return values
   - Link to source files with line numbers

5. **Testing**
   - Test with small datasets first
   - Verify outputs programmatically
   - Use Phoenix UI for visual verification
   - Test error handling paths

---

## Common Gotchas

### RAG Evaluation

1. **RESOURCE_EXHAUSTED Errors**
   - **Cause:** Auto-instrumentation with large documents creates too many spans
   - **Solution:** Set `auto_instrument=False` in `setup_phoenix_tracing()`
   - **Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:122`

2. **Empty RAGAS Testset**
   - **Cause:** Insufficient or low-quality documents
   - **Solution:** Ensure at least 10 diverse document chunks
   - **Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_golden_testset.py:52-66`

3. **BM25 Retriever Errors**
   - **Cause:** No documents loaded for BM25
   - **Solution:** Pass all documents to `create_retrievers()`
   - **Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:150`

4. **Cohere API Errors**
   - **Cause:** Missing or invalid COHERE_API_KEY
   - **Solution:** Set in `.env` file, check API key validity
   - **Source:** `.env.example:8`

5. **PostgreSQL Connection Errors**
   - **Cause:** Docker containers not running
   - **Solution:** Run `docker-compose up -d` before pipeline
   - **Source:** `docker-compose.yml`

### Orchestrator Framework

1. **Import Errors**
   - **Cause:** Running from wrong directory
   - **Solution:** Always run from repository root
   - **Example:** `python -m ra_orchestrators.architecture_orchestrator`

2. **Agent Not Writing Files**
   - **Cause:** Agent prompt doesn't mandate using Write tool
   - **Solution:** Include explicit instruction to use Write tool in prompt
   - **Best Practice:** "IMPORTANT: When asked to write to a file, ALWAYS use the Write tool"

3. **Timestamp Collisions**
   - **Cause:** Multiple orchestrators started in same second
   - **Solution:** Add `time.sleep(2)` between orchestrator creations
   - **Alternative:** Use `use_timestamp=False` for fixed directory names

4. **Missing Abstract Methods**
   - **Cause:** Forgot to implement required BaseOrchestrator methods
   - **Solution:** Implement `get_agent_definitions()`, `get_allowed_tools()`, `run()`
   - **Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:202-227`

5. **Client Not Available in run()**
   - **Cause:** Calling `self.client` before `run_with_client()` sets it
   - **Solution:** Only access `self.client` within `run()` method
   - **Source:** `/home/donbr/lila-graph/lila-research/ra_orchestrators/base_orchestrator.py:291-294`

---

## Appendix

### Dependency Versions

**Core Framework:**
- Python: >=3.13
- langchain: latest
- langchain-postgres: >=0.0.15
- claude-agent-sdk: >=0.1.0

**Evaluation:**
- ragas: latest
- arize-phoenix: latest
- arize-phoenix-otel: latest
- openinference-instrumentation-langchain: latest

**Data Processing:**
- pandas: latest
- pypdf: latest
- rank_bm25: latest

**Visualization:**
- matplotlib: >=3.10.3
- seaborn: >=0.13.2

**Dev Tools:**
- mypy: >=1.16.1
- pytest: >=8.4.1
- ruff: >=0.12.1

**Full dependency list:** `/home/donbr/lila-graph/lila-research/pyproject.toml:7-32`

---

### File Path Reference

All paths in this document use absolute paths from repository root:

**Base Path:** `/home/donbr/lila-graph/lila-research/`

**Key Directories:**
- Source code: `src/`
- Orchestrators: `ra_orchestrators/`
- Agents: `ra_agents/`
- Tools: `ra_tools/`
- Validation: `validation/`
- Scripts: `claude_code_scripts/`
- Data: `data/`
- Outputs: `ra_output/`

**Configuration Files:**
- Environment: `.env` (create from `.env.example`)
- Dependencies: `pyproject.toml`
- Docker: `docker-compose.yml`

---

### Quick Reference

**Start Services:**
```bash
docker-compose up -d
```

**Run Main Pipeline:**
```bash
python src/langchain_eval_foundations_e2e.py
```

**Generate Golden Testset:**
```bash
python src/langchain_eval_golden_testset.py
```

**Run Experiments:**
```bash
python src/langchain_eval_experiments.py
```

**Architecture Analysis:**
```bash
python -m ra_orchestrators.architecture_orchestrator
```

**View Phoenix UI:**
```
http://localhost:6006
```

**Check Service Status:**
```bash
docker ps | grep rag-eval
```

---

## Support and Resources

**Documentation:**
- Component Inventory: `ra_output/architecture_20251003_235103/docs/01_component_inventory.md`
- Architecture Diagrams: `ra_output/architecture_20251003_235103/diagrams/02_architecture_diagrams.md`
- Data Flows: `ra_output/architecture_20251003_235103/docs/03_data_flows.md`

**External Resources:**
- LangChain Documentation: https://python.langchain.com/docs/
- Phoenix Documentation: https://docs.arize.com/phoenix
- RAGAS Documentation: https://docs.ragas.io/
- Claude Agent SDK: https://github.com/anthropics/claude-agent-sdk

**Project Information:**
- Version: 0.1.0
- Python: >=3.13
- License: See repository

---

**Document Version:** 1.0
**Last Updated:** 2025-10-04
**Maintainer:** lila-research project
