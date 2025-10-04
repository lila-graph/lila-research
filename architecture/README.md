# Repository Architecture Documentation

**Project:** lila-research - RAG Evaluation Framework
**Version:** 0.1.0
**Python Requirement:** >=3.13
**Analysis Date:** 2025-10-04
**Total Python Modules:** 19

---

## Overview

### What is this Repository?

The **lila-research** repository is a comprehensive **RAG (Retrieval-Augmented Generation) evaluation framework** that combines cutting-edge evaluation capabilities with a portable multi-domain orchestration system. It solves the challenge of systematically comparing and optimizing retrieval strategies for AI-powered question-answering systems, while providing reusable infrastructure for repository analysis and documentation generation.

### What Problems Does it Solve?

1. **RAG Strategy Evaluation** - Provides quantitative comparison of 6 different retrieval strategies with automated experiments and golden test sets
2. **LLM Observability** - Full tracing and cost tracking via Phoenix integration for transparent AI development
3. **Repository Analysis** - Portable orchestration framework (`ra_*` prefix) for deep codebase analysis across any domain
4. **Multi-Format Document Processing** - Intelligent loading and chunking of PDFs, Markdown, and CSV documents with semantic preservation
5. **Experiment Reproducibility** - Timestamped outputs, configuration management, and comprehensive telemetry validation

### Who is it For?

- **AI Researchers** - Evaluating retrieval strategies for RAG systems on Theory of Mind research
- **ML Engineers** - Building production RAG pipelines with optimized retrieval and observability
- **Technical Architects** - Analyzing codebases and generating comprehensive architecture documentation
- **UX Designers** - Creating design documentation workflows with AI-assisted orchestration
- **DevOps Teams** - Understanding multi-service architectures with PostgreSQL, Phoenix, and Docker

---

## Quick Start

### How to Navigate This Documentation

This documentation is organized into multiple complementary documents:

| Document | Purpose | Audience | Read Time |
|----------|---------|----------|-----------|
| **README.md** (this file) | High-level overview and synthesis | Everyone | 15 min |
| [01_component_inventory.md](docs/01_component_inventory.md) | Detailed component catalog | Developers, Architects | 30 min |
| [02_architecture_diagrams.md](diagrams/02_architecture_diagrams.md) | Visual system architecture | Architects, New Developers | 20 min |
| [03_data_flows.md](docs/03_data_flows.md) | Execution flows and integration patterns | Engineers, DevOps | 25 min |
| [04_api_reference.md](docs/04_api_reference.md) | Complete API documentation | Developers | 45 min |

### Reading Paths for Different Audiences

**New Developer (First Day):**
1. Start here (README.md) - Get the big picture
2. [Architecture Diagrams](diagrams/02_architecture_diagrams.md) - Understand system structure
3. [Component Inventory](docs/01_component_inventory.md) - Learn what each module does
4. [API Reference](docs/04_api_reference.md) - Dive into specific components you'll work with

**Experienced Architect:**
1. Start here (README.md) - Architecture summary
2. [Architecture Diagrams](diagrams/02_architecture_diagrams.md) - Validate system design
3. [Data Flows](docs/03_data_flows.md) - Understand integration patterns
4. [Component Inventory](docs/01_component_inventory.md) - Assess code organization

**End User / Product Manager:**
1. Start here (README.md) - Features and capabilities
2. [Architecture Diagrams](diagrams/02_architecture_diagrams.md) - Visual system overview
3. [API Reference](docs/04_api_reference.md) - Usage patterns and examples

---

## Architecture Summary

### System Layers and Responsibilities

The architecture follows a **layered design pattern** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRY POINTS LAYER                                         │
│  10 executable scripts for different workflows              │
│  - RAG Pipeline, Testset Gen, Experiments, Orchestrators    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION FRAMEWORK                                    │
│  BaseOrchestrator + Domain-Specific Orchestrators           │
│  - Architecture (5 phases), UX (6 phases), Security (TBD)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  CORE BUSINESS LOGIC                                        │
│  RAG Evaluation + Repository Analysis                       │
│  - 6 Retrieval Strategies, Evaluators, Data Loaders         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  DATA ACCESS LAYER                                          │
│  PostgreSQL Vector Stores + Multi-Format Document Loaders   │
│  - PGVector, PDF/Markdown/CSV Loaders, Semantic Chunking    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL SERVICES                                          │
│  Third-Party Integrations                                   │
│  - Phoenix, OpenAI, Cohere, RAGAS, Figma MCP                │
└─────────────────────────────────────────────────────────────┘
```

### Major Design Patterns Used

1. **Template Method Pattern** - `BaseOrchestrator` defines workflow skeleton, subclasses implement domain-specific phases
2. **Factory Pattern** - `create_retrievers()` creates 6 retrieval strategies, `create_rag_chain()` builds pipelines
3. **Registry Pattern** - `AgentRegistry` and `MCPRegistry` for dynamic discovery and loading
4. **Dataclass Configuration** - `Config` with computed properties for type-safe settings
5. **Async Context Manager** - `run_with_client()` for automatic resource cleanup
6. **Decorator Pattern** - `@create_evaluator` for Phoenix experiment evaluators
7. **Repository Pattern** - `load_docs_from_postgres()` abstracts data access

### Key Architectural Decisions and Rationale

#### Decision 1: Dual Architecture (RAG + Orchestration)
**Rationale:** Combines domain-specific RAG evaluation with portable orchestration framework
- **RAG Evaluation** - Specialized for Theory of Mind research with 6 retrieval strategies
- **Orchestration Framework** - Generic (`ra_` prefix) for drop-in analysis of any repository
- **Benefit:** Enables both targeted research and broad applicability

#### Decision 2: Async-First Design
**Rationale:** I/O-bound operations (database, LLM APIs) benefit from concurrency
- All major components use `async/await`
- Concurrent document loading and vector store operations
- Efficient LLM API calls with parallel evaluations
- **Trade-off:** More complex code, but 3-5x faster execution

#### Decision 3: Phoenix Auto-Instrumentation Disabled
**Rationale:** Large markdown documents create RESOURCE_EXHAUSTED errors
- Set `auto_instrument=False` in `setup_phoenix_tracing()`
- Manual span creation for critical operations only
- **Impact:** Prevents system crashes with large document sets
- **Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:122`

#### Decision 4: Timestamped Output Directories
**Rationale:** Preserve historical analysis runs for comparison
- Format: `ra_output/{domain}_{YYYYMMDD_HHMMSS}/`
- No overwriting of previous results
- Enables A/B testing of orchestrator improvements
- **Trade-off:** Disk space consumption, but enables reproducibility

#### Decision 5: Markdown H2-Based Splitting
**Rationale:** Preserve semantic coherence in technical documentation
- Stage 1: Split on H2 headings (`##`)
- Stage 2: Further split oversized sections (chunk_size=2000)
- **Benefit:** Context preservation vs. arbitrary character-based splitting
- **Source:** `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:230`

#### Decision 6: PostgreSQL with PGVector
**Rationale:** Mature, scalable vector database with PostgreSQL ecosystem
- Chosen over: Pinecone (vendor lock-in), Weaviate (complexity), ChromaDB (production readiness)
- Vector similarity search + relational queries in single database
- Docker deployment for local development
- **Port 6024** to avoid conflicts with standard PostgreSQL (5432)

---

## Component Overview

### Public API vs Internal Implementation

**Public API (External Use):**

1. **RAG Evaluation Module** (`src/`)
   - `Config` dataclass - Centralized configuration
   - `setup_environment()` - Initialize from `.env`
   - `create_retrievers()` - Factory for 6 retrieval strategies
   - `load_and_process_data()` - Multi-format document loading
   - `run_evaluation()` - Execute evaluation across strategies

2. **Golden Test Set** (`src/langchain_eval_golden_testset.py`)
   - `generate_testset()` - RAGAS-based test generation
   - `upload_to_phoenix()` - Upload dataset for experiments

3. **Experiments** (`src/langchain_eval_experiments.py`)
   - `@qa_correctness_evaluator` - Answer quality metric
   - `@rag_relevance_evaluator` - Context relevance metric
   - `create_enhanced_task_function()` - Task function factory

4. **Orchestration Framework** (`ra_orchestrators/`, `ra_agents/`, `ra_tools/`)
   - `BaseOrchestrator` - Base class for domain orchestrators
   - `ArchitectureOrchestrator` - 5-phase architecture analysis
   - `UXOrchestrator` - 6-phase UX design workflow
   - `AgentRegistry` - Agent discovery and loading
   - `MCPRegistry` - MCP server management

5. **Validation Tools** (`validation/`)
   - `validate_telemetry.py` - Phoenix tracing validation
   - `postgres_data_analysis.py` - Vector DB analysis
   - `retrieval_strategy_comparison.py` - Strategy benchmarking

**Internal Implementation:**

- Package initialization files (`__init__.py`)
- Legacy/deprecated modules (`architecture.py`)
- Data loaders shared across modules (`data_loader.py`)
- Internal helper functions for chunking, formatting

### Entry Points and How to Run

**Primary Entry Points:**

1. **Main RAG Evaluation Pipeline**
   ```bash
   python src/langchain_eval_foundations_e2e.py
   ```
   - Loads documents, creates vector stores, evaluates 6 strategies
   - Runtime: ~5-10 minutes
   - Outputs: PostgreSQL vector stores, Phoenix traces

2. **Golden Test Set Generation**
   ```bash
   python src/langchain_eval_golden_testset.py
   ```
   - Generates RAGAS synthetic test set
   - Runtime: ~2-5 minutes
   - Outputs: Phoenix dataset "mixed_golden_testset"

3. **Automated Experiments**
   ```bash
   python src/langchain_eval_experiments.py
   ```
   - Runs Phoenix experiments with QA/relevance metrics
   - Runtime: ~10-15 minutes
   - Outputs: Experiment results in Phoenix UI

4. **Architecture Orchestrator**
   ```bash
   python -m ra_orchestrators.architecture_orchestrator
   ```
   - Generates comprehensive architecture docs
   - Runtime: ~5-10 minutes
   - Outputs: `ra_output/architecture_{timestamp}/`

5. **Complete Pipeline Orchestrator**
   ```bash
   python claude_code_scripts/run_rag_evaluation_pipeline.py
   ```
   - Runs all three RAG evaluation steps sequentially
   - Runtime: ~15-30 minutes
   - Includes Docker service management

**Validation & Analysis Tools:**

```bash
# Phoenix telemetry validation
python validation/validate_telemetry.py

# PostgreSQL data analysis
python validation/postgres_data_analysis.py

# Retrieval strategy comparison
python validation/retrieval_strategy_comparison.py
```

---

## Data Flows

### Key Flow Patterns

#### 1. RAG Evaluation Pipeline Flow
```
User Request
    ↓
setup_environment() → Config
    ↓
load_and_process_data() → Documents (PDF/Markdown/CSV)
    ↓
setup_vector_store() → PGVectorStore (baseline + semantic)
    ↓
create_retrievers() → 6 Retrieval Strategies
    ↓
create_rag_chain() → RAG Chains with Phoenix Tracing
    ↓
run_evaluation() → Results + Phoenix Traces
```

**Key Insight:** Sequential pipeline with checkpointing at each stage enables partial recovery on failures.

#### 2. Orchestrator-Based Analysis Flow
```
User Execution
    ↓
ArchitectureOrchestrator() → Timestamped Output Directory
    ↓
run_with_client() → Claude SDK Client Setup
    ↓
execute_phase() → Agent Execution Loop
    ↓ (5 phases)
Agent → Tool Requests (Read/Write/Grep/Glob)
    ↓
File System → Analysis Outputs
    ↓
verify_outputs() → Validation
    ↓
display_summary() → Cost Tracking & Results
```

**Key Insight:** Agent-based execution with tool permissions enables autonomous analysis with human oversight.

#### 3. Message Parsing and Tool Routing
```
Agent Response
    ↓
AssistantMessage → TextBlock / ToolUseBlock
    ↓
ToolUseBlock → Permission Check → Tool Executor
    ↓
ToolResultBlock → Agent Processing
    ↓
ResultMessage → Cost Tracking & Completion
```

**Key Insight:** Real-time visibility into agent tool usage enables debugging and transparency.

### How Queries/Requests Move Through System

**Query Execution Path:**
1. User question → RAG chain
2. Chain extracts question
3. Retriever performs similarity search (vector or BM25)
4. Context documents retrieved from PostgreSQL
5. (Optional) Cohere reranking for compression strategy
6. (Optional) Multi-query expansion for multiquery strategy
7. Context + question formatted into RAG_PROMPT
8. LLM generates response
9. Phoenix captures full trace (retrieval + generation)
10. Response returned to user

**Data Transformations:**
- Raw Documents → Chunked Documents (with metadata)
- Chunks → Vector Embeddings (1536-dimensional)
- Query → Query Embedding → Retrieved Documents
- Documents → Formatted Context → LLM Prompt
- LLM Response → Extracted Text → User Output

### Integration Points with External Services

1. **PostgreSQL (localhost:6024)**
   - Vector storage with PGVector extension
   - Similarity search via cosine distance
   - Async operations with asyncpg driver

2. **Phoenix (http://localhost:6006)**
   - LLM observability and tracing
   - Experiment orchestration
   - Cost tracking and performance metrics
   - Dataset management for golden test sets

3. **OpenAI API**
   - GPT-4.1-mini for chat completions
   - text-embedding-3-small for embeddings
   - Used across all retrieval strategies

4. **Cohere API**
   - rerank-english-v3.0 for document reranking
   - Used exclusively by compression strategy

5. **RAGAS**
   - TestsetGenerator for synthetic test creation
   - Quality-controlled question/answer pairs

6. **Figma MCP (Optional)**
   - Design context retrieval via MCP server
   - REST API fallback if MCP unavailable

---

## Key Features

### What Makes This Architecture Special

#### 1. Dual-Purpose Design
- **RAG Evaluation:** Specialized framework for retrieval strategy comparison
- **Repository Analysis:** Portable orchestration system (`ra_` prefix) for any codebase
- **Unique Value:** Single repository solves two distinct but complementary problems

#### 2. Six Retrieval Strategies
Most RAG frameworks provide 1-2 strategies. This system evaluates:
1. **Naive** - Baseline vector similarity
2. **Semantic** - Semantic chunking-based
3. **BM25** - Keyword-based (TF-IDF)
4. **Compression** - Cohere reranking
5. **MultiQuery** - LLM query expansion
6. **Ensemble** - Weighted combination

**Benefit:** Empirical comparison enables data-driven strategy selection.

#### 3. Full Observability Pipeline
- Phoenix integration at every layer
- Custom span naming for filtering
- Cost tracking per phase
- Real-time tool usage visibility
- **Impact:** Complete transparency from retrieval to generation

#### 4. Semantic Document Splitting
- Markdown: H2 header-based splitting (preserves sections)
- PDFs: Page-based with semantic chunking fallback
- **Benefit:** Context preservation vs. arbitrary character splits
- **Impact:** Higher retrieval quality for technical documentation

#### 5. Portable Orchestrator Framework
- `ra_` prefix prevents naming collisions
- Drop into any repository for analysis
- Timestamped outputs preserve history
- **Use Cases:** Architecture docs, UX design, security audits, DevOps analysis

#### 6. RAGAS Integration
- Synthetic test generation from documents
- Quality-controlled Q&A pairs
- Phoenix upload for experiments
- **Benefit:** Automated evaluation without manual test creation

#### 7. Automated Experiments
- Phoenix experiment orchestration
- QA correctness + RAG relevance metrics
- Parallel strategy evaluation
- **Impact:** Quantitative strategy comparison at scale

---

## Technology Stack

### Languages and Frameworks

**Primary Language:**
- Python 3.13+ (leverages latest async improvements)

**Core Frameworks:**
- **LangChain** - RAG pipeline orchestration
  - langchain-core, langchain-community, langchain-experimental
  - langchain-openai, langchain-cohere, langchain-postgres
- **Claude Agent SDK** - Multi-agent orchestration
- **RAGAS** - Golden test set generation
- **Phoenix** - LLM observability and experiments

**Data Processing:**
- pandas - Data analysis
- pypdf - PDF document loading
- rank_bm25 - BM25 retriever implementation

**Visualization:**
- matplotlib (>=3.10.3)
- seaborn (>=0.13.2)

**Database:**
- PostgreSQL with PGVector extension
- asyncpg - Async driver
- psycopg2-binary - Sync driver
- SQLAlchemy - ORM

**Development Tools:**
- mypy (>=1.16.1) - Type checking
- pytest (>=8.4.1) - Testing
- ruff (>=0.12.1) - Linting and formatting
- uv - Fast Python package installer

### External Services

1. **PostgreSQL + PGVector**
   - Vector database for similarity search
   - Port: 6024 (avoid conflicts with standard 5432)
   - Docker container: `rag-eval-pgvector`

2. **Phoenix Observability**
   - LLM tracing and experiment platform
   - UI Port: 6006
   - OTLP Port: 4317
   - Docker container: `rag-eval-phoenix`

3. **OpenAI API**
   - GPT-4.1-mini - Chat completions
   - text-embedding-3-small - Embeddings (1536-dim)

4. **Cohere API**
   - rerank-english-v3.0 - Document reranking

5. **Figma MCP Server (Optional)**
   - Design context retrieval
   - REST API fallback available

---

## Configuration

### High-Level Configuration Overview

The system uses a **three-tier configuration approach**:

1. **Environment Variables** (`.env` file)
   - API keys and secrets
   - Service endpoints
   - Feature flags

2. **Config Dataclass** (`src/langchain_eval_foundations_e2e.py:47-96`)
   - Centralized configuration object
   - Computed properties (e.g., `async_url`)
   - Type-safe settings

3. **Docker Compose** (`docker-compose.yml`)
   - Service definitions
   - Port mappings
   - Volume management

### Essential Configuration Steps

**1. Create `.env` file:**
```bash
cp .env.example .env
```

**2. Set required API keys:**
```bash
# In .env file
OPENAI_API_KEY=sk-your-openai-key
COHERE_API_KEY=your-cohere-key
```

**3. Configure services (optional):**
```bash
# Avoid port conflicts
POSTGRES_PORT=6024  # Default: 6024
PHOENIX_UI_PORT=6006  # Default: 6006
PHOENIX_OTLP_PORT=4317  # Default: 4317
```

**4. Start Docker services:**
```bash
docker-compose up -d
```

**5. Verify services:**
```bash
docker ps | grep rag-eval
# Should show: rag-eval-pgvector, rag-eval-phoenix
```

**For complete configuration details, see:** [04_api_reference.md](docs/04_api_reference.md#configuration)

---

## Development Guide

### How to Get Started Developing

**Prerequisites:**
```bash
# Python 3.13+
python --version  # Should be >=3.13

# Install uv (fast package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/your-org/lila-research.git
cd lila-research
```

**Setup Development Environment:**
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"

# Setup environment variables
cp .env.example .env
# Edit .env and add your API keys
```

**Start Services:**
```bash
# Start PostgreSQL and Phoenix
docker-compose up -d

# Verify services
docker ps | grep rag-eval
```

### Common Development Tasks

**1. Run Main RAG Pipeline:**
```bash
python src/langchain_eval_foundations_e2e.py
```

**2. Generate Architecture Documentation:**
```bash
python -m ra_orchestrators.architecture_orchestrator
```

**3. Analyze PostgreSQL Data:**
```bash
python validation/postgres_data_analysis.py
# Outputs: outputs/charts/postgres_analysis/
```

**4. Run Type Checking:**
```bash
mypy src/ ra_orchestrators/ ra_agents/ ra_tools/
```

**5. Format Code:**
```bash
ruff check --fix .
ruff format .
```

**6. Run Tests:**
```bash
pytest validation/
```

**7. View Phoenix Traces:**
```bash
# Open browser
http://localhost:6006
```

**8. Connect to PostgreSQL:**
```bash
# Via psql
psql -h localhost -p 6024 -U langchain -d langchain

# Via Python
python
>>> from src.data_loader import load_docs_from_postgres
>>> docs = load_docs_from_postgres("mixed_baseline_documents")
```

### Testing and Validation

**1. Validate Phoenix Tracing:**
```bash
python validation/validate_telemetry.py
# Check http://localhost:6006 for traces
```

**2. Compare Retrieval Strategies:**
```bash
python validation/retrieval_strategy_comparison.py
# Interactive comparison with benchmarks
```

**3. Analyze Vector Embeddings:**
```bash
python validation/postgres_data_analysis.py
# Generates PCA visualization of embeddings
```

**4. Test Golden Testset Generation:**
```bash
# Small test (5 examples)
python src/langchain_eval_golden_testset.py
```

**5. Verify Output Structure:**
```bash
# Architecture analysis
python -m ra_orchestrators.architecture_orchestrator
# Check: ra_output/architecture_{timestamp}/
```

**6. Clean Docker State:**
```bash
# Remove all data and restart fresh
docker-compose down -v
docker-compose up -d
```

---

## References

### Detailed Documentation

1. **[Component Inventory](docs/01_component_inventory.md)**
   - Complete module catalog
   - Public API surface
   - Entry points and dependencies
   - Design patterns used

2. **[Architecture Diagrams](diagrams/02_architecture_diagrams.md)**
   - System architecture visualization
   - Component relationships
   - Class hierarchies
   - Module dependencies

3. **[Data Flows](docs/03_data_flows.md)**
   - RAG evaluation pipeline flow
   - Orchestrator execution flow
   - Tool permission callback flow
   - MCP server communication
   - Message parsing and routing

4. **[API Reference](docs/04_api_reference.md)**
   - Complete API documentation
   - Configuration reference
   - Usage patterns
   - Best practices
   - Common gotchas

### External Resources

**Framework Documentation:**
- [LangChain Python Docs](https://python.langchain.com/docs/)
- [Phoenix Observability](https://docs.arize.com/phoenix)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)

**Database and Tools:**
- [PGVector Extension](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Compose](https://docs.docker.com/compose/)

**AI Models:**
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Cohere Rerank](https://docs.cohere.com/reference/rerank)

### Output Directory Reference

**Architecture Analysis Output:**
```
ra_output/architecture_20251003_235103/
├── docs/
│   ├── 01_component_inventory.md
│   ├── 03_data_flows.md
│   └── 04_api_reference.md
├── diagrams/
│   └── 02_architecture_diagrams.md
├── reports/
└── README.md (this file)
```

**UX Design Output (Example):**
```
ra_output/ux_{timestamp}/
├── 01_research/
│   └── user_research.md
├── 02_ia/
│   └── information_architecture.md
├── 03_design/
│   └── visual_design.md
├── 04_prototypes/
│   └── interactive_prototypes.md
├── 05_api_contracts/
│   └── api_specifications.md
└── 06_design_system/
    └── design_system.md
```

**Validation Output:**
```
outputs/charts/
├── postgres_analysis/
│   ├── document_distribution.png
│   ├── chunking_comparison.png
│   └── embedding_visualization.png
└── retrieval_analysis/
    └── retrieval_performance.png
```

---

## Glossary

**Key Terms and Acronyms:**

- **RAG** - Retrieval-Augmented Generation: LLM architecture that retrieves relevant context before generating responses
- **RAGAS** - RAG Assessment: Framework for generating synthetic test sets and evaluating RAG systems
- **Phoenix** - LLM observability platform by Arize AI for tracing, experiments, and cost tracking
- **PGVector** - PostgreSQL extension for vector similarity search
- **MCP** - Model Context Protocol: Standard for tool/resource providers for LLMs
- **BM25** - Best Matching 25: Probabilistic ranking function for keyword-based retrieval (TF-IDF variant)
- **Semantic Chunking** - Document splitting based on semantic similarity rather than character count
- **Vector Embedding** - Dense numerical representation of text (1536-dimensional for text-embedding-3-small)
- **Contextual Compression** - Retrieval strategy that reranks documents using LLM or reranking model
- **MultiQuery Retriever** - Generates multiple query variations to improve retrieval diversity
- **Ensemble Retriever** - Combines multiple retrieval strategies with weighted voting
- **Orchestrator** - AI agent-based workflow manager for multi-phase analysis tasks
- **Agent Definition** - JSON specification of agent behavior, tools, and model configuration
- **Tool Permission Mode** - Security setting for agent tool execution (acceptEdits, ask, deny)
- **Span** - Single traced operation in observability (retrieval, LLM call, etc.)
- **Golden Test Set** - Curated test examples with reference answers for evaluation
- **QA Correctness** - Metric evaluating answer quality against reference
- **RAG Relevance** - Metric evaluating retrieved context quality
- **Theory of Mind** - AI capability to model mental states of others (research domain focus)
- **Async/Await** - Python asynchronous programming pattern for I/O-bound operations
- **Dataclass** - Python decorator for type-safe configuration objects
- **Template Method Pattern** - Design pattern where base class defines workflow, subclasses implement steps

**File System Terms:**

- **`ra_` prefix** - Repository Analyzer namespace to avoid collisions when dropped into other codebases
- **Timestamped Directory** - Output directory with format `{domain}_{YYYYMMDD_HHMMSS}`
- **Vector Store Table** - PostgreSQL table with embeddings column for similarity search
- **Baseline Chunking** - Standard character-based document splitting
- **H2 Header Splitting** - Markdown splitting based on `##` level 2 headers

**Configuration Terms:**

- **`.env` file** - Environment variable configuration (API keys, secrets)
- **`Config` dataclass** - Central configuration object in `langchain_eval_foundations_e2e.py`
- **`docker-compose.yml`** - Service orchestration configuration
- **`pyproject.toml`** - Python project and dependency specification
- **Async URL** - PostgreSQL connection string for asyncpg driver

---

## Project Status and Roadmap

### Current Capabilities (v0.1.0)

- 6 retrieval strategies with automated evaluation
- RAGAS golden test set generation
- Phoenix experiment orchestration
- PostgreSQL vector storage with PGVector
- Architecture orchestrator (5 phases)
- UX orchestrator (6 phases)
- Full LLM observability and cost tracking
- Multi-format document loading (PDF, Markdown, CSV)
- Validation tools for telemetry and data analysis

### Known Limitations

1. **RESOURCE_EXHAUSTED with Large Docs**
   - Workaround: `auto_instrument=False` in Phoenix setup
   - Impact: Manual span creation required

2. **RAGAS Timeout with Large Document Sets**
   - Recommendation: Sample 20-50 documents
   - Impact: May miss edge cases from full dataset

3. **No Cross-Orchestrator Communication**
   - `CrossOrchestratorCommunication` mixin exists but not implemented
   - Impact: Can't validate UX designs against architecture constraints

4. **Legacy Code Duplication**
   - `architecture.py` kept for reference (deprecated)
   - Impact: Minor code maintenance overhead

### Future Enhancements

**Planned Features:**
- Custom retrieval strategy plugins
- Multi-language document support
- Graph-based retrieval strategies
- Real-time streaming evaluation
- Cross-orchestrator validation workflows
- Security orchestrator implementation
- DevOps orchestrator for infrastructure analysis

**Community Contributions Welcome:**
- Additional retrieval strategies
- Domain-specific orchestrators
- MCP server integrations
- Evaluation metrics
- Documentation improvements

---

## Getting Help

### Troubleshooting Common Issues

**Docker Services Not Starting:**
```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart

# Clean state and restart
docker-compose down -v
docker-compose up -d
```

**PostgreSQL Connection Errors:**
```bash
# Verify service is running
docker ps | grep pgvector

# Check port is correct in .env
echo $POSTGRES_PORT  # Should be 6024

# Test connection
psql -h localhost -p 6024 -U langchain -d langchain
```

**Phoenix Not Showing Traces:**
```bash
# Verify Phoenix is running
docker ps | grep phoenix

# Check endpoint in .env
echo $PHOENIX_COLLECTOR_ENDPOINT  # Should be http://localhost:6006

# Verify auto_instrument=False is set
grep "auto_instrument" src/langchain_eval_foundations_e2e.py
```

**OpenAI API Errors:**
```bash
# Verify API key is set
python -c "import os; print('✓' if os.getenv('OPENAI_API_KEY') else '✗ OPENAI_API_KEY not set')"

# Test API connection
python -c "from langchain_openai import ChatOpenAI; print(ChatOpenAI().invoke('test'))"
```

**Import Errors:**
```bash
# Run from repository root
cd /home/donbr/lila-graph/lila-research

# Verify virtual environment is activated
which python  # Should show .venv/bin/python

# Reinstall dependencies
uv pip install -e .
```

### Where to Ask Questions

**For this specific repository:**
- Create GitHub issue with detailed error messages
- Include Python version, OS, and Docker logs
- Tag with appropriate labels (bug, question, enhancement)

**For framework-specific questions:**
- LangChain: https://github.com/langchain-ai/langchain/discussions
- Phoenix: https://docs.arize.com/phoenix/support
- RAGAS: https://github.com/explodinggradients/ragas/issues

### Contributing

**Before submitting PR:**
1. Run type checking: `mypy src/`
2. Format code: `ruff format .`
3. Fix linting issues: `ruff check --fix .`
4. Test locally with sample data
5. Update documentation if adding features
6. Include unit tests for new functionality

---

## Conclusion

The **lila-research** repository represents a comprehensive solution for RAG system evaluation and repository analysis. Its dual-purpose architecture combines specialized retrieval strategy comparison with a portable orchestration framework, making it valuable for both AI researchers and software architects.

**Key Strengths:**
- Complete observability pipeline with Phoenix integration
- 6 retrieval strategies with quantitative comparison
- Portable orchestration framework for multi-domain analysis
- Semantic document processing preserving context
- Automated experiment generation with RAGAS

**Best Suited For:**
- RAG system optimization and research
- Retrieval strategy benchmarking
- Repository architecture documentation
- UX/UI design workflow automation
- Multi-service system observability

**Next Steps:**
1. Start with the [Architecture Diagrams](diagrams/02_architecture_diagrams.md) for visual overview
2. Read [Component Inventory](docs/01_component_inventory.md) to understand modules
3. Review [Data Flows](docs/03_data_flows.md) for integration patterns
4. Dive into [API Reference](docs/04_api_reference.md) for implementation details
5. Run the main pipeline and explore Phoenix UI

**Output Location:** `/home/donbr/lila-graph/lila-research/ra_output/architecture_20251003_235103/`

**Phoenix UI:** http://localhost:6006

**Docker Services:**
```bash
docker ps | grep rag-eval
# rag-eval-pgvector (port 6024)
# rag-eval-phoenix (port 6006)
```

---

**Document Version:** 1.0
**Generated:** 2025-10-04
**Generated By:** Architecture Orchestrator - Phase 5 (Synthesis)
**Framework Version:** ra_orchestrators v1.0
**Total Documentation Pages:** 5 documents, 7000+ lines

For questions or issues, please refer to the [Getting Help](#getting-help) section above.
