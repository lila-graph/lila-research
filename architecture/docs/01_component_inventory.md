# Component Inventory

**Project:** lila-research - RAG Evaluation Framework
**Analysis Date:** 2025-10-03
**Total Python Modules:** 19
**Code Structure:** Multi-domain research framework with RAG evaluation, orchestration, and validation components

## Overview

This codebase is a comprehensive RAG (Retrieval-Augmented Generation) evaluation framework built on LangChain, Phoenix, and Claude Agent SDK. It combines three major domains:

1. **RAG Evaluation Pipeline** - Core evaluation framework for retrieval strategies
2. **Repository Analyzer Framework** - Portable multi-domain orchestration system
3. **Validation & Analysis Tools** - Data analysis and telemetry validation utilities

---

## Public API

### Core RAG Evaluation Modules

#### 1. `src/langchain_eval_foundations_e2e.py`
**Purpose:** Main RAG evaluation pipeline orchestrator
**Lines:** 479
**Entry Point:** `main()` (line 390)

**Public Classes:**
- `Config` (lines 47-96) - Centralized configuration management
  - Properties: API keys, database settings, model configurations
  - Methods: `async_url` property for database connection string

**Public Functions:**
- `setup_environment() -> Config` (lines 99-113)
  Purpose: Initialize environment and load configuration from .env file

- `setup_phoenix_tracing(config: Config)` (lines 116-124)
  Purpose: Configure Phoenix observability with auto-instrumentation disabled to prevent resource exhaustion

- `async setup_vector_store(config: Config, table_name: str, embeddings) -> PGVectorStore` (lines 127-141)
  Purpose: Reusable vector store initialization for PostgreSQL

- `create_retrievers(baseline_vectorstore, semantic_vectorstore, all_docs, llm) -> Dict[str, Any]` (lines 143-174)
  Purpose: Create all 6 retrieval strategies (naive, semantic, BM25, compression, multiquery, ensemble)

- `async load_pdf_documents(data_dir: Path) -> List` (lines 176-208)
  Purpose: Load PDF documents with metadata for theory of mind research

- `async load_markdown_documents(data_dir: Path) -> List` (lines 211-286)
  Purpose: Load markdown documents with H2 header-based semantic splitting

- `async load_and_process_data(config: Config) -> List` (lines 289-357)
  Purpose: Orchestrate loading of CSV, PDF, and Markdown data sources

- `create_rag_chain(retriever, llm, method_name: str)` (lines 359-371)
  Purpose: Create RAG chain with Phoenix auto-tracing and method tagging

- `async run_evaluation(question: str, chains: Dict[str, Any]) -> Dict[str, str]` (lines 374-387)
  Purpose: Execute evaluation across all retrieval strategies

**Key Constants:**
- `RAG_PROMPT` (lines 40-44) - Centralized prompt template for RAG chains

---

#### 2. `src/langchain_eval_golden_testset.py`
**Purpose:** RAGAS-based golden test set generation for evaluation
**Lines:** 199
**Entry Point:** `main()` (line 130)

**Public Functions:**
- `generate_testset(docs: list, llm, embeddings, testset_size: int = 10)` (lines 16-42)
  Purpose: Generate golden test set using RAGAS TestsetGenerator
  Returns: RAGAS testset with synthetic questions and reference answers

- `upload_to_phoenix(golden_testset, dataset_name: str = "mixed_golden_testset") -> dict` (lines 45-128)
  Purpose: Convert RAGAS testset to Phoenix format and upload
  Returns: Dictionary with upload status and dataset metadata

**Key Features:**
- Handles both old and new RAGAS API formats (lines 46-80)
- Maps RAGAS columns to Phoenix expected format (lines 92-107)
- Validates document count and provides helpful error messages (lines 52-66)

---

#### 3. `src/langchain_eval_experiments.py`
**Purpose:** Automated Phoenix experiments for retrieval strategy comparison
**Lines:** 282
**Entry Point:** `main()` (line 113)

**Public Evaluators:**
- `@create_evaluator(name="qa_correctness_score")` (lines 24-59)
  Purpose: Evaluate answer correctness against ground truth using QAEvaluator

- `@create_evaluator(name="rag_relevance_score")` (lines 61-92)
  Purpose: Evaluate retrieved context relevance using RelevanceEvaluator

**Public Functions:**
- `create_enhanced_task_function(strategy_chain, strategy)` (lines 96-111)
  Purpose: Factory for task functions that capture retrieval context for evaluators
  Returns: Task function compatible with Phoenix experiments

**Key Integration Points:**
- Phoenix experiment orchestration (lines 215-270)
- Six retrieval strategies evaluation (lines 204-211)
- QA correctness and RAG relevance metrics (line 250)

---

#### 4. `src/data_loader.py`
**Purpose:** PostgreSQL document loader utility
**Lines:** 46

**Public Functions:**
- `load_docs_from_postgres(table_name: str = "mixed_baseline_documents") -> List[Document]` (lines 10-46)
  Purpose: Load documents from PostgreSQL table into LangChain Document objects
  Returns: List of Document objects with content and metadata

**Usage Pattern:**
```python
from data_loader import load_docs_from_postgres
docs = load_docs_from_postgres("mixed_baseline_documents")
```

---

### Repository Analyzer Framework (Portable Orchestration System)

#### 5. `ra_orchestrators/base_orchestrator.py`
**Purpose:** Base framework for domain-specific orchestrators
**Lines:** 357
**Public Prefix:** `ra_` to avoid naming collisions when dropped into other repositories

**Public Classes:**

##### `BaseOrchestrator` (lines 30-309) - ABC for all orchestrators
**Constructor Parameters:**
- `domain_name: str` - Name of domain (e.g., 'architecture', 'ux', 'devops')
- `output_base_dir: Path` - Base directory for outputs (default: ra_output)
- `show_tool_details: bool` - Display detailed tool usage (default: True)
- `use_timestamp: bool` - Append timestamp to output directory (default: True)

**Key Methods:**
- `create_output_structure(subdirs: Optional[List[str]])` (lines 75-85)
  Purpose: Create output directory structure with optional subdirectories

- `display_message(msg, show_tools: bool = True)` (lines 87-121)
  Purpose: Display message content with full visibility into tool usage

- `display_phase_header(phase_number: int, phase_name: str, emoji: str)` (lines 130-141)
  Purpose: Display formatted phase header for progress tracking

- `track_phase_cost(phase_name: str, cost: float)` (lines 142-150)
  Purpose: Track cost for specific phase execution

- `mark_phase_complete(phase_name: str)` (lines 152-158)
  Purpose: Mark phase as completed in tracking system

- `async verify_outputs(expected_files: List[Path]) -> bool` (lines 160-183)
  Purpose: Verify all expected outputs were created

- `display_summary()` (lines 185-200)
  Purpose: Display orchestrator run summary with costs and completion status

- `async execute_phase(phase_name, agent_name, prompt, client)` (lines 229-253)
  Purpose: Execute single phase of workflow with cost tracking

- `create_client_options(permission_mode, cwd) -> ClaudeAgentOptions` (lines 255-277)
  Purpose: Create Claude SDK client options for orchestrator

- `async run_with_client()` (lines 279-308)
  Purpose: Run orchestrator with automatic client setup and teardown

**Abstract Methods (must be implemented by subclasses):**
- `get_agent_definitions() -> Dict[str, AgentDefinition]` (lines 202-209)
- `get_allowed_tools() -> List[str]` (lines 211-218)
- `async run()` (lines 220-227)

**Attributes:**
- `output_dir: Path` - Timestamped output directory (e.g., ra_output/architecture_20251003_235103/)
- `total_cost: float` - Total cost tracking across all phases
- `phase_costs: Dict[str, float]` - Per-phase cost breakdown
- `completed_phases: List[str]` - List of completed phase names

##### `CrossOrchestratorCommunication` (lines 311-357) - Mixin for inter-orchestrator communication
**Purpose:** Enable orchestrators to invoke each other for cross-domain validation

**Methods:**
- `register_orchestrator(name: str, orchestrator: BaseOrchestrator)` (lines 318-325)
- `async invoke_orchestrator(orchestrator_name: str, phase_name: str, context: Dict) -> Any` (lines 327-356)

---

#### 6. `ra_orchestrators/architecture_orchestrator.py`
**Purpose:** Comprehensive repository architecture analysis orchestrator
**Lines:** 316
**Entry Point:** `main()` (line 307)

**Public Class:**

##### `ArchitectureOrchestrator(BaseOrchestrator)` (lines 20-305)
**Purpose:** Execute 5-phase architecture analysis workflow

**Phase Methods:**
- `async phase_1_component_inventory()` (lines 114-145)
  Output: `docs/01_component_inventory.md`

- `async phase_2_architecture_diagrams()` (lines 147-184)
  Output: `diagrams/02_architecture_diagrams.md`

- `async phase_3_data_flows()` (lines 186-221)
  Output: `docs/03_data_flows.md`

- `async phase_4_api_documentation()` (lines 223-242)
  Output: `docs/04_api_reference.md`

- `async phase_5_synthesis()` (lines 244-284)
  Output: `README.md` (synthesis document)

**Agent Definitions:**
- `analyzer` - Code structure, patterns, Mermaid diagrams (lines 65-81)
- `doc-writer` - Technical documentation expert (lines 83-99)

**Directory Structure:**
```
ra_output/architecture_{timestamp}/
├── docs/
│   ├── 01_component_inventory.md
│   ├── 03_data_flows.md
│   └── 04_api_reference.md
├── diagrams/
│   └── 02_architecture_diagrams.md
├── reports/
└── README.md
```

---

#### 7. `ra_orchestrators/ux_orchestrator.py`
**Purpose:** Comprehensive UX/UI design workflow orchestrator
**Lines:** 623
**Entry Point:** `main()` (line 611)

**Public Class:**

##### `UXOrchestrator(BaseOrchestrator)` (lines 21-609)
**Purpose:** Execute 6-phase UX design workflow

**Constructor Parameters:**
- `project_name: str` - Name of project being designed
- Additional parameters inherited from BaseOrchestrator

**Phase Methods:**
- `async phase_1_ux_research()` (lines 169-217)
  Output: `01_research/user_research.md` (personas, journeys, competitive analysis)

- `async phase_2_information_architecture()` (lines 219-274)
  Output: `02_ia/information_architecture.md` (sitemaps, navigation, wireframes)

- `async phase_3_visual_design()` (lines 276-344)
  Output: `03_design/visual_design.md` (design system, mockups, accessibility)

- `async phase_4_interactive_prototyping()` (lines 346-413)
  Output: `04_prototypes/interactive_prototypes.md` (flows, micro-interactions)

- `async phase_5_api_contract_design()` (lines 415-489)
  Output: `05_api_contracts/api_specifications.md` (data models, endpoints)

- `async phase_6_design_system_documentation()` (lines 491-586)
  Output: `06_design_system/design_system.md` (component library, tokens)

**Agent Definitions:**
- `ux-researcher` - User research, personas, journey mapping (lines 76-92)
- `ia-architect` - Information architecture, sitemaps (lines 94-110)
- `ui-designer` - Visual design, mockups (lines 112-131)
- `prototype-developer` - Interactive prototypes, user flows (lines 133-152)

**Directory Structure:**
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

---

#### 8. `ra_agents/registry.py`
**Purpose:** Agent definition discovery and loading system
**Lines:** 100

**Public Class:**

##### `AgentRegistry` (lines 10-100)
**Purpose:** Discover and load agent definitions from JSON files

**Methods:**
- `discover_agents(domain: Optional[str] = None) -> Dict[str, str]` (lines 22-43)
  Purpose: Discover all available agent definition files
  Returns: Dictionary mapping agent names to file paths

- `load_agent(agent_name: str, domain: Optional[str] = None) -> Optional[AgentDefinition]` (lines 45-80)
  Purpose: Load agent definition from JSON file with caching

- `load_domain_agents(domain: str) -> Dict[str, AgentDefinition]` (lines 82-99)
  Purpose: Load all agents for specific domain

**Usage Pattern:**
```python
from ra_agents.registry import AgentRegistry
registry = AgentRegistry()
agents = registry.discover_agents(domain='architecture')
analyzer = registry.load_agent('analyzer', domain='architecture')
```

---

#### 9. `ra_tools/mcp_registry.py`
**Purpose:** MCP (Model Context Protocol) server discovery and management
**Lines:** 153

**Public Class:**

##### `MCPRegistry` (lines 8-153)
**Purpose:** Discover and validate MCP tool availability

**Methods:**
- `discover_mcp_servers() -> Dict[str, Dict[str, Any]]` (lines 16-53)
  Purpose: Auto-discover available MCP servers
  Returns: Dictionary of MCP servers with capabilities

- `is_server_available(server_name: str) -> bool` (lines 55-67)
  Purpose: Check if MCP server is available

- `get_server_tools(server_name: str) -> List[str]` (lines 69-81)
  Purpose: Get list of tools provided by MCP server

- `validate_tool_availability(tool_name: str) -> bool` (lines 83-96)
  Purpose: Validate if specific tool is available

- `get_configuration_requirements(server_name: str) -> Optional[Dict]` (lines 98-128)
  Purpose: Get configuration requirements for MCP server

- `get_fallback_options(tool_name: str) -> List[str]` (lines 130-152)
  Purpose: Get fallback options if tool is unavailable

**Supported MCP Servers:**
- Figma - Design context retrieval (lines 27-32)
- v0 - Vercel UI generation (lines 33-38)
- Sequential Thinking - Advanced reasoning (lines 39-44)
- Playwright - Browser automation (lines 45-50)

---

#### 10. `ra_tools/figma_integration.py`
**Purpose:** Figma MCP and REST API integration wrapper
**Lines:** 157

**Public Class:**

##### `FigmaIntegration` (lines 7-157)
**Purpose:** Wrapper for Figma MCP server and REST API

**Methods:**
- `is_available() -> bool` (lines 15-21)
  Purpose: Check if Figma integration is configured

- `get_design_context(file_id: str) -> Optional[Dict]` (lines 23-57)
  Purpose: Get design context from Figma file

- `export_to_code(component_id: str, framework: str = "react") -> Optional[str]` (lines 59-84)
  Purpose: Export Figma component to code

- `create_component(spec: Dict[str, Any]) -> Optional[str]` (lines 86-101)
  Purpose: Create Figma component from specification

- `get_setup_instructions() -> str` (lines 103-157)
  Purpose: Get comprehensive setup instructions for Figma integration

---

### Validation & Analysis Tools

#### 11. `validation/validate_telemetry.py`
**Purpose:** Phoenix telemetry validation and LLM observability examples
**Lines:** 247
**Entry Point:** `main()` (line 214)

**Public Functions:**
- `setup_phoenix()` (lines 23-32)
  Purpose: Configure Phoenix tracing with auto-instrumentation

- `test_simple_chain()` (lines 34-48)
  Purpose: Test simple chain with Phoenix tracing

- `test_complex_chains()` (lines 50-83)
  Purpose: Test math calculator and text analyzer chains

- `test_rag_components()` (lines 85-136)
  Purpose: Test embedding generation and mock RAG pipeline

- `test_error_handling()` (lines 138-167)
  Purpose: Test error handling and debugging

- `test_streaming()` (lines 169-185)
  Purpose: Test streaming responses with Phoenix

**Key Features:**
- Demonstrates custom span naming (line 62, 78)
- Mock RAG pipeline for testing (lines 107-136)
- Error handling patterns (lines 156-167)
- Real-time streaming visualization (lines 183-184)

---

#### 12. `validation/postgres_data_analysis.py`
**Purpose:** PostgreSQL vector database analysis and visualization
**Lines:** 304
**Entry Point:** `main()` (line 255)
**Output Directory:** `outputs/charts/postgres_analysis/`

**Public Functions:**
- `ensure_output_directory()` (lines 28-32)
  Purpose: Create output directory for chart files

- `setup_connection()` (lines 34-47)
  Purpose: Setup PostgreSQL database connection
  Returns: SQLAlchemy engine

- `analyze_baseline_table(engine)` (lines 49-89)
  Purpose: Analyze baseline documents table structure and content
  Output: `document_distribution.png`

- `analyze_content(df)` (lines 91-109)
  Purpose: Analyze document content statistics

- `compare_chunking_strategies(engine, df_baseline)` (lines 111-154)
  Purpose: Compare baseline vs semantic chunking approaches
  Output: `chunking_comparison.png`

- `analyze_embeddings(df)` (lines 156-210)
  Purpose: Analyze and visualize embeddings using PCA
  Output: `embedding_visualization.png`

- `run_sample_queries(engine)` (lines 212-253)
  Purpose: Run sample SQL queries on document database

**Generated Visualizations:**
- Document distribution by source (seaborn countplot)
- Baseline vs semantic chunk length histograms
- 2D PCA embedding visualization with document clustering

---

#### 13. `validation/retrieval_strategy_comparison.py`
**Purpose:** Interactive comparison of 6 retrieval strategies
**Lines:** 362
**Entry Point:** `main()` (line 318)
**Output Directory:** `outputs/charts/retrieval_analysis/`

**Public Functions:**
- `ensure_output_directory()` (lines 35-39)
- `setup_environment()` (lines 41-55)
  Purpose: Setup environment and Phoenix tracing

- `async initialize_models_and_stores()` (lines 57-86)
  Purpose: Initialize LLM models and vector stores

- `load_documents_for_bm25(baseline_vectorstore)` (lines 88-97)
  Purpose: Load documents for BM25 retriever

- `create_retrievers(llm, baseline_vectorstore, semantic_vectorstore, all_docs)` (lines 99-142)
  Purpose: Create all 6 retrieval strategies
  Returns: Dictionary of retrievers

- `async compare_retrievers(query: str, retrievers: Dict) -> pd.DataFrame` (lines 144-173)
  Purpose: Compare all retrieval strategies for given query

- `display_results(df: pd.DataFrame)` (lines 175-187)
  Purpose: Display retrieval results in formatted output

- `async compare_rag_responses(query: str, retrievers: Dict, llm) -> Dict[str, str]` (lines 189-219)
  Purpose: Generate RAG responses using each strategy

- `async benchmark_retrievers(query: str, retrievers: Dict, runs: int = 3) -> pd.DataFrame` (lines 221-243)
  Purpose: Benchmark retrieval strategies for speed
  Output: `retrieval_performance.png`

**Retrieval Strategies Analyzed:**
1. Naive vector search
2. Semantic chunking vector search
3. BM25 keyword retriever
4. Contextual compression with Cohere reranking
5. Multi-query retriever
6. Ensemble retriever (combines all strategies)

---

#### 14. `claude_code_scripts/run_rag_evaluation_pipeline.py`
**Purpose:** RAG evaluation pipeline orchestrator script
**Lines:** 443
**Entry Point:** `main()` (line 326)

**Public Functions:**
- `setup_logging(verbose: bool = False) -> logging.Logger` (lines 38-71)
  Purpose: Configure logging with timestamps and file output

- `check_environment() -> Tuple[bool, List[str]]` (lines 74-123)
  Purpose: Validate environment setup and API keys

- `check_docker_availability() -> Tuple[bool, str]` (lines 126-160)
  Purpose: Check Docker installation and daemon status

- `run_service_check() -> Tuple[bool, str]` (lines 163-183)
  Purpose: Run service check script to verify container status

- `start_docker_services(logger: logging.Logger) -> bool` (lines 186-235)
  Purpose: Start PostgreSQL and Phoenix Docker services

- `execute_pipeline_step(step_name, script_path, logger, timeout=600) -> Tuple[bool, str]` (lines 238-286)
  Purpose: Execute single pipeline step with error handling

- `print_summary(results: Dict, total_time: float, logger: logging.Logger)` (lines 289-323)
  Purpose: Print comprehensive pipeline execution summary

**Pipeline Steps (lines 414-418):**
1. Main E2E Pipeline (600s timeout)
2. Golden Test Set Generation (300s timeout)
3. Automated Experiments (900s timeout)

**Command-Line Arguments:**
- `--skip-services` - Skip Docker service startup
- `--verbose` - Enable debug logging
- `--testset-size N` - Override golden test set size

---

## Internal Implementation

### Package Initialization Modules

#### 15. `src/__init__.py`
**Purpose:** RAG Evaluation Foundation - Core Implementation marker
**Lines:** 1
**Type:** Package initialization with documentation comment

#### 16. `ra_orchestrators/__init__.py`
**Purpose:** Multi-domain agent orchestration system package
**Lines:** 5
**Exports:** `BaseOrchestrator`

#### 17. `ra_agents/__init__.py`
**Purpose:** Agent library for multi-domain orchestration
**Lines:** 5
**Exports:** `AgentRegistry`

#### 18. `ra_tools/__init__.py`
**Purpose:** Tool integration layer for orchestrators
**Lines:** 6
**Exports:** `MCPRegistry`, `FigmaIntegration`

---

### Legacy/Deprecated Modules

#### 19. `ra_orchestrators/architecture.py`
**Purpose:** Legacy standalone architecture analyzer
**Lines:** 363
**Status:** Deprecated - Refactored into `architecture_orchestrator.py`

**Note:** This is the original implementation before the BaseOrchestrator framework was extracted. It contains the same functionality as `architecture_orchestrator.py` but without the base class pattern. Kept for reference.

**Key Differences:**
- Standalone implementation without base class inheritance
- Manual phase management and cost tracking
- Hardcoded output directory structure (lines 27-30)
- Direct agent definitions without registry pattern (lines 296-330)

---

## Entry Points

### Primary Entry Points (Executable Scripts)

1. **Main RAG Evaluation Pipeline**
   - File: `/home/donbr/lila-graph/lila-research/src/langchain_eval_foundations_e2e.py:479`
   - Entry: `if __name__ == "__main__": asyncio.run(main())`
   - Purpose: Execute end-to-end RAG evaluation with 6 retrieval strategies
   - Runtime: ~5-10 minutes
   - Outputs: PostgreSQL vector stores, Phoenix traces

2. **Golden Test Set Generation**
   - File: `/home/donbr/lila-graph/lila-research/src/langchain_eval_golden_testset.py:198`
   - Entry: `if __name__ == "__main__": main()`
   - Purpose: Generate RAGAS-based golden test set
   - Runtime: ~2-5 minutes
   - Outputs: Phoenix dataset "mixed_golden_testset"

3. **Automated Experiments**
   - File: `/home/donbr/lila-graph/lila-research/src/langchain_eval_experiments.py:282`
   - Entry: `if __name__ == "__main__": asyncio.run(main())`
   - Purpose: Run Phoenix experiments across all retrieval strategies
   - Runtime: ~10-15 minutes
   - Outputs: Phoenix experiment results with QA correctness and RAG relevance scores

4. **Architecture Orchestrator**
   - File: `/home/donbr/lila-graph/lila-research/ra_orchestrators/architecture_orchestrator.py:313`
   - Entry: `if __name__ == "__main__": asyncio.run(main())`
   - Purpose: Generate comprehensive repository architecture documentation
   - Runtime: ~5-10 minutes
   - Outputs: `ra_output/architecture_{timestamp}/` with 5 analysis documents

5. **UX Orchestrator**
   - File: `/home/donbr/lila-graph/lila-research/ra_orchestrators/ux_orchestrator.py:620`
   - Entry: `if __name__ == "__main__": asyncio.run(main())`
   - Purpose: Generate comprehensive UX design documentation
   - Runtime: ~10-20 minutes
   - Outputs: `ra_output/ux_{timestamp}/` with 6 design documents

6. **Legacy Architecture Analyzer**
   - File: `/home/donbr/lila-graph/lila-research/ra_orchestrators/architecture.py:362`
   - Entry: `if __name__ == "__main__": asyncio.run(main())`
   - Purpose: Legacy standalone architecture analysis (deprecated)
   - Note: Use `architecture_orchestrator.py` instead

7. **Telemetry Validation**
   - File: `/home/donbr/lila-graph/lila-research/validation/validate_telemetry.py:247`
   - Entry: `if __name__ == "__main__": main()`
   - Purpose: Validate Phoenix telemetry with example chains
   - Runtime: ~2 minutes
   - Outputs: Phoenix traces for validation

8. **PostgreSQL Data Analysis**
   - File: `/home/donbr/lila-graph/lila-research/validation/postgres_data_analysis.py:304`
   - Entry: `if __name__ == "__main__": main()`
   - Purpose: Analyze vector database content and embeddings
   - Runtime: ~1-2 minutes
   - Outputs: 3 PNG charts in `outputs/charts/postgres_analysis/`

9. **Retrieval Strategy Comparison**
   - File: `/home/donbr/lila-graph/lila-research/validation/retrieval_strategy_comparison.py:362`
   - Entry: `if __name__ == "__main__": asyncio.run(main())`
   - Purpose: Interactive comparison of retrieval strategies
   - Runtime: ~5 minutes
   - Outputs: Performance chart in `outputs/charts/retrieval_analysis/`

10. **Pipeline Orchestrator**
    - File: `/home/donbr/lila-graph/lila-research/claude_code_scripts/run_rag_evaluation_pipeline.py:443`
    - Entry: `if __name__ == "__main__": main()`
    - Purpose: Orchestrate complete RAG evaluation pipeline
    - Runtime: ~15-30 minutes
    - Outputs: Logs in `logs/`, all pipeline artifacts

---

### Module-Level Entry Points (Import Targets)

**Core RAG Evaluation:**
- `from src.data_loader import load_docs_from_postgres`
- `from src.langchain_eval_foundations_e2e import Config, setup_environment`
- `from src.langchain_eval_golden_testset import generate_testset, upload_to_phoenix`

**Orchestration Framework:**
- `from ra_orchestrators.base_orchestrator import BaseOrchestrator`
- `from ra_orchestrators import BaseOrchestrator` (via __init__.py)
- `from ra_agents import AgentRegistry` (via __init__.py)
- `from ra_tools import MCPRegistry, FigmaIntegration` (via __init__.py)

**Validation Tools:**
- `from validation.postgres_data_analysis import setup_connection, analyze_baseline_table`
- `from validation.retrieval_strategy_comparison import create_retrievers`

---

## Component Dependencies

### External Dependencies (from pyproject.toml)

**Core Framework:**
- `langchain`, `langchain-core`, `langchain-community`, `langchain-experimental`
- `langchain-openai` - OpenAI LLM and embeddings
- `langchain-cohere` - Cohere reranking
- `langchain-postgres>=0.0.15` - PostgreSQL vector store

**Evaluation & Observability:**
- `ragas` - Golden test set generation
- `arize-phoenix`, `arize-phoenix-otel` - LLM observability
- `openinference-instrumentation-langchain` - Phoenix + LangChain integration

**Claude Agent SDK:**
- `claude-agent-sdk>=0.1.0` - Multi-agent orchestration framework

**Database & Data:**
- `sqlalchemy` - Database ORM
- `asyncpg` - Async PostgreSQL driver
- `psycopg2-binary` - Sync PostgreSQL driver
- `pandas` - Data analysis

**Utilities:**
- `python-dotenv` - Environment variable management
- `requests` - HTTP client
- `pypdf` - PDF document loading
- `rank_bm25` - BM25 retriever implementation
- `rapidfuzz` - Fuzzy string matching

**Visualization:**
- `matplotlib>=3.10.3`
- `seaborn>=0.13.2`

**Dev Tools:**
- `mypy>=1.16.1` - Type checking
- `pytest>=8.4.1` - Testing
- `ruff>=0.12.1` - Linting and formatting

---

## Key Design Patterns

### 1. Configuration Management Pattern
**File:** `src/langchain_eval_foundations_e2e.py:47-96`

Uses dataclass for centralized configuration with property methods for derived values (e.g., `async_url` property on line 95).

### 2. Factory Pattern for Retrievers
**File:** `src/langchain_eval_foundations_e2e.py:143-174`

`create_retrievers()` function creates all 6 retrieval strategies using consistent interface pattern.

### 3. Abstract Base Class Pattern for Orchestrators
**File:** `ra_orchestrators/base_orchestrator.py:30-309`

Base class with abstract methods `get_agent_definitions()`, `get_allowed_tools()`, and `run()` ensures consistent orchestrator implementation.

### 4. Registry Pattern for Agents
**File:** `ra_agents/registry.py:10-100`

Agent registry with caching (line 20, 56-58) for efficient agent definition loading from JSON files.

### 5. Async Context Manager Pattern
**File:** `ra_orchestrators/base_orchestrator.py:279-308`

`run_with_client()` uses async context manager for automatic client setup and teardown (lines 291-294).

### 6. Evaluator Decorator Pattern
**File:** `src/langchain_eval_experiments.py:24-92`

`@create_evaluator` decorator wraps custom evaluation functions for Phoenix experiments.

### 7. Pipeline Orchestration Pattern
**File:** `claude_code_scripts/run_rag_evaluation_pipeline.py:326-439`

Sequential pipeline execution with checkpointing and error recovery (lines 420-431).

---

## API Surface Analysis

### Public API (External Use)

**RAG Evaluation:**
- 6 retrieval strategies (naive, semantic, BM25, compression, multiquery, ensemble)
- RAGAS golden test set generation
- Phoenix experiment orchestration
- Document loaders (PDF, Markdown, CSV)

**Orchestration Framework:**
- `BaseOrchestrator` - Portable base class for any domain
- `ArchitectureOrchestrator` - 5-phase architecture analysis
- `UXOrchestrator` - 6-phase UX design workflow
- `AgentRegistry` - Agent discovery and loading
- `MCPRegistry` - MCP tool availability checking

**Validation Tools:**
- PostgreSQL data analysis with visualizations
- Retrieval strategy benchmarking
- Phoenix telemetry validation examples

### Internal API (Inter-Module Use)

**Data Loading:**
- `load_docs_from_postgres()` - Used by `langchain_eval_golden_testset.py` and `langchain_eval_experiments.py`
- `load_pdf_documents()`, `load_markdown_documents()` - Used internally by `load_and_process_data()`

**Configuration:**
- `Config` dataclass - Shared across RAG evaluation modules
- `setup_environment()` - Reusable environment setup

**Orchestration Infrastructure:**
- `display_message()`, `display_phase_header()` - Progress visualization
- `execute_phase()` - Phase execution with cost tracking
- `verify_outputs()` - Output validation

---

## Architectural Observations

### 1. Multi-Domain Architecture
The codebase cleanly separates three major concerns:
- RAG evaluation (src/)
- Orchestration framework (ra_orchestrators/, ra_agents/, ra_tools/)
- Validation and analysis (validation/)

### 2. Portable Design
The `ra_` prefix pattern makes the orchestration framework portable across repositories without naming collisions. Designed to be "dropped into any repository for deep analysis" (per CLAUDE.md).

### 3. Timestamped Outputs
All orchestrators use timestamped output directories (e.g., `ra_output/architecture_20251003_235103/`) to prevent overwriting previous runs and enable historical analysis.

### 4. Observability-First
Phoenix integration throughout enables comprehensive LLM observability:
- Auto-instrumentation (disabled for large docs to prevent resource exhaustion)
- Custom span names for filtering
- Cost tracking per phase
- Trace visualization at http://localhost:6006

### 5. Async-First Design
All orchestrators and pipeline components use async/await for:
- Concurrent database operations
- Efficient LLM API calls
- Scalable agent execution

### 6. Configuration over Code
- JSON agent definitions instead of hardcoded prompts
- Environment variable configuration
- Dataclass-based settings

---

## Version Information

**Project Name:** rag-eval-foundations
**Version:** 0.1.0
**Python Requirement:** >=3.13
**License:** Not specified in pyproject.toml

---

## File Path Reference

All paths referenced in this document are absolute paths from repository root:
- `/home/donbr/lila-graph/lila-research/`

For relative imports within the codebase, use Python module notation:
- `from src.data_loader import load_docs_from_postgres`
- `from ra_orchestrators import BaseOrchestrator`
- `from ra_agents import AgentRegistry`
