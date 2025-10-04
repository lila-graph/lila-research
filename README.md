# 🚀 RAG Evaluation Toolkit: Research-Grade Pipeline for LangChain & RAGAS

A complete 3-stage pipeline for evaluating and benchmarking retrieval strategies on technical and academic documents. Built for researchers and engineers who need objective metrics to compare RAG approaches.

## 🔬 Current Configuration: Theory of Mind Research

This toolkit is currently configured to evaluate retrieval strategies on **Theory of Mind** research documents - specifically technical documentation about the Lila system architecture for persistent AI personas with psychological realism.

**Current Data** (`/data/`):
- `theory-of-mind-ai-agents.pdf` - Academic research on AI agent cognition
- Lila system architecture Markdown documents (temporal design, graph-based memory, psychological models)
- Technical specifications for AI persona development

**Easily adaptable** to any research domain by replacing documents in `/data/` directory and updating the `research_domain` configuration.

## 📚 What This Toolkit Provides

- **6 Retrieval Strategies**: Naive vector search, semantic chunking, BM25, contextual compression, multi-query, and ensemble approaches
- **Automated Evaluation**: RAGAS golden test set generation and Phoenix experiment tracking
- **Complete Observability**: Full Phoenix tracing with OpenTelemetry instrumentation
- **Production-Ready**: PostgreSQL + pgvector for scalable vector search
- **Domain-Agnostic**: Easily adapt for medical, legal, scientific, or any technical documents

## 📋 Prerequisites

- Python 3.13+ (required by the project)
- Basic understanding of LLMs and embeddings
- Familiarity with async Python (we use asyncio)
- ~$5 in API credits (OpenAI + Cohere)
- Docker installed and running

## 🎯 Pipeline Architecture

This toolkit implements a complete 3-stage RAG evaluation pipeline:

1. **Stage 1 - Infrastructure**: Vector database setup, document ingestion, and retrieval strategy implementation
2. **Stage 2 - Test Generation**: Automated RAGAS golden test set creation with LLM-synthesized questions and answers
3. **Stage 3 - Evaluation**: Systematic benchmarking with QA correctness, relevance metrics, and Phoenix experiment tracking

**Result**: Objective, reproducible metrics showing which retrieval strategy performs best for your document corpus.

## 🎯 The Complete 3-Stage Pipeline

This toolkit implements a production-ready evaluation pipeline that progresses through three stages:

### 🏭 Stage 1: Foundation & Infrastructure
**Script**: `langchain_eval_foundations_e2e.py`
- Sets up PostgreSQL with pgvector for hybrid search
- Implements 6 different retrieval strategies
- Provides side-by-side comparison with Phoenix tracing
- **You learn**: How different retrieval methods work and when to use each

### 🧪 Stage 2: Golden Test Set Generation  
**Script**: `langchain_eval_golden_testset.py`
- Uses RAGAS to automatically generate diverse test questions
- Creates ground-truth answers and reference contexts
- Uploads datasets to Phoenix for experiment tracking
- **You get**: A reusable test set for consistent evaluation

### 📊 Stage 3: Automated Evaluation
**Script**: `langchain_eval_experiments.py`
- Runs all strategies against the golden test set
- Calculates QA correctness and relevance metrics
- Provides quantitative rankings and performance data
- **You discover**: Which strategy objectively performs best

🔄 **All three stages work together** to give you a complete evaluation workflow from setup to metrics!

---

## 🚀 Quick Start (for the impatient)

### Option A: One-Command Pipeline (Recommended)

```bash
# 1. Clone and setup
git clone <repo-url>
cd rag-eval-foundations
cp .env.example .env  # Edit with your API keys

# 2. Install dependencies
uv venv --python 3.13 && source .venv/bin/activate
uv sync

# 3. Run the complete pipeline
python claude_code_scripts/run_rag_evaluation_pipeline.py
```

The orchestration script will:
- ✅ Validate your environment and API keys
- 🐳 Start Docker services (PostgreSQL + Phoenix)
- 🔄 Execute all 3 pipeline steps in correct order
- 📊 Generate comprehensive evaluation results

### Option B: Manual Step-by-Step

```bash
# 1. Clone and setup
git clone <repo-url>
cd rag-eval-foundations
cp .env.example .env  # Edit with your API keys

# 2. Start services
docker-compose up -d  # Or use individual docker run commands below

# 3. Install and run manually
uv venv --python 3.13 && source .venv/bin/activate
uv sync
python src/langchain_eval_foundations_e2e.py
python src/langchain_eval_golden_testset.py
python src/langchain_eval_experiments.py
```

## 🌉 Next Steps: Deepen Your Understanding

Now that you've got the pipeline running, here's where to go next:

### 🧪 **For Hands-On Exploration**
- **[Validation Scripts](validation/README.md)**: Interactive tools to explore your data and compare strategies
  - `postgres_data_analysis.py`: Visualize embeddings and chunking strategies
  - `retrieval_strategy_comparison.py`: Benchmark and compare all 6 strategies
  - `validate_telemetry.py`: Understand Phoenix tracing in depth

### 🚀 **For Production Readiness**
- Learn about [RAGAS golden test sets](https://docs.ragas.io/) for automated evaluation
- Explore [Phoenix documentation](https://docs.arize.com/phoenix) for advanced observability
- Check out [LangChain's retriever docs](https://python.langchain.com/docs/modules/data_connection/retrievers/) for custom implementations

## 🎯 Pipeline Orchestration Script

The `claude_code_scripts/run_rag_evaluation_pipeline.py` script provides a comprehensive, repeatable process for executing all 3 pipeline steps with proper error handling and logging.

### Features
- **🔍 Environment Validation**: Checks .env file, API keys, and dependencies
- **🐳 Service Management**: Automatically starts Docker services if needed
- **📋 Step-by-Step Execution**: Runs all 3 scripts in correct dependency order
- **📊 Comprehensive Logging**: Detailed logs with timestamps and progress tracking
- **❌ Error Handling**: Graceful failure recovery and clear error messages

### Usage Examples

```bash
# Standard execution (recommended)
python claude_code_scripts/run_rag_evaluation_pipeline.py

# Skip Docker service management (if already running)
python claude_code_scripts/run_rag_evaluation_pipeline.py --skip-services

# Enable verbose debug logging
python claude_code_scripts/run_rag_evaluation_pipeline.py --verbose

# Get help
python claude_code_scripts/run_rag_evaluation_pipeline.py --help
```

### Pipeline Steps Executed

1. **Main E2E Pipeline** (`langchain_eval_foundations_e2e.py`)
   - Loads research documents (PDFs, Markdown, optional CSVs)
   - Creates PostgreSQL vector stores with semantic + baseline chunking
   - Implements 6 retrieval strategies
   - Generates Phoenix traces for all operations

2. **Golden Test Set Generation** (`langchain_eval_golden_testset.py`)
   - Uses RAGAS to auto-generate evaluation questions from documents
   - Creates ground-truth answers and reference contexts
   - Uploads test set to Phoenix for experiment tracking

3. **Automated Experiments** (`langchain_eval_experiments.py`)
   - Runs systematic evaluation on all strategies
   - Calculates QA correctness and relevance scores
   - Creates detailed experiment reports in Phoenix

### Logs and Output

The script creates detailed logs in the `logs/` directory with timestamps. All output includes:
- ✅ Success indicators for each step
- ⏱️ Execution time tracking  
- 🔗 Direct links to Phoenix UI for viewing results
- 📊 Summary statistics and experiment IDs

## 🛠️ Pre-Flight Checklist

### Step 1: Gather Your Supplies

This project uses `uv`, an extremely fast Python package and project manager.

1.  **Install `uv`**

    If you don't have `uv` installed, open your terminal and run the official installer:

    ```bash
    # Install uv (macOS & Linux)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    For Windows and other installation methods, please refer to the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation).

2.  **Create Environment & Install Dependencies**

    With `uv` installed, you can create a virtual environment and install all the necessary packages from `pyproject.toml` in two commands:

    ```bash
    # Create a virtual environment with Python 3.13+
    uv venv --python 3.13

    # Activate the virtual environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (CMD):
    # .venv\Scripts\activate.bat

    # Install dependencies into the virtual environment
    uv sync
    ```

*If you're new to `uv`, think of `uv venv` as a replacement for `python -m venv` and `uv sync` as a much faster version of `pip install -r requirements.txt`.*

### Step 2: Secret Agent Setup
Create a `.env` file (because hardcoding API keys is how we end up on r/ProgrammerHumor):

```bash
OPENAI_API_KEY=sk-your-actual-key-not-this-placeholder
COHERE_API_KEY=your-cohere-key-goes-here
PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
# Optional:
HUGGINGFACE_TOKEN=hf_your_token_here
PHOENIX_CLIENT_HEADERS='...'  # For cloud Phoenix instances
```

📝 **Note:** See `.env.example` for a complete template with all supported variables.

**Pro tip:** Yes, you need both keys. Yes, they cost money. Yes, it's worth it. Think of it as buying premium gas for your AI Ferrari.

---

## 🐳 Docker: Your New Best Friends

**Quick heads-up:** We're using interactive mode (`-it --rm`) for easy cleanup - when you kill these containers, all data vanishes. Perfect for demos, terrible if you want to keep anything. For persistent setups, use `docker-compose` instead.

### Friend #1: PostgreSQL + pgvector (The Data Vault)


```bash
docker run -it --rm --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  pgvector/pgvector:pg16
```

*This is your vector database. It's like a regular database, but it can do math with meanings. Fancy stuff.*

### Friend #2: Phoenix Observability (The All-Seeing Eye)
```bash
export TS=$(date +"%Y%m%d_%H%M%S")
docker run -it --rm --name phoenix-container \
  -e PHOENIX_PROJECT_NAME="retrieval-comparison-${TS}" \
  -p 6006:6006 \
  -p 4317:4317 \
  arizephoenix/phoenix:latest
```

*Phoenix watches everything your AI does and tells you where it went wrong. It's like having a really helpful, non-judgmental therapist for your code.*

⚠️ **Port Notes:** 
- Port 6006: Phoenix UI (view traces here)
- Port 4317: OpenTelemetry collector (receives trace data)

### Alternative: Use Docker Compose (Easier!)
```bash
docker-compose up -d
```
This starts both PostgreSQL and Phoenix with the correct settings.

---

## 🚦 Running the Pipeline

### Launch Sequence
```bash
python src/langchain_eval_foundations_e2e.py
```

**What happens:**
1. 📥 Loads research documents from `/data/` directory
2. 🗄️ Creates vector stores in PostgreSQL with pgvector
3. 🔍 Implements and tests 6 retrieval strategies
4. 📊 Compares performance across strategies
5. 🕵️ Sends traces to Phoenix UI at `http://localhost:6006`

**Expected runtime:** 2-5 minutes depending on document corpus size

---

## 🚨 When Things Go Sideways (And They Will)

### The Greatest Hits of Failure

**"ModuleNotFoundError: No module named 'whatever'"**
- *Translation:* You forgot to install something
- *Solution:* Make sure you ran `uv sync` after activating your venv
- *Encouragement:* Even senior developers forget to activate their venv

**"Connection refused" or "Port already in use"**
- *Translation:* Docker containers aren't happy
- *Solution:* 
  ```bash
  docker ps  # Check what's running
  docker logs pgvector-container  # Check PostgreSQL logs
  docker logs phoenix-container   # Check Phoenix logs
  # If ports are taken:
  lsof -i :6024  # Check what's using PostgreSQL port
  lsof -i :6006  # Check what's using Phoenix port
  ```
- *Encouragement:* Docker is like a moody teenager—sometimes you just need to restart everything

**"Invalid API key" or "Rate limit exceeded"**
- *Translation:* OpenAI/Cohere is giving you the cold shoulder
- *Solution:* Check your `.env` file, verify your API keys have credits
- *Encouragement:* At least the error is clear! Better than "something went wrong" 🤷

**"Async this, await that, event loop already running"**
- *Translation:* Python's async system is having an existential crisis
- *Solution:* Restart your Python session, try again
- *Encouragement:* Async programming is hard. If it was easy, we'd all be doing it

---

## 🆘 Emergency Protocols

### When All Else Fails: The AI Debugging Lifeline

Copy your error message and ask an AI assistant:
> "I'm running a RAG evaluation pipeline with LangChain, PostgreSQL/pgvector, and Phoenix. I'm getting this error: [paste error]. The code compares 6 retrieval strategies on research documents. What's the issue?"

**Why this works:** AI assistants excel at debugging when given proper context about your stack and objectives.

### The Nuclear Option: Start Fresh
```bash
# Kill all Docker containers
docker kill $(docker ps -q)

# Clear Python cache (sometimes helps with import issues)
find . -type d -name __pycache__ -exec rm -rf {} +

# Start over with containers
# (Re-run the docker commands from above)
```

---

## ✅ Success Indicators

### Pipeline Completed Successfully When:

- ✅ Scripts run without errors through all 3 stages
- ✅ Phoenix UI at `http://localhost:6006` shows comprehensive traces
- ✅ PostgreSQL contains vector stores with embedded documents
- ✅ Golden test set generated and uploaded to Phoenix
- ✅ Experiment results show comparative metrics across all 6 strategies

### What You've Built:

- ✅ **Complete 3-stage evaluation pipeline** from infrastructure to automated metrics
- ✅ **6 retrieval strategies** implemented and benchmarked
- ✅ **RAGAS golden test sets** for consistent, repeatable evaluation
- ✅ **Automated scoring** with QA correctness and relevance metrics
- ✅ **Phoenix observability** tracking every operation and experiment
- ✅ **Production-ready foundation** adaptable to any research domain

### Complete Toolkit Components:
- **Stage 1**: Infrastructure setup and strategy implementation
- **Stage 2**: RAGAS-powered golden test set generation
- **Stage 3**: Automated experiments with objective metrics
- **Validation Scripts**: Interactive tools for data exploration and strategy comparison
- **Phoenix Integration**: Full observability and experiment tracking

---

## 🚀 Next Steps

Now that you have a working evaluation pipeline:

1. **Adapt to Your Research Domain**:
   - Replace documents in `/data/` directory
   - Update `research_domain` in configuration
   - Generate domain-specific test questions

2. **Customize Evaluation**:
   - Add domain-specific metrics
   - Extend golden test set size
   - Create custom evaluators

3. **Optimize for Production**:
   - Scale to larger document corpora
   - Fine-tune retrieval weights based on metrics
   - Integrate with CI/CD pipelines

4. **Explore Retrieval Strategies**:
   - Understand trade-offs between speed and accuracy
   - Experiment with ensemble weights
   - Add custom retrieval implementations

---

## 🔄 Adapting to Different Research Domains

This toolkit is designed to work with **any research domain**. Here's how to adapt it:

### Step 1: Replace Your Documents
```bash
# Clear existing data (keeping structure)
rm -f data/*.pdf data/*.md data/*.csv

# Add your documents
cp /path/to/your/research/docs/* data/
```

### Step 2: Update Configuration
Edit `src/langchain_eval_foundations_e2e.py`:
```python
@dataclass
class Config:
    # Update research domain
    research_domain: str = "your_domain"  # e.g., "medical_research", "legal_analysis"

    # Configure data loaders
    load_pdfs: bool = True
    load_markdowns: bool = True
    load_csvs: bool = False
```

### Step 3: Generate Domain-Specific Test Set
```bash
# Adjust test set size for your corpus
export GOLDEN_TESTSET_SIZE=20
python src/langchain_eval_golden_testset.py
```

### Example Configurations

**Medical Research:**
```python
research_domain: str = "medical_research"
# Data: Clinical trials, research papers, treatment guidelines
# Test queries: "What are contraindications for...", "Describe the mechanism of..."
```

**Legal Documents:**
```python
research_domain: str = "legal_analysis"
# Data: Case law, statutes, legal briefs
# Test queries: "What precedent governs...", "How does statute X define..."
```

**Scientific Papers:**
```python
research_domain: str = "scientific_research"
# Data: Journal articles, conference papers, technical reports
# Test queries: "What methodology was used for...", "What were the key findings..."
```

**Current (Theory of Mind):**
```python
research_domain: str = "theory_of_mind"
# Data: AI agent architecture docs, psychological AI models
# Test queries: "How does the Lila system implement ToM?", "What is temporal architecture?"
```

---

## 📚 Additional Resources

### Understanding the Code
- **Main Scripts:**
  - `langchain_eval_foundations_e2e.py` - Stage 1: Foundation & infrastructure
  - `langchain_eval_golden_testset.py` - Stage 2: RAGAS golden test set generation
  - `langchain_eval_experiments.py` - Stage 3: Automated evaluation & metrics
  - `data_loader.py` - Utilities for loading data

## 🔍 Validation & Analysis Tools

The `validation/` directory contains interactive scripts for exploring and validating the RAG system components.

### Prerequisites for Validation Scripts

```bash
# 1. Ensure services are running
docker-compose up -d

# 2. Run the main pipeline first to populate data
python claude_code_scripts/run_rag_evaluation_pipeline.py
```

### Available Validation Scripts

#### 1. PostgreSQL Data Analysis

```bash
python validation/postgres_data_analysis.py
```
**Purpose:** Comprehensive analysis of the vector database
- Analyzes document distribution by source type and research domain
- Compares baseline vs semantic chunking strategies
- Generates PCA visualization of embeddings
- **Outputs:** Creates 3 PNG charts in `outputs/charts/postgres_analysis/`

#### 2. Phoenix Telemetry Validation

```bash
python validation/validate_telemetry.py
```
**Purpose:** Demonstrates Phoenix OpenTelemetry tracing integration
- Tests various LLM chain patterns with tracing
- Shows streaming responses with real-time trace updates
- Validates token usage and latency tracking
- **View traces:** http://localhost:6006

#### 3. Interactive Retrieval Strategy Comparison

```bash
python validation/retrieval_strategy_comparison.py  
```
**Purpose:** Interactive comparison of all 6 retrieval strategies
- Compares naive, semantic, BM25, compression, multiquery, and ensemble strategies
- Runs performance benchmarks across strategies
- Demonstrates query-specific strategy strengths
- **Outputs:** Performance visualization in `outputs/charts/retrieval_analysis/`

### Validation Script Features

- ✅ **Phoenix Integration:** All scripts include OpenTelemetry tracing
- 📊 **Visualization:** Generates charts and performance metrics  
- 🔧 **Interactive:** Real-time comparison and analysis capabilities
- 📝 **Documentation:** Each script includes detailed output explanations

**📖 Detailed Instructions:** See [`validation/README.md`](validation/README.md) for comprehensive usage guide and troubleshooting.

### Cost Estimates

- **OpenAI:** ~$0.50-$2.00 per full run (depending on data size)
- **Cohere:** ~$0.10-$0.50 for reranking
- **Total:** Budget $5 for experimentation

### Performance Benchmarks

- Data loading: 30-60 seconds
- Embedding generation: 1-2 minutes for ~100 reviews
- Retrieval comparison: 30-60 seconds
- Total runtime: 2-5 minutes

### Glossary

- **RAG**: Retrieval-Augmented Generation - enhancing LLM responses with retrieved context
- **Embeddings**: Vector representations of text for semantic search
- **BM25**: Best Matching 25 - a keyword-based ranking algorithm
- **Semantic Search**: Finding similar content based on meaning, not just keywords
- **Phoenix**: Open-source LLM observability platform by Arize
- **pgvector**: PostgreSQL extension for vector similarity search
- **RAGAS**: Framework for evaluating RAG pipelines

---

## 📞 Emergency Contacts
- **Docker Issues:** `docker logs container-name`
- **Python Issues:** Your friendly neighborhood AI assistant
- **Existential Crisis:** Remember, even PostgreSQL had bugs once
- **Success Stories:** Share them! The community loves a good victory lap

*P.S. If this guide helped you succeed, pay it forward by helping the next intrepid adventurer who's staring at the same error messages you just conquered.*

---

## 📚 Appendix: Useful Links
- **[uv Documentation](https://docs.astral.sh/uv/)**: Learn more about the fast Python package and project manager used in this guide.
