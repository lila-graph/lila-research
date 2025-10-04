# langchain_eval_golden_testset.py

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from data_loader import load_docs_from_postgres
from langchain_eval_foundations_e2e import Config, setup_environment

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

def generate_testset(
    docs: list, llm, embeddings, testset_size: int = 10
):
    print(f"🔧 Setting up RAGAS TestsetGenerator...")
    print(f"   Documents: {len(docs)}")
    print(f"   Target testset size: {testset_size}")

    # Show sample document content
    if docs:
        print(f"   Sample doc[0] content length: {len(docs[0].page_content)} chars")
        print(f"   Sample doc[0] preview: {docs[0].page_content[:200]}...")

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    print(f"🎯 Generating testset (this may take several minutes)...")
    try:
        golden_testset = generator.generate_with_langchain_docs(
            documents=docs, testset_size=testset_size
        )
        print(f"✅ Generation complete. Samples created: {len(golden_testset.samples)}")
    except Exception as e:
        print(f"❌ RAGAS generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    return golden_testset


def upload_to_phoenix(golden_testset, dataset_name: str = "mixed_golden_testset") -> dict:
    # Convert testset to DataFrame - handle both old and new RAGAS API
    if hasattr(golden_testset, 'to_pandas'):
        testset_df = golden_testset.to_pandas()
        if testset_df.empty:
            raise ValueError(
                "No samples generated in golden testset. RAGAS generation produced empty DataFrame. "
                "Ensure you have sufficient documents loaded (at least 10+ chunks recommended)."
            )
    elif hasattr(golden_testset, 'samples'):
        # New RAGAS API - convert samples to DataFrame manually
        samples = golden_testset.samples
        if not samples:
            raise ValueError(
                "No samples generated in golden testset. RAGAS generation produced 0 samples. "
                "This typically happens when:\n"
                "  1. Too few documents in database (found 6, need 10+ chunks)\n"
                "  2. Document content is too short or low quality\n"
                "  3. Documents lack sufficient semantic diversity\n"
                "\n"
                "Solution: Run the main pipeline first to load Theory of Mind PDF documents:\n"
                "  → python src/langchain_eval_foundations_e2e.py"
            )

        # Extract data from Sample objects
        testset_df = pd.DataFrame([
            {
                'user_input': sample.user_input,
                'reference': sample.reference,
                'reference_contexts': sample.reference_contexts,
                'synthesizer_name': sample.synthesizer_name if hasattr(sample, 'synthesizer_name') else 'unknown'
            }
            for sample in samples
        ])
    else:
        raise ValueError(f"Unknown testset format. Type: {type(golden_testset)}")

    # Debug: Print available columns to understand RAGAS schema
    print(f"📋 Available columns in RAGAS testset: {list(testset_df.columns)}")
    print(f"📊 Number of rows: {len(testset_df)}")
    if len(testset_df) > 0:
        print(f"📊 First row sample:\n{testset_df.head(1)}")

    if len(testset_df) == 0:
        raise ValueError("Generated testset is empty. Check RAGAS generation logs for errors.")

    # Map RAGAS columns to Phoenix expected format
    question_col = 'user_input' if 'user_input' in testset_df.columns else 'question'
    answer_col = 'reference' if 'reference' in testset_df.columns else 'reference_answer'
    contexts_col = 'reference_contexts' if 'reference_contexts' in testset_df.columns else 'contexts'

    phoenix_df = pd.DataFrame(
        {
            "input": testset_df[question_col],
            "output": testset_df[answer_col],
            "contexts": testset_df[contexts_col].apply(
                lambda x: str(x) if isinstance(x, list) else str(x)
            ),
            "synthesizer": testset_df.get("synthesizer_name", "unknown"),
            "question_type": testset_df.get("synthesizer_name", "unknown"),
            "dataset_source": "ragas_golden_testset",
        }
    )

    # Use fixed dataset name for experiments script compatibility
    px_dataset_name = dataset_name

    # Use Phoenix client API with upload_dataset method
    import phoenix as px
    phoenix_client = px.Client()
    dataset = phoenix_client.upload_dataset(
        dataframe=phoenix_df,
        dataset_name=px_dataset_name,
        input_keys=["input"],
        output_keys=["output"],
        metadata_keys=["contexts", "synthesizer", "question_type", "dataset_source"]
    )

    return {
        "dataset_name": dataset_name,
        "num_samples": len(phoenix_df),
        "status": "success",
        "dataset": dataset,
    }

def main():

    # Setup configuration using the centralized config system
    config = setup_environment()

    llm = ChatOpenAI(model=config.model_name)
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    all_review_docs = load_docs_from_postgres(config.table_baseline)
    print(f"📊 Loaded {len(all_review_docs)} documents from database")

    # Limit documents for RAGAS processing (too many documents cause timeouts)
    max_docs_for_ragas = 50  # RAGAS works best with 20-50 documents
    original_doc_count = len(all_review_docs)
    if original_doc_count > max_docs_for_ragas:
        import random
        random.seed(42)  # Reproducible sampling
        all_review_docs = random.sample(all_review_docs, max_docs_for_ragas)
        print(f"📉 Sampled {max_docs_for_ragas} documents from {original_doc_count} total (RAGAS optimization)")

    # Validate we have enough documents (minimum 3 for RAGAS to work)
    min_docs_required = 3
    if len(all_review_docs) < min_docs_required:
        print(f"\n❌ ERROR: Only {len(all_review_docs)} document chunks available in database.")
        print(f"   Expected at least {min_docs_required} chunks from Theory of Mind research PDF files.")
        print(f"\n   The database needs to be populated first!")
        print(f"   Run the main pipeline to load documents:")
        print(f"   → python src/langchain_eval_foundations_e2e.py")
        print(f"\n   Or use the orchestration script:")
        print(f"   → python claude_code_scripts/run_rag_evaluation_pipeline.py\n")
        raise ValueError(f"Insufficient documents in database. Found {len(all_review_docs)}, need at least {min_docs_required} chunks.")

    # Warn if document count is low
    if len(all_review_docs) < 10:
        print(f"⚠️  WARNING: Only {len(all_review_docs)} documents available.")
        print(f"   RAGAS works best with 10+ diverse document chunks.")
        print(f"   Results may be limited. Consider adding more Theory of Mind PDFs to data/ directory.\n")

    # Use configurable testset size with environment variable override
    # Adjust testset_size based on available documents (RAGAS needs sufficient content)
    requested_size = int(os.getenv("GOLDEN_TESTSET_SIZE", config.golden_testset_size))
    testset_size = min(requested_size, max(1, len(all_review_docs) // 2))

    if testset_size != requested_size:
        print(f"⚠️  Adjusted testset size from {requested_size} to {testset_size} based on available documents")

    print(f"🧪 Generating golden test set with {testset_size} examples")
    
    golden_testset = generate_testset(
        all_review_docs, generator_llm, generator_embeddings, testset_size
    )

    # Debug: Check testset structure
    print(f"🔍 Testset type: {type(golden_testset)}")
    print(f"🔍 Testset attributes: {dir(golden_testset)}")
    if hasattr(golden_testset, 'samples'):
        print(f"🔍 Number of samples: {len(golden_testset.samples)}")
        if len(golden_testset.samples) > 0:
            print(f"🔍 Sample 0 type: {type(golden_testset.samples[0])}")
            print(f"🔍 Sample 0 attributes: {dir(golden_testset.samples[0])}")

    dataset_result = upload_to_phoenix(golden_testset, dataset_name="mixed_golden_testset")

    print(f"🚀 Workflow completed. Phoenix upload status: {dataset_result['status']}")

if __name__ == "__main__":
    main()
