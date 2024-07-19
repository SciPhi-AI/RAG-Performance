"""
Benchmarking script for bulk ingestion. Seeks to push the limits by firehosing
approximately ten million tokens of data in batches of 256 files, simulating a
high-volume data environment.

Example usage:
    python rag_performance/scalable_ingestion_benchmark.py --provider r2r --output r2r_results.csv --batch-size 256
"""

import argparse
import asyncio
import csv
import os
import tempfile
import time

# Some fsspec dependency issues with poetry.
# You should install this on your own using pip install datasets
from datasets import load_dataset
from haystack import Document as HaystackDocument
from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from r2r.main.execution import R2RExecutionWrapper


class ScalableIngestionBenchmark:
    def __init__(
        self, dataset_name="wikipedia", dataset_config="20220301.en", split="train"
    ):
        self.dataset = load_dataset(
            dataset_name, dataset_config, split=split, streaming=True
        )
        self.token_limit = 10000000  # Default token limit

    def estimate_tokens(self, text):
        return len(text) // 4  # Rough estimate: 1 token ~= 4 characters

    def run_benchmark(self, provider, output_file, batch_size=256):
        method = getattr(self, f"run_{provider}")
        method(output_file, batch_size)

    def run_r2r(self, output_file, batch_size):
        total_tokens = 0
        start_time = time.time()
        r2r_client = R2RExecutionWrapper(
            client_mode=True, base_url="http://localhost:8000"
        )

        with open(output_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["Article_Number", "Total_Tokens", "Elapsed_Time", "Success"]
            )

            batch_file_paths = []

            for i, article in enumerate(self.dataset):
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".txt"
                ) as temp_file:
                    temp_file.write(article["text"])
                    temp_file_path = temp_file.name

                batch_file_paths.append(temp_file_path)
                tokens = self.estimate_tokens(article["text"])
                total_tokens += tokens

                if len(batch_file_paths) == batch_size:
                    success, elapsed_time = self.process_r2r_batch(
                        r2r_client, batch_file_paths, start_time
                    )
                    self.write_batch_results(
                        csv_writer, i, batch_size, total_tokens, elapsed_time, success
                    )
                    self.cleanup_temp_files(batch_file_paths)
                    batch_file_paths = []

                if self.should_stop(i, total_tokens, success):
                    break

            if batch_file_paths:
                success, elapsed_time = self.process_r2r_batch(
                    r2r_client, batch_file_paths, start_time
                )
                self.write_batch_results(
                    csv_writer,
                    i,
                    len(batch_file_paths),
                    total_tokens,
                    elapsed_time,
                    success,
                )
                self.cleanup_temp_files(batch_file_paths)

        print(f"Benchmark complete. Results saved to {output_file}")

    def run_llamaindex(self, output_file, batch_size):
        total_tokens = 0
        start_time = time.time()
        batch = []

        with open(output_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["Article_Number", "Total_Tokens", "Elapsed_Time", "Success"]
            )

            for i, article in enumerate(self.dataset):
                doc = Document(
                    text=article["text"], metadata={"title": article["title"]}
                )
                batch.append(doc)
                tokens = self.estimate_tokens(article["text"])
                total_tokens += tokens

                if len(batch) == batch_size:
                    success, elapsed_time = self.process_llamaindex_batch(
                        batch, start_time
                    )
                    self.write_batch_results(
                        csv_writer, i, batch_size, total_tokens, elapsed_time, success
                    )
                    batch = []

                if self.should_stop(i, total_tokens, success):
                    break

            if batch:
                success, elapsed_time = self.process_llamaindex_batch(batch, start_time)
                self.write_batch_results(
                    csv_writer, i, len(batch), total_tokens, elapsed_time, success
                )

        print(f"Benchmark complete. Results saved to {output_file}")

    def run_llamaindex_async(self, output_file, batch_size):
        async def _run():
            total_tokens = 0
            start_time = time.time()
            batch = []

            # Build pipeline
            transformations = [
                SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                OpenAIEmbedding(num_workers=8),
            ]
            pipeline = IngestionPipeline(transformations=transformations)

            with open(output_file, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(
                    ["Article_Number", "Total_Tokens", "Elapsed_Time", "Success"]
                )

                for i, article in enumerate(self.dataset):
                    doc = Document(
                        text=article["text"], metadata={"title": article["title"]}
                    )
                    batch.append(doc)
                    tokens = self.estimate_tokens(article["text"])
                    total_tokens += tokens

                    if len(batch) == batch_size:
                        try:
                            nodes = await pipeline.arun(documents=batch)
                            VectorStoreIndex(nodes)
                            success = True
                        except Exception as e:
                            print(f"Ingestion failed: {e}")
                            success = False
                        elapsed_time = time.time() - start_time

                        self.write_batch_results(
                            csv_writer,
                            i,
                            batch_size,
                            total_tokens,
                            elapsed_time,
                            success,
                        )
                        batch = []

                    if self.should_stop(i, total_tokens, success):
                        break

                if batch:
                    try:
                        nodes = await pipeline.arun(documents=batch)
                        VectorStoreIndex(nodes)
                        success = True
                    except Exception as e:
                        print(f"Ingestion failed: {e}")
                        success = False
                    elapsed_time = time.time() - start_time

                    self.write_batch_results(
                        csv_writer, i, len(batch), total_tokens, elapsed_time, success
                    )

            print(f"Benchmark complete. Results saved to {output_file}")

        asyncio.run(_run())

    def run_haystack(self, output_file, batch_size):
        total_tokens = 0
        start_time = time.time()
        batch = []

        # Create Haystack pipeline
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "splitter",
            DocumentSplitter(split_by="word", split_length=500, split_overlap=50),
        )
        indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
        indexing_pipeline.add_component(
            "writer", DocumentWriter(document_store=document_store)
        )
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")

        with open(output_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Article_Number",
                    "Total_Tokens",
                    "Elapsed_Time",
                    "Success",
                    "Documents_in_Store",
                ]
            )

            for i, article in enumerate(self.dataset):
                doc = HaystackDocument(
                    content=article["text"], meta={"title": article["title"]}
                )
                batch.append(doc)
                tokens = self.estimate_tokens(article["text"])
                total_tokens += tokens

                if len(batch) == batch_size:
                    success, elapsed_time = self.process_haystack_batch(
                        indexing_pipeline, batch, start_time
                    )
                    self.write_batch_results(
                        csv_writer,
                        i,
                        batch_size,
                        total_tokens,
                        elapsed_time,
                        success,
                        document_store.count_documents(),
                    )
                    batch = []

                if self.should_stop(i, total_tokens, success):
                    break

            if batch:
                success, elapsed_time = self.process_haystack_batch(
                    indexing_pipeline, batch, start_time
                )
                self.write_batch_results(
                    csv_writer,
                    i,
                    len(batch),
                    total_tokens,
                    elapsed_time,
                    success,
                    document_store.count_documents(),
                )

        print(f"Benchmark complete. Results saved to {output_file}")

    def run_langchain(self, output_file, batch_size):
        total_tokens = 0
        start_time = time.time()
        batch = []

        # Set up document splitter and embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        embeddings = OpenAIEmbeddings()

        # Set up Chroma vector store
        vector_store = Chroma(embedding_function=embeddings)

        with open(output_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Article_Number",
                    "Total_Tokens",
                    "Elapsed_Time",
                    "Success",
                    "Documents_in_Store",
                ]
            )

            for i, article in enumerate(self.dataset):
                doc = LangchainDocument(
                    page_content=article["text"], metadata={"title": article["title"]}
                )
                batch.append(doc)
                tokens = self.estimate_tokens(article["text"])
                total_tokens += tokens

                if len(batch) == batch_size:
                    success, elapsed_time = self.process_langchain_batch(
                        vector_store, text_splitter, batch, start_time
                    )
                    self.write_batch_results(
                        csv_writer,
                        i,
                        batch_size,
                        total_tokens,
                        elapsed_time,
                        success,
                        vector_store._collection.count(),
                    )
                    batch = []

                if self.should_stop(i, total_tokens, success):
                    break

            if batch:
                success, elapsed_time = self.process_langchain_batch(
                    vector_store, text_splitter, batch, start_time
                )
                self.write_batch_results(
                    csv_writer,
                    i,
                    len(batch),
                    total_tokens,
                    elapsed_time,
                    success,
                    vector_store._collection.count(),
                )

        print(f"Benchmark complete. Results saved to {output_file}")

    def process_r2r_batch(self, r2r_client, batch_file_paths, start_time):
        try:
            r2r_client.ingest_files(file_paths=batch_file_paths)
            success = True
        except Exception as e:
            print(f"Ingestion failed: {e}")
            success = False
        elapsed_time = time.time() - start_time
        return success, elapsed_time

    def process_llamaindex_batch(self, batch, start_time):
        try:
            VectorStoreIndex.from_documents(batch)
            success = True
        except Exception as e:
            print(f"Ingestion failed: {e}")
            success = False
        elapsed_time = time.time() - start_time
        return success, elapsed_time

    def process_haystack_batch(self, pipeline, batch, start_time):
        try:
            pipeline.run({"splitter": {"documents": batch}})
            success = True
        except Exception as e:
            print(f"Ingestion failed: {e}")
            success = False
        elapsed_time = time.time() - start_time
        return success, elapsed_time

    def process_langchain_batch(self, vector_store, text_splitter, batch, start_time):
        try:
            split_docs = text_splitter.split_documents(batch)
            vector_store.add_documents(split_docs)
            success = True
        except Exception as e:
            print(f"Ingestion failed: {e}")
            success = False
        elapsed_time = time.time() - start_time
        return success, elapsed_time

    def write_batch_results(
        self,
        csv_writer,
        i,
        batch_size,
        total_tokens,
        elapsed_time,
        success,
        documents_in_store=None,
    ):
        for j in range(batch_size):
            row = [i - batch_size + j + 2, total_tokens, elapsed_time, success]
            if documents_in_store is not None:
                row.append(documents_in_store)
            csv_writer.writerow(row)

    def cleanup_temp_files(self, file_paths):
        for file_path in file_paths:
            os.unlink(file_path)

    def should_stop(self, i, total_tokens, success):
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} articles, Total tokens: {total_tokens}")

        if not success:
            print(f"Ingestion failed at article {i+1}. Stopping benchmark.")
            return True

        if total_tokens > self.token_limit:
            print("Reached token limit. Stopping benchmark.")
            return True

        return False


def main():
    parser = argparse.ArgumentParser(description="Run Scalable Ingestion Benchmark")
    parser.add_argument(
        "--provider",
        choices=["r2r", "llamaindex", "llamaindex_async", "haystack", "langchain"],
        required=True,
        help="Provider to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file name for benchmark results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for ingestion (default: 256)",
    )
    args = parser.parse_args()

    benchmark = ScalableIngestionBenchmark()
    benchmark.run_benchmark(args.provider, args.output, args.batch_size)


if __name__ == "__main__":
    main()
