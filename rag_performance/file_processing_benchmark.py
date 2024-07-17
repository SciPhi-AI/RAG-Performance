import argparse
import os
import time
from pathlib import Path


class RAGBenchmark:
    def __init__(self):
        self.file_paths = [
            "rag_performance/data/shakespearecompleteworks.txt",
            "rag_performance/data/churchillcompleteworks.txt",
            "rag_performance/data/UniversityPhysicsVolume3.pdf",
            "rag_performance/data/IntroductoryStatistics.pdf",
        ]
        self.combinations = [
            (
                "rag_performance/data/shakespearecompleteworks.txt",
                "rag_performance/data/churchillcompleteworks.txt",
            ),
            (
                "rag_performance/data/UniversityPhysicsVolume3.pdf",
                "rag_performance/data/IntroductoryStatistics.pdf",
            ),
        ]

    def run_r2r(self, files: list[str]):
        from r2r import R2RClient

        client = R2RClient(base_url="http://localhost:8000")
        start_time = time.time()
        client.ingest_files(files)
        indexing_time = time.time() - start_time

        return indexing_time, len(files)

    def r2r_cleanup(self):
        """
        R2R has file deduplication. This method deletes the files that were ingested in the benchmark.
        """
        from r2r import R2RClient

        client = R2RClient(base_url="http://localhost:8000")

        document_ids = [
            "bee7d4bd-d5da-5c3b-9419-c7971aecf427",
            "8a50e508-f16d-5fc3-9d22-d5205c1a75a3",
            "b1310e7c-e307-575d-b405-1a8136d9fb82",
            "2301829a-b7ef-571a-8d56-3a287eb82ceb",
        ]

        for doc_id in document_ids:
            try:
                client.delete(["document_id"], [doc_id])
            except Exception as e:
                continue

    def run_haystack(self, files: list[str]):
        # Import Haystack modules
        from haystack import Pipeline
        from haystack.components.converters import PyPDFToDocument, TextFileToDocument
        from haystack.components.embedders import OpenAIDocumentEmbedder
        from haystack.components.joiners import DocumentJoiner
        from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
        from haystack.components.routers import FileTypeRouter
        from haystack.components.writers import DocumentWriter
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "router", FileTypeRouter(mime_types=["application/pdf", "text/plain"])
        )
        indexing_pipeline.add_component("pdf_converter", PyPDFToDocument())
        indexing_pipeline.add_component("text_converter", TextFileToDocument())
        indexing_pipeline.add_component("joiner", DocumentJoiner())
        indexing_pipeline.add_component("cleaner", DocumentCleaner())
        indexing_pipeline.add_component(
            "splitter",
            DocumentSplitter(split_by="word", split_length=500, split_overlap=50),
        )
        indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
        indexing_pipeline.add_component(
            "writer", DocumentWriter(document_store=document_store)
        )

        indexing_pipeline.connect("router.application/pdf", "pdf_converter")
        indexing_pipeline.connect("router.text/plain", "text_converter")
        indexing_pipeline.connect("pdf_converter", "joiner")
        indexing_pipeline.connect("text_converter", "joiner")
        indexing_pipeline.connect("joiner", "cleaner")
        indexing_pipeline.connect("cleaner", "splitter")
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")

        start_time = time.time()
        result = indexing_pipeline.run(
            {"router": {"sources": [Path(f) for f in files]}}
        )
        indexing_time = time.time() - start_time

        return indexing_time, document_store.count_documents()

    def run_llamaindex(self, files: list[str]):
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

        start_time = time.time()
        documents = SimpleDirectoryReader(input_files=files).load_data()
        index = VectorStoreIndex.from_documents(documents)
        indexing_time = time.time() - start_time

        return indexing_time, len(documents)

    def run_langchain(self, files: list[str]):
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

        embeddings = OpenAIEmbeddings()
        vector_store = Chroma(embedding_function=embeddings)

        def process_file(file_path):
            file_path = Path(file_path)
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents = loader.load()
            return text_splitter.split_documents(documents)

        def add_documents_in_batches(vector_store, documents, batch_size=500):
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                vector_store.add_documents(batch)

        start_time = time.time()

        for file_path in files:
            documents = process_file(file_path)
            add_documents_in_batches(vector_store, documents)

        indexing_time = time.time() - start_time

        return indexing_time, vector_store._collection.count()

    def run_ragflow(self, files: list[str]):
        import requests

        BASE_URL = os.getenv("RAGFLOW_BASE_URL")
        API_KEY = os.getenv("RAGFLOW_API_KEY")
        KB_NAME = os.getenv("RAGFLOW_KB_NAME")

        headers = {"Authorization": f"Bearer {API_KEY}"}

        url = f"{BASE_URL}/api/document/upload"

        with open(files, "rb") as file:
            files = {"file": file}
            data = {"kb_name": KB_NAME, "run": "1"}

            requests.post(url, headers=headers, files=files, data=data)

        # RagFlow does not provide a way to check the ingestion status, you have to check in their UI.
        return 0, 0

    def run_benchmark(self, provider: str):
        print(f"Running benchmark for {provider}")

        run_method = getattr(self, f"run_{provider}")

        # Process individual files
        for file in self.file_paths:
            if provider == "r2r":
                self.r2r_cleanup()
            time_taken, doc_count = run_method([file])
            print(
                f"{file}: Time taken: {time_taken:.2f} seconds, Documents: {doc_count}"
            )

        # Process combinations
        for combo in self.combinations:
            if provider == "r2r":
                self.r2r_cleanup()
            time_taken, doc_count = run_method(list(combo))
            print(
                f"{' + '.join(combo)}: Time taken: {time_taken:.2f} seconds, Documents: {doc_count}"
            )


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmarks")
    parser.add_argument(
        "--provider",
        choices=["r2r", "haystack", "llamaindex", "langchain", "ragflow"],
        required=True,
        help="RAG provider to benchmark",
    )
    args = parser.parse_args()

    benchmark = RAGBenchmark()
    benchmark.run_benchmark(args.provider)


if __name__ == "__main__":
    main()
