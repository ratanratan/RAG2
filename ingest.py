
import os
import glob
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# STEP 1: Load configuration
load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

embedding_model = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

COLLECTION_NAME = "company_hr_policies"


def main():

    # STEP 2: Find all PDF files
    pdf_files = glob.glob(
        os.path.join(DATA_DIR, "*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found inside the data folder."
        )

    print("PDF files found:", len(pdf_files))

    # STEP 3: Read all PDFs
    documents = []

    for pdf_path in pdf_files:

        print("Reading:", os.path.basename(pdf_path))

        loader = PyPDFLoader(pdf_path)
        pdf_documents = loader.load()

        # Keep the original filename in metadata
        for document in pdf_documents:
            document.metadata["source"] = os.path.basename(pdf_path)

        documents.extend(pdf_documents)

    print("Total pages:", len(documents))

    # STEP 4: Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print("Total chunks:", len(chunks))

    # STEP 5: Create the embedding model
    embeddings = OpenAIEmbeddings(
        model=embedding_model
    )

    # STEP 6: Rebuild the local database cleanly
    # WARNING: This removes the previous local Chroma DB.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # STEP 7: Store chunks and embeddings in ChromaDB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print("\nIngestion completed successfully!")
    print("Stored chunks:", db._collection.count())
    print("Database location:", CHROMA_DIR)


if __name__ == "__main__":
    main()