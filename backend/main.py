"""FastAPI REST application for IntelliDoc AI Document Q&A Platform."""

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import (
    add_document,
    delete_document,
    get_all_documents,
    get_document_by_hash,
    get_document_by_id,
    init_db,
)
from backend.models import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
    UploadResponse,
)
from backend.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("intellidoc")

# Global singleton RAG pipeline
rag_pipeline: RAGPipeline = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to initialize DB and RAG vector store."""
    global rag_pipeline
    logger.info("Initializing IntelliDoc backend service...")

    # Ensure required directories exist
    settings.init_directories()

    # Initialize SQLite database schema
    init_db()

    # Initialize RAG Pipeline and load existing FAISS index if present
    rag_pipeline = RAGPipeline()
    logger.info("RAG Pipeline and database ready.")

    yield

    logger.info("Shutting down IntelliDoc backend service...")


app = FastAPI(
    title="IntelliDoc API",
    description="AI-Powered Document Q&A Platform REST API using RAG and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit frontend and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return system health status and component statistics."""
    docs = get_all_documents()
    vectors_count = (
        rag_pipeline.index.ntotal
        if (rag_pipeline and rag_pipeline.index is not None)
        else 0
    )
    is_gemini_set = bool(
        settings.GOOGLE_API_KEY
        and settings.GOOGLE_API_KEY.strip() != ""
        and settings.GOOGLE_API_KEY != "your_google_gemini_api_key_here"
    )

    return HealthResponse(
        status="healthy",
        total_documents=len(docs),
        total_vectors=vectors_count,
        embedding_model=settings.EMBEDDING_MODEL,
        gemini_configured=is_gemini_set,
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and index a PDF document.

    Computes MD5 hash to prevent duplicate uploads, chunks and embeds the document,
    stores vectors in FAISS, and saves metadata in SQLite.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF (.pdf) documents are supported.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Compute MD5 checksum for deduplication
    file_hash = hashlib.md5(content).hexdigest()

    existing_doc = get_document_by_hash(file_hash)
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Document with identical content has already been uploaded as "
                f"'{existing_doc['filename']}' on {existing_doc['upload_date']}."
            ),
        )

    # Persist PDF file to storage directory
    saved_filename = f"{file_hash}_{file.filename}"
    saved_path = settings.DATA_DIR / saved_filename
    with open(saved_path, "wb") as f:
        f.write(content)

    try:
        # Ingest into RAG pipeline
        num_chunks = rag_pipeline.ingest_document(
            filepath=saved_path,
            filename=file.filename,
            file_hash=file_hash,
        )

        # Store document record in SQLite
        doc_id = add_document(
            filename=file.filename,
            num_chunks=num_chunks,
            file_hash=file_hash,
        )

        record = get_document_by_id(doc_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve saved document record.",
            )

        doc_response = DocumentResponse(
            id=record["id"],
            filename=record["filename"],
            upload_date=str(record["upload_date"]),
            num_chunks=record["num_chunks"],
            file_hash=record["file_hash"],
        )

        return UploadResponse(
            message=f"Document '{file.filename}' successfully processed into {num_chunks} chunks.",
            document=doc_response,
        )

    except ValueError as e:
        # Clean up saved file if processing failed (e.g. empty or unreadable PDF)
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        logger.error(f"Unexpected error processing document '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while indexing document: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Ask a question against indexed documents using RAG.

    Retrieves top-k relevant chunks from FAISS, generates a grounded answer with
    Google Gemini, and returns citations with execution duration.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )

    start_time = time.time()
    try:
        answer, raw_sources = rag_pipeline.query(
            question=request.question.strip(),
            top_k=request.top_k,
        )
        duration = round(time.time() - start_time, 3)

        source_items = [
            SourceItem(
                content=src.get("text", ""),
                filename=src.get("filename", "Unknown"),
                page=src.get("page"),
                score=src.get("score"),
            )
            for src in raw_sources
        ]

        return QueryResponse(
            answer=answer,
            sources=source_items,
            processing_time=duration,
        )

    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        duration = round(time.time() - start_time, 3)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents() -> DocumentListResponse:
    """Retrieve all uploaded documents and their indexing metadata."""
    rows = get_all_documents()
    docs = [
        DocumentResponse(
            id=r["id"],
            filename=r["filename"],
            upload_date=str(r["upload_date"]),
            num_chunks=r["num_chunks"],
            file_hash=r["file_hash"],
        )
        for r in rows
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@app.get("/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(doc_id: int) -> DocumentResponse:
    """Retrieve metadata for a specific document by its ID."""
    row = get_document_by_id(doc_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found.",
        )
    return DocumentResponse(
        id=row["id"],
        filename=row["filename"],
        upload_date=str(row["upload_date"]),
        num_chunks=row["num_chunks"],
        file_hash=row["file_hash"],
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def remove_document(doc_id: int) -> DeleteResponse:
    """Delete a document, its vector index entries, and its stored PDF file."""
    record = get_document_by_id(doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found.",
        )

    file_hash = record["file_hash"]
    filename = record["filename"]

    # 1. Remove vector chunks from FAISS and rebuild index
    rag_pipeline.delete_document_chunks(file_hash)

    # 2. Delete physical PDF file from disk
    saved_filename = f"{file_hash}_{filename}"
    saved_path = settings.DATA_DIR / saved_filename
    if saved_path.exists():
        try:
            saved_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete stored file {saved_path}: {e}")

    # 3. Delete record from SQLite
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document from database.",
        )

    return DeleteResponse(
        message=f"Document '{filename}' and all associated vectors deleted successfully.",
        doc_id=doc_id,
    )

