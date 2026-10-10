"""Pydantic schemas and models for IntelliDoc request/response payloads."""

from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Schema for single document metadata response."""

    id: int = Field(..., description="Unique document ID in database")
    filename: str = Field(..., description="Original name of the uploaded file")
    upload_date: Union[datetime, str] = Field(..., description="Timestamp of document upload")
    num_chunks: int = Field(..., description="Number of text chunks indexed")
    file_hash: str = Field(..., description="MD5 hash of document contents")

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for listing all uploaded documents."""

    documents: List[DocumentResponse] = Field(default_factory=list, description="List of document records")
    total: int = Field(..., description="Total count of documents indexed")


class QueryRequest(BaseModel):
    """Schema for asking a question against indexed documents."""

    question: str = Field(..., min_length=1, description="User question in natural language")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override default number of chunks to retrieve")


class SourceItem(BaseModel):
    """Schema for an individual retrieved source citation."""

    content: str = Field(..., description="Text content of the retrieved chunk")
    filename: str = Field(..., description="Filename containing this chunk")
    page: Optional[int] = Field(None, description="Page number of the chunk in original PDF")
    score: Optional[float] = Field(None, description="Similarity distance score (lower is closer in L2)")


class QueryResponse(BaseModel):
    """Schema for answering user query with citations."""

    answer: str = Field(..., description="AI generated answer based on retrieved context")
    sources: List[SourceItem] = Field(default_factory=list, description="Source chunks cited in the answer")
    processing_time: float = Field(..., description="Time taken to process query in seconds")


class UploadResponse(BaseModel):
    """Schema for response after PDF upload and indexing."""

    message: str = Field(..., description="Status message of the upload operation")
    document: DocumentResponse = Field(..., description="Metadata of the newly indexed document")


class DeleteResponse(BaseModel):
    """Schema for response after deleting a document."""

    message: str = Field(..., description="Status message confirming deletion")
    doc_id: int = Field(..., description="ID of deleted document")


class HealthResponse(BaseModel):
    """Schema for service health check response."""

    status: str = Field(..., description="System operational status")
    total_documents: int = Field(..., description="Count of documents in SQLite database")
    total_vectors: int = Field(..., description="Count of vector chunks in FAISS index")
    embedding_model: str = Field(..., description="Active embedding model name")
    gemini_configured: bool = Field(..., description="Whether GOOGLE_API_KEY is configured")

