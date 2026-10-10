"""Unit tests for document chunking and text splitting functionality.

Compatible with both pytest and Python standard unittest.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag_pipeline import RecursiveCharacterTextSplitter, RAGPipeline


class TestTextChunking:
    """Test suite for RecursiveCharacterTextSplitter and chunk generation."""

    def test_empty_text_returns_empty_list(self):
        """Splitting empty or whitespace-only text should return an empty list."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        assert splitter.split_text("") == []
        assert splitter.split_text("   \n\n   ") == []

    def test_text_shorter_than_chunk_size(self):
        """Text shorter than chunk_size should return as a single chunk."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        sample = "IntelliDoc is an AI document assistant."
        chunks = splitter.split_text(sample)
        assert len(chunks) == 1
        assert chunks[0] == sample

    def test_chunk_size_and_overlap(self):
        """Long text should be divided into chunks respecting size boundaries."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15)
        sample = (
            "Natural Language Processing powers retrieval augmented generation. "
            "FAISS provides efficient vector similarity search for dense vectors. "
            "Gemini generates fluent and grounded responses based on context."
        )
        chunks = splitter.split_text(sample)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0
            assert len(chunk) <= 80

    def test_paragraphs_split_cleanly(self):
        """Splitting on paragraphs should honor paragraph delimiters."""
        p1 = "Paragraph 1 discusses vector embeddings and indexing."
        p2 = "Paragraph 2 discusses prompt engineering and context grounding."
        text = f"{p1}\n\n{p2}"
        splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert any("Paragraph 1" in c for c in chunks)
        assert any("Paragraph 2" in c for c in chunks)

    def test_chunk_document_metadata_preservation(self, tmp_path=None):
        """Verify chunk_document attaches page numbers, file name, and file hash."""
        pipeline = object.__new__(RAGPipeline)
        pipeline.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=80, chunk_overlap=10
        )

        pages_content = [
            ("Page one introduces the overall architecture and setup.", 1),
            ("Page two details the FAISS vector database and indexing pipeline.", 2),
        ]

        chunks = pipeline.chunk_document(
            pages_content=pages_content,
            filename="test_guide.pdf",
            file_hash="abc123hash",
        )

        assert len(chunks) == 2
        assert chunks[0]["page"] == 1
        assert chunks[0]["filename"] == "test_guide.pdf"
        assert chunks[0]["file_hash"] == "abc123hash"
        assert "architecture" in chunks[0]["text"]

        assert chunks[1]["page"] == 2
        assert chunks[1]["filename"] == "test_guide.pdf"
        assert chunks[1]["file_hash"] == "abc123hash"
        assert "FAISS" in chunks[1]["text"]


if __name__ == "__main__":
    test = TestTextChunking()
    test.test_empty_text_returns_empty_list()
    test.test_text_shorter_than_chunk_size()
    test.test_chunk_size_and_overlap()
    test.test_paragraphs_split_cleanly()
    test.test_chunk_document_metadata_preservation()
    print("All chunking tests passed successfully!")
