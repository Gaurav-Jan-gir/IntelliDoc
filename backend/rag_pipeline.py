"""RAG (Retrieval-Augmented Generation) Pipeline for IntelliDoc.

Orchestrates PDF text extraction, recursive chunking, vector embedding,
FAISS vector search, and LLM-powered question answering with source citations.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings

logger = logging.getLogger(__name__)

# Fallback-aware import for RecursiveCharacterTextSplitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # High quality pure-Python fallback if langchain is not present
        class RecursiveCharacterTextSplitter:  # type: ignore
            """Pure Python recursive character text splitter fallback matching LangChain behavior."""

            def __init__(
                self,
                chunk_size: int = 500,
                chunk_overlap: int = 50,
                separators: Optional[List[str]] = None,
            ) -> None:
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

            def split_text(self, text: str) -> List[str]:
                if not text or not text.strip():
                    return []
                text = text.strip()
                if len(text) <= self.chunk_size:
                    return [text]

                def _split(s: str, seps: List[str]) -> List[str]:
                    if len(s) <= self.chunk_size or not seps:
                        return [s] if s.strip() else []
                    sep = seps[0]
                    splits = s.split(sep) if sep else list(s)
                    good_splits: List[str] = []
                    for part in splits:
                        part_clean = part.strip()
                        if not part_clean:
                            continue
                        if len(part_clean) > self.chunk_size and len(seps) > 1:
                            good_splits.extend(_split(part_clean, seps[1:]))
                        else:
                            good_splits.append(part_clean)
                    return good_splits

                splits = _split(text, self.separators)

                final_chunks: List[str] = []
                current_chunk: List[str] = []
                current_len = 0

                for s in splits:
                    s_len = len(s)
                    if current_len + s_len > self.chunk_size and current_chunk:
                        doc = " ".join(current_chunk).strip()
                        if doc:
                            final_chunks.append(doc)
                        while current_chunk and current_len > self.chunk_overlap:
                            popped = current_chunk.pop(0)
                            current_len -= len(popped) + 1
                    current_chunk.append(s)
                    current_len += s_len + 1

                if current_chunk:
                    doc = " ".join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)

                return final_chunks


class RAGPipeline:
    """Manages document chunking, embedding index, retrieval, and LLM generation."""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        vectorstore_dir: Optional[Path] = None,
        google_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            embedding_model_name: Name of the sentence-transformers model.
            vectorstore_dir: Directory to persist FAISS index and metadata.
            google_api_key: Google Gemini API key.
        """
        self.embedding_model_name = (
            embedding_model_name or settings.EMBEDDING_MODEL
        )
        self.vectorstore_dir = Path(vectorstore_dir or settings.VECTORSTORE_DIR)
        self.google_api_key = (
            google_api_key if google_api_key is not None else settings.GOOGLE_API_KEY
        )

        self.index_path = self.vectorstore_dir / "index.faiss"
        self.metadata_path = self.vectorstore_dir / "metadata.pkl"

        self._embedding_model: Any = None
        self._faiss_module: Any = None
        self._numpy_module: Any = None
        self.index: Any = None
        self.chunks_metadata: List[Dict[str, Any]] = []

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.load_index()

    @property
    def np(self) -> Any:
        """Lazy load numpy module."""
        if self._numpy_module is None:
            import numpy as np

            self._numpy_module = np
        return self._numpy_module

    @property
    def embedding_model(self) -> Any:
        """Lazy load SentenceTransformer embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def faiss(self) -> Any:
        """Lazy load FAISS module."""
        if self._faiss_module is None:
            import faiss

            self._faiss_module = faiss
        return self._faiss_module

    def load_index(self) -> bool:
        """Load persisted FAISS index and chunk metadata from disk.

        Returns:
            bool: True if index was successfully loaded, False otherwise.
        """
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = self.faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "rb") as f:
                    self.chunks_metadata = pickle.load(f)
                logger.info(
                    f"Loaded FAISS index with {self.index.ntotal} vectors and "
                    f"{len(self.chunks_metadata)} metadata entries."
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing FAISS index: {e}. Starting fresh.")
                self.index = None
                self.chunks_metadata = []
                return False
        return False

    def save_index(self) -> None:
        """Persist current FAISS index and chunk metadata to disk."""
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None and self.index.ntotal > 0:
            self.faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.chunks_metadata, f)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors.")
        else:
            # If empty, clean up stale index files
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()

    def extract_text_from_pdf(
        self, filepath: Path | str
    ) -> List[Tuple[str, int]]:
        """Extract text from a PDF file with page number tracking.

        Args:
            filepath: Path to the target PDF file.

        Returns:
            List[Tuple[str, int]]: List of (page_text, page_number) tuples.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the PDF contains no extractable text.
        """
        import pymupdf4llm

        pdf_path = Path(filepath)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            md_docs = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            pages_content: List[Tuple[str, int]] = []
            
            for doc in md_docs:
                page_idx = doc.get("metadata", {}).get("page", 0) + 1  # 1-indexed
                text = doc.get("text", "").strip()
                if text:
                    pages_content.append((text, page_idx))
                    
        except Exception as e:
            logger.warning(f"Error extracting text with pymupdf4llm: {e}")
            pages_content = []

        if not pages_content:
            raise ValueError(
                f"The PDF '{pdf_path.name}' contains no extractable text or is empty."
            )

        return pages_content

    def chunk_document(
        self,
        pages_content: List[Tuple[str, int]],
        filename: str,
        file_hash: str,
    ) -> List[Dict[str, Any]]:
        """Split page texts into uniform overlapping chunks with full provenance metadata.

        Args:
            pages_content: List of (page_text, page_number) extracted from PDF.
            filename: Source PDF file name.
            file_hash: MD5 hash of source file.

        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries containing text and metadata.
        """
        all_chunks: List[Dict[str, Any]] = []

        for text, page_num in pages_content:
            page_chunks = self.text_splitter.split_text(text)
            for chunk_str in page_chunks:
                clean_chunk = chunk_str.strip()
                if clean_chunk:
                    all_chunks.append(
                        {
                            "text": clean_chunk,
                            "filename": filename,
                            "page": page_num,
                            "file_hash": file_hash,
                        }
                    )

        return all_chunks

    def ingest_document(
        self,
        filepath: Path | str,
        filename: str,
        file_hash: str,
    ) -> int:
        """Extract, chunk, embed, and index a PDF document into FAISS.

        Args:
            filepath: Path to the stored PDF file.
            filename: Original PDF filename.
            file_hash: MD5 hash of the PDF file.

        Returns:
            int: Number of chunks indexed.
        """
        pages_content = self.extract_text_from_pdf(filepath)
        chunks = self.chunk_document(pages_content, filename, file_hash)

        if not chunks:
            raise ValueError(f"No chunks could be generated for '{filename}'.")

        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        embeddings_np = self.np.array(embeddings, dtype=self.np.float32)
        dim = embeddings_np.shape[1]

        if self.index is None:
            self.index = self.faiss.IndexFlatL2(dim)
        elif self.index.d != dim:
            # Re-initialize index if dimension mismatch occurs
            self.index = self.faiss.IndexFlatL2(dim)
            self.chunks_metadata = []

        self.index.add(embeddings_np)
        self.chunks_metadata.extend(chunks)

        self.save_index()
        return len(chunks)

    def retrieve(
        self, question: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant document chunks for a question.

        Args:
            question: User query string.
            top_k: Number of chunks to retrieve. Defaults to settings.TOP_K.

        Returns:
            List[Dict[str, Any]]: Retrieved chunk dictionaries with similarity scores.
        """
        k = top_k or settings.TOP_K
        if self.index is None or self.index.ntotal == 0 or not self.chunks_metadata:
            return []

        actual_k = min(k, self.index.ntotal)
        query_embedding = self.embedding_model.encode(
            [question],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        query_np = self.np.array(query_embedding, dtype=self.np.float32)

        distances, indices = self.index.search(query_np, actual_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks_metadata):
                chunk_data = dict(self.chunks_metadata[idx])
                chunk_data["score"] = float(round(float(dist), 4))
                results.append(chunk_data)

        return results

    def _build_prompt(self, context: str, question: str) -> str:
        """Build strict context-grounded prompt for the Gemini LLM.

        Args:
            context: Formatted context passages.
            question: User's question.

        Returns:
            str: Grounded LLM prompt.
        """
        return (
            "You are an expert document analysis assistant. Answer the user's question "
            "accurately and concisely using ONLY the provided context.\n"
            "If the answer cannot be found in the context, explicitly state: "
            '"I cannot find the answer in the provided document context."\n'
            "Do not make up information or use outside knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    def query(
        self, question: str, top_k: Optional[int] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """End-to-end RAG query: retrieve relevant chunks and generate answer via Gemini.

        Args:
            question: User query string.
            top_k: Optional override for number of chunks to retrieve.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: (answer_text, list_of_retrieved_source_chunks).
        """
        if self.index is None or self.index.ntotal == 0 or not self.chunks_metadata:
            return (
                "No documents have been indexed yet. Please upload a PDF document first.",
                [],
            )

        retrieved_chunks = self.retrieve(question, top_k=top_k)
        if not retrieved_chunks:
            return (
                "I cannot find any relevant context in the uploaded documents to answer your question.",
                [],
            )

        # Assemble context string with document citations
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source_tag = f"[{chunk['filename']}, Page {chunk.get('page', 'Unknown')}]"
            context_parts.append(f"Passage {i} {source_tag}:\n{chunk['text']}")

        context_str = "\n\n".join(context_parts)
        prompt = self._build_prompt(context=context_str, question=question)

        # Handle Gemini API Key configuration
        api_key = self.google_api_key or settings.GOOGLE_API_KEY
        if not api_key or api_key.strip() == "" or api_key == "your_google_gemini_api_key_here":
            return (
                "Google Gemini API key is not configured. "
                "Please configure `GOOGLE_API_KEY` in your `.env` file to enable AI generation. "
                "The most relevant context passages have been retrieved and are listed in the sources below.",
                retrieved_chunks,
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-3.8-flash")
            response = model.generate_content(prompt)

            answer = response.text if response and response.text else (
                "I could not generate an answer from the provided context."
            )
            return answer.strip(), retrieved_chunks

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return (
                f"Error generating answer with Gemini: {str(e)}. "
                "The retrieved context passages are provided in the sources below.",
                retrieved_chunks,
            )

    def delete_document_chunks(self, file_hash: str) -> int:
        """Remove all indexed chunks belonging to a document and rebuild FAISS index.

        Args:
            file_hash: MD5 hash of document to delete.

        Returns:
            int: Count of removed chunks.
        """
        if not self.chunks_metadata:
            return 0

        remaining_metadata = [
            c for c in self.chunks_metadata if c.get("file_hash") != file_hash
        ]
        removed_count = len(self.chunks_metadata) - len(remaining_metadata)

        if removed_count == 0:
            return 0

        self.chunks_metadata = remaining_metadata

        if not self.chunks_metadata:
            self.index = None
            self.save_index()
            return removed_count

        # Rebuild FAISS index from remaining chunk texts
        chunk_texts = [c["text"] for c in self.chunks_metadata]
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings_np = self.np.array(embeddings, dtype=self.np.float32)
        dim = embeddings_np.shape[1]

        self.index = self.faiss.IndexFlatL2(dim)
        self.index.add(embeddings_np)
        self.save_index()

        return removed_count
