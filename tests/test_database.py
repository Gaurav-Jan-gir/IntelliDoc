"""Unit tests for SQLite document metadata database operations.

Compatible with both pytest and Python standard unittest.
"""

import sqlite3
import tempfile
import sys
from pathlib import Path
from contextlib import contextmanager

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import (
    add_document,
    delete_document,
    get_all_documents,
    get_document_by_hash,
    get_document_by_id,
    init_db,
)

# Optional pytest support
try:
    import pytest
except ImportError:
    class DummyPytest:
        @staticmethod
        @contextmanager
        def raises(expected_exception):
            try:
                yield
            except expected_exception:
                pass
            except Exception as e:
                raise AssertionError(f"Expected {expected_exception}, got {type(e)}")
            else:
                raise AssertionError(f"Expected {expected_exception} but nothing was raised")
    pytest = DummyPytest()  # type: ignore


class TestDatabaseOperations:
    """Test suite covering CRUD operations and constraints for documents."""

    def test_init_db_creates_table(self, temp_db=None):
        """init_db should create documents table without error."""
        if temp_db is None:
            with tempfile.NamedTemporaryFile(suffix=".db") as tf:
                init_db(tf.name)
                docs = get_all_documents(db_path=tf.name)
                assert docs == []
                return

        docs = get_all_documents(db_path=temp_db)
        assert docs == []

    def test_add_and_retrieve_document(self, temp_db=None):
        """add_document should insert a row and return its auto-increment ID."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tf:
            path = temp_db or tf.name
            init_db(path)
            doc_id = add_document(
                filename="sample_report.pdf",
                num_chunks=15,
                file_hash="hash123456",
                db_path=path,
            )
            assert isinstance(doc_id, int)
            assert doc_id > 0

            record = get_document_by_id(doc_id, db_path=path)
            assert record is not None
            assert record["filename"] == "sample_report.pdf"
            assert record["num_chunks"] == 15
            assert record["file_hash"] == "hash123456"
            assert "upload_date" in record

    def test_get_document_by_hash(self, temp_db=None):
        """get_document_by_hash should locate an existing document by its hash."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tf:
            path = temp_db or tf.name
            init_db(path)
            add_document(
                filename="ai_paper.pdf",
                num_chunks=42,
                file_hash="unique_hash_999",
                db_path=path,
            )

            found = get_document_by_hash("unique_hash_999", db_path=path)
            assert found is not None
            assert found["filename"] == "ai_paper.pdf"
            assert found["num_chunks"] == 42

            not_found = get_document_by_hash("nonexistent_hash", db_path=path)
            assert not_found is None

    def test_unique_file_hash_constraint(self, temp_db=None):
        """Inserting duplicate file_hash should raise sqlite3.IntegrityError."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tf:
            path = temp_db or tf.name
            init_db(path)
            add_document(
                filename="first.pdf",
                num_chunks=5,
                file_hash="duplicate_hash",
                db_path=path,
            )

            with pytest.raises(sqlite3.IntegrityError):
                add_document(
                    filename="second.pdf",
                    num_chunks=8,
                    file_hash="duplicate_hash",
                    db_path=path,
                )

    def test_get_all_documents_ordering(self, temp_db=None):
        """get_all_documents should return all rows."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tf:
            path = temp_db or tf.name
            init_db(path)
            add_document("doc1.pdf", 10, "hash_1", db_path=path)
            add_document("doc2.pdf", 20, "hash_2", db_path=path)
            add_document("doc3.pdf", 30, "hash_3", db_path=path)

            all_docs = get_all_documents(db_path=path)
            assert len(all_docs) == 3
            filenames = [d["filename"] for d in all_docs]
            assert "doc1.pdf" in filenames
            assert "doc2.pdf" in filenames
            assert "doc3.pdf" in filenames

    def test_delete_document(self, temp_db=None):
        """delete_document should remove row and return True, or False if missing."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tf:
            path = temp_db or tf.name
            init_db(path)
            doc_id = add_document("to_delete.pdf", 7, "hash_del", db_path=path)

            assert delete_document(doc_id, db_path=path) is True
            assert get_document_by_id(doc_id, db_path=path) is None
            assert delete_document(doc_id, db_path=path) is False
            assert delete_document(9999, db_path=path) is False


if __name__ == "__main__":
    test = TestDatabaseOperations()
    test.test_init_db_creates_table()
    test.test_add_and_retrieve_document()
    test.test_get_document_by_hash()
    test.test_unique_file_hash_constraint()
    test.test_get_all_documents_ordering()
    test.test_delete_document()
    print("All database tests passed successfully!")
