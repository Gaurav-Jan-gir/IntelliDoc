"""Database layer for document metadata management using SQLite.

Provides thread-safe connections, schema initialization, and CRUD operations.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from backend.config import settings


@contextmanager
def get_db_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite database connection with row factory.

    Args:
        db_path: Optional path to SQLite database. Defaults to configured DB_PATH.

    Yields:
        sqlite3.Connection: Active SQLite connection with dict-like row access.
    """
    path = Path(db_path) if db_path else settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize SQLite database schema for document metadata.

    Creates the `documents` table if it does not already exist.

    Args:
        db_path: Optional custom path to SQLite database.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_chunks INTEGER NOT NULL,
                file_hash TEXT UNIQUE NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_file_hash
            ON documents(file_hash);
            """
        )


def add_document(
    filename: str,
    num_chunks: int,
    file_hash: str,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert a newly processed document record.

    Args:
        filename: Original name of the uploaded document.
        num_chunks: Number of chunked text segments generated.
        file_hash: MD5 hash of document contents for deduplication.
        db_path: Optional custom path to SQLite database.

    Returns:
        int: The database ID of the newly inserted document record.

    Raises:
        sqlite3.IntegrityError: If a document with the same file_hash already exists.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (filename, num_chunks, file_hash)
            VALUES (?, ?, ?);
            """,
            (filename, num_chunks, file_hash),
        )
        return int(cursor.lastrowid)


def get_all_documents(
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all document records ordered by upload date descending.

    Args:
        db_path: Optional custom path to SQLite database.

    Returns:
        List[Dict[str, Any]]: List of document record dictionaries.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, upload_date, num_chunks, file_hash
            FROM documents
            ORDER BY upload_date DESC;
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_document_by_hash(
    file_hash: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a document by its content MD5 hash.

    Args:
        file_hash: MD5 hash string to search for.
        db_path: Optional custom path to SQLite database.

    Returns:
        Optional[Dict[str, Any]]: Document record dictionary if found, else None.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, upload_date, num_chunks, file_hash
            FROM documents
            WHERE file_hash = ?;
            """,
            (file_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_document_by_id(
    doc_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a document record by its primary key ID.

    Args:
        doc_id: Primary key document identifier.
        db_path: Optional custom path to SQLite database.

    Returns:
        Optional[Dict[str, Any]]: Document record dictionary if found, else None.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, upload_date, num_chunks, file_hash
            FROM documents
            WHERE id = ?;
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_document(
    doc_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete a document record by its primary key ID.

    Args:
        doc_id: Primary key document identifier to delete.
        db_path: Optional custom path to SQLite database.

    Returns:
        bool: True if a record was deleted, False if not found.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM documents
            WHERE id = ?;
            """,
            (doc_id,),
        )
        return cursor.rowcount > 0

