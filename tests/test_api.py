import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.database import init_db
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fixture providing a test client configured with isolated database and storage."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed in current environment")
    test_db = tmp_path / "test_api.db"
    test_data = tmp_path / "data"
    test_data.mkdir()
    test_vectorstore = tmp_path / "vectorstore"
    test_vectorstore.mkdir()

    init_db(db_path=test_db)

    # Monkeypatch settings
    from backend.config import settings
    monkeypatch.setattr(settings, "DB_PATH", test_db)
    monkeypatch.setattr(settings, "DATA_DIR", test_data)
    monkeypatch.setattr(settings, "VECTORSTORE_DIR", test_vectorstore)

    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """GET /health should return 200 with system statistics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_documents" in data
    assert "total_vectors" in data
    assert "embedding_model" in data


def test_list_documents_empty(client):
    """GET /documents should return empty list initially."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["documents"] == []


def test_query_empty_question_bad_request(client):
    """POST /query with empty question should return 400."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422 or response.status_code == 400


def test_upload_non_pdf_fails(client):
    """POST /upload with non-PDF file should return 400."""
    file_content = b"Not a pdf file"
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]


def test_upload_empty_pdf_fails(client):
    """POST /upload with 0-byte file should return 400."""
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400


def test_get_nonexistent_document(client):
    """GET /documents/{id} for missing ID should return 404."""
    response = client.get("/documents/99999")
    assert response.status_code == 404


def test_delete_nonexistent_document(client):
    """DELETE /documents/{id} for missing ID should return 404."""
    response = client.delete("/documents/99999")
    assert response.status_code == 404
