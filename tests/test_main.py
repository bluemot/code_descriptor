import io

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_upload_single_file(tmp_path):
    # Prepare a simple file upload
    file_content = b"print('hello')"
    files = [("files", ("test.py", io.BytesIO(file_content), "text/plain"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "success"


def test_ask_endpoint():
    response = client.post("/ask", json={"question": "What is 2+2?"})
    assert response.status_code == 200
    data = response.json()
    # Even if answer is empty, key should be present
    assert "answer" in data
