import sys
from pathlib import Path

# Asegura que se pueda importar src/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api_main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()
