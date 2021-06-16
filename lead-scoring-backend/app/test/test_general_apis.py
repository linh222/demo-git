import unittest

from fastapi.testclient import TestClient

from app.main import app


class TestGeneralApis(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_heathcheck(self):
        response = self.client.get("/health")
        assert response.status_code == 200
