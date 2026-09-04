"""
Tests pour l'authentification JWT.
"""

import sys
sys.path.insert(0, r"d:\Téléchargements\projet_actuariat\backend")

import pytest

# Les dépendances d'auth (jose, passlib) ne sont installées qu'en production/CI.
# Si absentes, on saute élégamment les tests.
pytest.importorskip("jose")
pytest.importorskip("passlib")


class TestAuthEndpoints:
    """Tests pour /auth/register, /auth/login, /auth/me."""

    def test_register_user(self, client):
        response = client.post("/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "secret123",
            "full_name": "Test User",
            "role": "viewer",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "testuser"
        assert data["role"] == "viewer"
        assert "id" in data

    def test_register_duplicate_username(self, client):
        client.post("/auth/register", json={
            "username": "dup_user",
            "email": "dup1@test.com",
            "password": "secret123",
        })
        response = client.post("/auth/register", json={
            "username": "dup_user",
            "email": "dup2@test.com",
            "password": "secret123",
        })
        assert response.status_code == 400

    def test_login_success(self, client):
        client.post("/auth/register", json={
            "username": "logintest",
            "email": "login@test.com",
            "password": "mypass123",
        })
        response = client.post("/auth/login", json={
            "username": "logintest",
            "password": "mypass123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={
            "username": "wrongpw",
            "email": "wrongpw@test.com",
            "password": "correct",
        })
        response = client.post("/auth/login", json={
            "username": "wrongpw",
            "password": "incorrect",
        })
        assert response.status_code == 401

    def test_me_with_token(self, client):
        # Register
        client.post("/auth/register", json={
            "username": "meuser",
            "email": "me@test.com",
            "password": "me123456",
        })
        # Login
        login_resp = client.post("/auth/login", json={
            "username": "meuser",
            "password": "me123456",
        })
        token = login_resp.json()["access_token"]

        # Get /me
        response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        assert response.status_code == 200
        assert response.json()["username"] == "meuser"

    def test_me_without_token(self, client):
        response = client.get("/auth/me")
        assert response.status_code == 401
