def test_read_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_read_dashboard(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_register_user(client):
    response = client.post(
        "/api/auth/register",
        json={
            "email": "testuser@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_register_existing_user(client):
    # First registration
    client.post(
        "/api/auth/register",
        json={
            "email": "existinguser@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    # Second registration should fail
    response = client.post(
        "/api/auth/register",
        json={
            "email": "existinguser@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"

def test_login_success(client):
    client.post(
        "/api/auth/register",
        json={
            "email": "loginuser@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    response = client.post(
        "/api/auth/login",
        data={
            "username": "loginuser@example.com",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_failure(client):
    response = client.post(
        "/api/auth/login",
        data={
            "username": "nonexistent@example.com",
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401
