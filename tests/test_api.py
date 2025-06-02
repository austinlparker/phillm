from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from phillm.api.routes import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_check_redis_healthy():
    # Test business logic: health endpoint response when Redis is healthy
    with patch("phillm.api.routes.RedisVectorStore") as mock_vector_store:
        mock_instance = AsyncMock()
        mock_instance.health_check.return_value = True
        mock_vector_store.return_value = mock_instance

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "PhiLLM",
            "redis": "connected",
        }


def test_health_check_redis_unhealthy():
    # Test business logic: health endpoint response when Redis is unhealthy
    with patch("phillm.api.routes.RedisVectorStore") as mock_vector_store:
        mock_instance = AsyncMock()
        mock_instance.health_check.return_value = False
        mock_vector_store.return_value = mock_instance

        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "status": "unhealthy",
                "service": "PhiLLM",
                "redis": "disconnected",
            }
        }


@patch("phillm.api.routes.RedisVectorStore")
@patch("phillm.api.routes.CompletionService")
def test_query_ai_twin_success(mock_completion_service, mock_vector_store):
    # Setup mocks
    mock_store_instance = AsyncMock()
    mock_store_instance.find_similar_messages.return_value = [
        {"message": "Hello world", "similarity": 0.9}
    ]
    mock_store_instance.close = AsyncMock()
    mock_vector_store.return_value = mock_store_instance

    mock_completion_instance = AsyncMock()
    mock_completion_instance.generate_response.return_value = "AI response"
    mock_completion_service.return_value = mock_completion_instance

    response = client.post(
        "/query", json={"query": "How are you?", "user_id": "U123", "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "similar_messages" in data


@patch("phillm.api.routes.RedisVectorStore")
@patch("phillm.api.routes.CompletionService")
def test_query_ai_twin_no_messages_found(mock_completion_service, mock_vector_store):
    mock_store_instance = AsyncMock()
    mock_store_instance.find_similar_messages.return_value = []
    mock_store_instance.close = AsyncMock()
    mock_vector_store.return_value = mock_store_instance

    response = client.post("/query", json={"query": "How are you?", "user_id": "U123"})

    assert (
        response.status_code == 500
    )  # Should be 500 because HTTPException gets caught and wrapped
    assert "No messages found" in str(response.json())


@patch("phillm.api.routes.RedisVectorStore")
def test_get_user_message_count(mock_vector_store):
    mock_store_instance = AsyncMock()
    mock_store_instance.get_user_message_count.return_value = 42
    mock_store_instance.close = AsyncMock()
    mock_vector_store.return_value = mock_store_instance

    response = client.get("/users/U123/messages/count")

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "U123"
    assert data["message_count"] == 42


@patch("phillm.api.routes.RedisVectorStore")
def test_get_recent_messages(mock_vector_store):
    mock_store_instance = AsyncMock()
    mock_store_instance.get_recent_messages.return_value = [
        {"message": "Hello", "timestamp": 1234567890.123}
    ]
    mock_store_instance.close = AsyncMock()
    mock_vector_store.return_value = mock_store_instance

    response = client.get("/users/U123/messages/recent?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "U123"
    assert len(data["messages"]) == 1


@patch("phillm.api.routes.RedisVectorStore")
def test_delete_user_messages(mock_vector_store):
    mock_store_instance = AsyncMock()
    mock_store_instance.delete_user_messages.return_value = 5
    mock_store_instance.close = AsyncMock()
    mock_vector_store.return_value = mock_store_instance

    response = client.delete("/users/U123/messages")

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "U123"
    assert data["deleted_count"] == 5
