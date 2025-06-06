from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from phillm.api.routes import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_check_healthy():
    """Test health endpoint with new conversation session manager"""
    with patch("phillm.api.routes.ConversationSessionManager") as mock_session_manager:
        mock_instance = AsyncMock()
        mock_instance._ensure_redis_connection = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_session_manager.return_value = mock_instance

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "PhiLLM",
            "redis": "connected",
        }


def test_health_check_unhealthy():
    """Test health endpoint when Redis connection fails"""
    with patch("phillm.api.routes.ConversationSessionManager") as mock_session_manager:
        mock_instance = AsyncMock()
        mock_instance._ensure_redis_connection.side_effect = Exception(
            "Connection failed"
        )
        mock_instance.close = AsyncMock()
        mock_session_manager.return_value = mock_instance

        response = client.get("/health")
        assert response.status_code == 503
        assert "unhealthy" in response.json()["detail"]["status"]


def test_disabled_endpoints_return_error():
    """Test that remaining disabled endpoints are properly disabled"""

    # Test query endpoint (kept but disabled)
    response = client.post(
        "/query", json={"query": "How are you?", "user_id": "U123", "limit": 5}
    )
    assert response.status_code in [500, 503]  # May be wrapped in exception handler
    assert "disabled" in response.json()["detail"] or "disabled" in str(response.json())

    # Test chat endpoint (kept but disabled)
    response = client.post("/chat", json={"message": "Hello", "user_id": "U123"})
    assert response.status_code in [500, 503]
    assert "disabled" in str(response.json())

    # Test memory stats endpoint (kept but disabled)
    response = client.get("/memory/U123/stats")
    assert response.status_code in [500, 503]
    assert "disabled" in str(response.json())


def test_removed_endpoints_return_404():
    """Test that removed endpoints return 404"""

    # Test removed message count endpoint
    response = client.get("/users/U123/messages/count")
    assert response.status_code == 404

    # Test removed recent messages endpoint
    response = client.get("/users/U123/messages/recent?limit=10")
    assert response.status_code == 404

    # Test removed delete messages endpoint
    response = client.delete("/users/U123/messages")
    assert response.status_code == 404

    # Test removed scraping status endpoint
    response = client.get("/scraping/status/C123")
    assert response.status_code == 404

    # Test removed reset scraping endpoint
    response = client.post("/scraping/reset/C123")
    assert response.status_code == 404

    # Test removed completeness endpoint
    response = client.get("/scraping/completeness/C123")
    assert response.status_code == 404

    # Test removed debug vector search endpoint
    response = client.get("/debug/vector-search")
    assert response.status_code == 404

    # Test removed memory recall endpoint
    response = client.get("/memory/U123/recall")
    assert response.status_code == 404

    # Test removed memory context endpoint
    response = client.get("/memory/U123/context")
    assert response.status_code == 404


def test_working_endpoints():
    """Test endpoints that should still work"""

    # User info endpoint should work (it only uses UserManager)
    with patch("phillm.slack.bot.SlackBot") as mock_slack_bot:
        mock_bot_instance = AsyncMock()
        mock_user_manager = AsyncMock()
        mock_user_manager.get_user_info.return_value = {
            "user_id": "U123",
            "display_name": "Test User",
            "real_name": "Test User",
        }
        mock_user_manager.close = AsyncMock()
        mock_bot_instance.app.client = AsyncMock()
        mock_slack_bot.return_value = mock_bot_instance

        with patch(
            "phillm.user.user_manager.UserManager", return_value=mock_user_manager
        ):
            response = client.get("/users/U123/info")
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "U123"

    # Cache stats endpoint should work
    with patch("phillm.user.user_manager.UserManager") as mock_user_manager_class:
        mock_user_manager = AsyncMock()
        mock_user_manager.get_cache_stats.return_value = {"total_cached_users": 10}
        mock_user_manager.close = AsyncMock()
        mock_user_manager_class.return_value = mock_user_manager

        response = client.get("/users/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "cache_stats" in data

    # Cache invalidation endpoint should work
    with patch("phillm.user.user_manager.UserManager") as mock_user_manager_class:
        mock_user_manager = AsyncMock()
        mock_user_manager.invalidate_user_cache = AsyncMock()
        mock_user_manager.close = AsyncMock()
        mock_user_manager_class.return_value = mock_user_manager

        response = client.post("/users/U123/cache/invalidate")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "U123"
        assert "successfully" in data["message"]
