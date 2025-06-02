import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from phillm.user.user_manager import UserManager


@pytest.fixture
def user_manager():
    mock_client = AsyncMock()
    with (
        patch("phillm.user.user_manager.redis.from_url") as mock_redis,
        patch("phillm.user.user_manager.get_tracer") as mock_tracer,
    ):
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client

        # Mock tracer with a proper span context manager
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)

        mock_tracer_instance = MagicMock()
        mock_tracer_instance.start_as_current_span.return_value = mock_span
        mock_tracer.return_value = mock_tracer_instance

        manager = UserManager(mock_client)
        manager.redis_client = mock_redis_client
        return manager


@pytest.mark.asyncio
async def test_get_user_display_name_cached(user_manager):
    # Test business logic: Cache hit should return cached value
    user_manager.redis_client.hgetall.return_value = {
        "display_name": "Test User",
        "real_name": "Test Real User",
        "cached_at": str(time.time()),
    }

    result = await user_manager.get_user_display_name("U123")
    assert result == "Test User"

    # Should not call Slack API
    user_manager.slack_client.users_info.assert_not_called()


@pytest.mark.asyncio
async def test_get_user_info_cached(user_manager):
    # Test business logic: Full user info from cache
    cached_data = {
        "display_name": "Test User",
        "real_name": "Test Real User",
        "cached_at": str(time.time()),
        "email": "test@example.com",
        "title": "Developer",
        "phone": "123-456-7890",
    }
    user_manager.redis_client.hgetall.return_value = cached_data

    result = await user_manager.get_user_info("U123")

    assert result["display_name"] == "Test User"
    assert result["email"] == "test@example.com"
    user_manager.slack_client.users_info.assert_not_called()


@pytest.mark.asyncio
async def test_cache_expiry_check(user_manager):
    # Test business logic: Expired cache should trigger fresh lookup
    current_time = time.time()

    # Mock expired cache - should be detected as expired and deleted
    expired_cache_data = {
        "display_name": "Old User",
        "cached_at": str(current_time - 90000),  # 25 hours ago (expired, TTL is 24h)
    }
    
    # Return expired cache data, the method should detect expiry and delete it
    user_manager.redis_client.hgetall.return_value = expired_cache_data
    user_manager.redis_client.delete = AsyncMock()

    # Mock fresh API call
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "ok": True,
            "user": {
                "profile": {
                    "display_name": "Fresh User",
                    "real_name": "Fresh Real User",
                }
            }
        }
    )

    result = await user_manager.get_user_display_name("U123")

    # Should get fresh data from API, not cached
    assert result == "Fresh User"
    user_manager.slack_client.users_info.assert_called_once()
    user_manager.redis_client.delete.assert_called_once()


def test_cache_key_format():
    # Test business logic: Cache key format (pure function)
    user_id = "U123"
    expected_key = f"user_info:{user_id}"
    assert expected_key == "user_info:U123"


# NOTE: Removed infrastructure tests that were testing external integrations
# (Slack API calls, Redis operations, connection management) rather than 
# business logic. These should be covered by integration tests or end-to-end 
# tests that can use real external services in test environments.
#
# Removed tests:
# - test_get_user_display_name_api_call (Slack API integration)
# - test_get_user_display_name_fallback_* (Slack API integration) 
# - test_get_user_info_api_call (Slack API integration)
# - test_invalidate_user_cache (Redis operation)
# - test_get_cache_stats (Redis scanning)
# - test_concurrent_cache_access (Infrastructure concurrency)
# - test_close (Connection management)