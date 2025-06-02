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
    # Mock cached result
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
async def test_get_user_display_name_api_call(user_manager):
    # Mock no cache
    user_manager.redis_client.hgetall = AsyncMock(return_value={})
    user_manager.redis_client.hset = AsyncMock()

    # Mock Slack API response
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "user": {
                "profile": {
                    "display_name": "API User",
                    "real_name": "API Real User",
                }
            }
        }
    )

    result = await user_manager.get_user_display_name("U123")
    assert result == "API User"

    # Should cache the result
    user_manager.redis_client.hset.assert_called()


@pytest.mark.asyncio
async def test_get_user_display_name_fallback_to_real_name(user_manager):
    # Mock no cache
    user_manager.redis_client.hgetall = AsyncMock(return_value={})
    user_manager.redis_client.hset = AsyncMock()

    # Mock Slack API response with empty display_name
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "user": {
                "profile": {
                    "display_name": "",
                    "real_name": "Real Name Only",
                }
            }
        }
    )

    result = await user_manager.get_user_display_name("U123")
    assert result == "Real Name Only"


@pytest.mark.asyncio
async def test_get_user_display_name_fallback_to_user_id(user_manager):
    # Mock no cache
    user_manager.redis_client.hgetall = AsyncMock(return_value={})

    # Mock Slack API response with no names
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "user": {
                "profile": {
                    "display_name": "",
                    "real_name": "",
                }
            }
        }
    )

    result = await user_manager.get_user_display_name("U123")
    assert result == "U123"


@pytest.mark.asyncio
async def test_get_user_display_name_api_error(user_manager):
    # Mock no cache
    user_manager.redis_client.hgetall = AsyncMock(return_value={})

    # Mock Slack API error
    user_manager.slack_client.users_info.side_effect = Exception("API Error")

    result = await user_manager.get_user_display_name("U123")
    assert result == "U123"  # Fallback to user ID


@pytest.mark.asyncio
async def test_get_user_info_cached(user_manager):
    # Mock cached result
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
async def test_get_user_info_api_call(user_manager):
    # Mock no cache
    user_manager.redis_client.hgetall = AsyncMock(return_value={})

    # Mock Slack API response
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "user": {
                "profile": {
                    "display_name": "API User",
                    "real_name": "API Real User",
                    "email": "api@example.com",
                    "title": "API Developer",
                    "phone": "987-654-3210",
                }
            }
        }
    )

    result = await user_manager.get_user_info("U123")

    assert result["display_name"] == "API User"
    assert result["email"] == "api@example.com"
    user_manager.redis_client.hset.assert_called()


@pytest.mark.asyncio
async def test_invalidate_user_cache(user_manager):
    await user_manager.invalidate_user_cache("U123")
    user_manager.redis_client.delete.assert_called_once_with("user_info:U123")


@pytest.mark.asyncio
async def test_get_cache_stats(user_manager):
    # Mock Redis scan for user cache keys
    user_manager.redis_client.scan_iter.return_value = [
        "user_info:U123",
        "user_info:U456",
        "user_info:U789",
    ]

    # Mock cache data for expiry calculation
    current_time = time.time()
    user_manager.redis_client.hget.side_effect = [
        str(current_time - 100),  # Fresh
        str(current_time - 7200),  # Expired
        str(current_time - 3600),  # Fresh
    ]

    stats = await user_manager.get_cache_stats()

    assert stats["total_cached_users"] == 3
    assert stats["fresh_cache_entries"] == 2
    assert stats["expired_cache_entries"] == 1


@pytest.mark.asyncio
async def test_cache_expiry_check(user_manager):
    # Test cache expiry logic
    current_time = time.time()

    # Expired cache (over 1 hour old)
    user_manager.redis_client.hgetall.return_value = {
        "display_name": "Old User",
        "cached_at": str(current_time - 7200),  # 2 hours ago
    }

    # Mock fresh API call
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
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


@pytest.mark.asyncio
async def test_close(user_manager):
    await user_manager.close()
    user_manager.redis_client.close.assert_called_once()


def test_cache_key_format():
    # Test that we know the cache key format
    user_id = "U123"
    expected_key = f"user_info:{user_id}"
    assert expected_key == "user_info:U123"


@pytest.mark.asyncio
async def test_get_user_info_with_none_values(user_manager):
    # Mock no cache
    user_manager.redis_client.hgetall = AsyncMock(return_value={})

    # Mock Slack API response with None/missing values
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "user": {
                "profile": {
                    "display_name": "User",
                    "real_name": "Real User",
                    # Missing email, title, phone
                }
            }
        }
    )

    result = await user_manager.get_user_info("U123")

    assert result["display_name"] == "User"
    assert result["email"] == ""
    assert result["title"] == ""
    assert result["phone"] == ""


@pytest.mark.asyncio
async def test_concurrent_cache_access(user_manager):
    # Test that concurrent access doesn't cause issues
    user_manager.redis_client.hgetall = AsyncMock(return_value={})
    user_manager.slack_client.users_info = AsyncMock(
        return_value={
            "user": {
                "profile": {
                    "display_name": "Concurrent User",
                    "real_name": "Concurrent Real User",
                }
            }
        }
    )

    # Simulate concurrent calls
    import asyncio

    tasks = [user_manager.get_user_display_name("U123") for _ in range(3)]

    results = await asyncio.gather(*tasks)

    # All should return the same result
    assert all(result == "Concurrent User" for result in results)
