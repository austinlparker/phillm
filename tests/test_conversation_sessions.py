import pytest
from unittest.mock import MagicMock, patch
from phillm.conversation import ConversationSessionManager


class TestConversationSessionManager:
    """Test the conversation session management system"""

    @pytest.fixture
    async def session_manager(self):
        """Create a session manager for testing"""
        with (
            patch("phillm.conversation.session_manager.redis.from_url"),
            patch("redis.Redis.from_url"),
        ):
            manager = ConversationSessionManager()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_add_conversation_turn_calls_store(self, session_manager):
        """Test that adding a conversation turn calls the session store method"""
        # Mock the session
        mock_session = MagicMock()
        session_manager.user_sessions["test_user"] = mock_session

        venue_info = {"type": "dm", "channel_id": "D123456", "timestamp": 1234567890.0}

        await session_manager.add_conversation_turn(
            user_id="test_user",
            user_message="Hello, how are you?",
            bot_response="I'm doing well, thanks!",
            venue_info=venue_info,
        )

        # Verify the session store method was called with correct parameters
        mock_session.store.assert_called_once_with(
            prompt="Hello, how are you?", response="I'm doing well, thanks!"
        )

    @pytest.mark.asyncio
    async def test_get_conversation_history_for_prompt_returns_formatted_messages(
        self, session_manager
    ):
        """Test that get_conversation_history_for_prompt returns properly formatted messages"""
        # Mock the session and its get_relevant method
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = [
            {
                "role": "user",
                "content": "Previous question",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "assistant",
                "content": "Previous answer",
                "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
            },
        ]

        session_manager.user_sessions["test_user"] = mock_session
        venue_info = {"type": "dm", "channel_id": "D123456"}

        result = await session_manager.get_conversation_history_for_prompt(
            user_id="test_user",
            current_query="Tell me more",
            venue_info=venue_info,
            max_messages=5,
        )

        # Verify the result format
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Previous question"}
        assert result[1] == {"role": "assistant", "content": "Previous answer"}

    @pytest.mark.asyncio
    async def test_venue_privacy_filtering(self, session_manager):
        """Test that DM messages are filtered when responding in public channels"""
        # Mock session with mixed DM and channel messages
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = [
            {
                "role": "user",
                "content": "DM message",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "user",
                "content": "Channel message",
                "metadata": {"venue_type": "channel", "timestamp": 1234567891.0},
            },
        ]

        session_manager.user_sessions["test_user"] = mock_session
        venue_info = {"type": "channel", "channel_id": "C123456"}

        result = await session_manager.get_conversation_history_for_prompt(
            user_id="test_user",
            current_query="Tell me more",
            venue_info=venue_info,
        )

        # Should only contain channel message, DM message should be filtered out
        assert len(result) == 1
        assert result[0]["content"] == "Channel message"

    @pytest.mark.asyncio
    async def test_clear_user_session(self, session_manager):
        """Test clearing a user's session"""
        # Mock session
        mock_session = MagicMock()
        session_manager.user_sessions["test_user"] = mock_session

        await session_manager.clear_user_session("test_user")

        # Verify clear was called
        mock_session.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_stats_with_messages(self, session_manager):
        """Test getting session statistics"""
        # Mock session with messages
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = [
            {
                "role": "user",
                "content": "Question 1",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "assistant",
                "content": "Answer 1",
                "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
            },
            {
                "role": "user",
                "content": "Question 2",
                "metadata": {"venue_type": "channel", "timestamp": 1234567892.0},
            },
        ]

        session_manager.user_sessions["test_user"] = mock_session

        stats = await session_manager.get_session_stats("test_user")

        # Verify stats structure
        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 2
        assert stats["bot_messages"] == 1
        assert "dm" in stats["venue_breakdown"]
        assert "channel" in stats["venue_breakdown"]
        assert stats["venue_breakdown"]["dm"] == 2  # 1 user + 1 bot message
        assert stats["venue_breakdown"]["channel"] == 1

    @pytest.mark.asyncio
    async def test_get_session_stats_empty_session(self, session_manager):
        """Test getting stats for empty session"""
        # Mock session with no messages
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = []

        session_manager.user_sessions["test_user"] = mock_session

        stats = await session_manager.get_session_stats("test_user")

        # Verify empty stats
        assert stats["total_messages"] == 0
        assert stats["user_messages"] == 0
        assert stats["bot_messages"] == 0
        assert stats["venue_breakdown"] == {}
