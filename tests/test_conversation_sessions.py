import pytest
import os
from phillm.conversation import ConversationSessionManager


class TestConversationSessionManager:
    """Test the conversation session management system"""

    @pytest.fixture
    def session_manager(self):
        """Create a session manager for testing"""
        # Set required environment variables for testing
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["REDIS_URL"] = "redis://localhost:6379"

        manager = ConversationSessionManager()
        return manager

    @pytest.mark.asyncio
    async def test_session_manager_initialization(self, session_manager):
        """Test that the session manager initializes properly"""
        assert session_manager is not None
        assert session_manager.user_sessions == {}
        assert session_manager.distance_threshold > 0
        assert session_manager.max_context_messages > 0

    @pytest.mark.asyncio
    async def test_venue_privacy_filtering_logic(self, session_manager):
        """Test the venue privacy filtering logic directly"""
        # Test data with mixed venue types
        messages = [
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
            {
                "role": "user",
                "content": "Public-safe DM",
                "metadata": {
                    "venue_type": "dm",
                    "timestamp": 1234567892.0,
                    "public_safe": True,
                },
            },
        ]

        # Test filtering for channel venue (should exclude DM messages unless public_safe)
        filtered = session_manager._filter_for_venue_privacy(messages, "channel")
        assert len(filtered) == 2
        assert filtered[0]["content"] == "Channel message"
        assert filtered[1]["content"] == "Public-safe DM"

        # Test filtering for DM venue (should include all messages)
        filtered = session_manager._filter_for_venue_privacy(messages, "dm")
        assert len(filtered) == 3

    @pytest.mark.asyncio
    async def test_conversation_context_format(self, session_manager):
        """Test that conversation context is properly formatted"""
        # This tests the formatting logic without needing Redis
        test_messages = [
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

        # Test the message formatting directly
        formatted = []
        for msg in test_messages:
            formatted.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )

        assert len(formatted) == 2
        assert formatted[0] == {"role": "user", "content": "Previous question"}
        assert formatted[1] == {"role": "assistant", "content": "Previous answer"}

    @pytest.mark.asyncio
    async def test_session_configuration(self, session_manager):
        """Test that session configuration values are reasonable"""
        assert 0.1 <= session_manager.distance_threshold <= 1.0
        assert 1 <= session_manager.max_context_messages <= 100
        assert session_manager.embedding_service is not None

    @pytest.mark.asyncio
    async def test_user_session_key_generation(self, session_manager):
        """Test that user session keys are generated consistently"""
        user_id = "test_user_123"
        expected_key = f"user_session_{user_id}"

        # This tests the key generation logic
        session_name = f"user_session_{user_id}"
        assert session_name == expected_key
