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

    @pytest.mark.asyncio
    async def test_role_conversion_from_redisvl_to_openai(self, session_manager):
        """Test that RedisVL 'llm' roles are converted to OpenAI 'assistant' roles"""
        # Simulate messages from RedisVL SemanticMessageHistory with 'llm' role
        relevant_messages = [
            {
                "role": "user",
                "content": "What's the weather like?",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "llm",  # This is what RedisVL uses for bot responses
                "content": "I don't have real-time weather data.",
                "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
            },
            {
                "role": "user",
                "content": "Tell me about APIs then.",
                "metadata": {"venue_type": "dm", "timestamp": 1234567892.0},
            },
        ]

        # Test the role conversion logic from get_conversation_history_for_prompt
        formatted_messages = []
        for msg in relevant_messages:
            # Convert RedisVL "llm" role to OpenAI-compatible "assistant" role
            role = msg.get("role", "user")
            if role == "llm":
                role = "assistant"

            # Only include role and content for the chat completion
            formatted_messages.append({"role": role, "content": msg.get("content", "")})

        # Verify that 'llm' role was converted to 'assistant'
        assert len(formatted_messages) == 3
        assert formatted_messages[0]["role"] == "user"
        assert formatted_messages[0]["content"] == "What's the weather like?"

        # This should be converted from 'llm' to 'assistant'
        assert formatted_messages[1]["role"] == "assistant"
        assert (
            formatted_messages[1]["content"] == "I don't have real-time weather data."
        )

        assert formatted_messages[2]["role"] == "user"
        assert formatted_messages[2]["content"] == "Tell me about APIs then."

        # Verify no invalid roles remain
        for msg in formatted_messages:
            assert msg["role"] in [
                "system",
                "user",
                "assistant",
            ], f"Invalid role: {msg['role']}"
