import pytest
import asyncio
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch
from phillm.conversation import ConversationSessionManager


class TestConversationSessionManager:
    """Test the new conversation session management system"""

    @pytest.fixture
    def session_manager(self):
        """Create a conversation session manager for testing"""
        # Mock the EmbeddingService initialization
        with patch("phillm.conversation.session_manager.EmbeddingService"):
            manager = ConversationSessionManager()

        # Mock Redis connections to avoid real Redis dependency
        from redis import Redis

        manager.redis_client = AsyncMock()
        manager.sync_redis_client = MagicMock(spec=Redis)
        manager.redis_client.ping = AsyncMock()

        # Mock embedding service
        manager.embedding_service = MagicMock()
        manager.embedding_service.create_embedding = AsyncMock(
            return_value=[0.1] * 3072  # Mock embedding
        )

        return manager

    @pytest.mark.asyncio
    async def test_add_conversation_turn(self, session_manager):
        """Test adding a conversation turn to the session"""
        # Mock the session
        mock_session = MagicMock()
        # Mock get_relevant method for the total messages count call
        mock_session.get_relevant.return_value = [
            {"role": "user", "content": "Previous message 1"},
            {"role": "assistant", "content": "Previous response 1"},
        ]
        session_manager.user_sessions["test_user"] = mock_session

        venue_info = {"type": "dm", "channel_id": "D123456", "timestamp": 1234567890.0}

        await session_manager.add_conversation_turn(
            user_id="test_user",
            user_message="Hello, how are you?",
            bot_response="I'm doing well, thanks!",
            venue_info=venue_info,
        )

        # Verify the session store method was called
        mock_session.store.assert_called_once()
        call_args = mock_session.store.call_args

        assert "Hello, how are you?" in str(call_args)
        assert "I'm doing well, thanks!" in str(call_args)

        # Verify get_relevant was called (once for total messages count, once for immediate test)
        assert mock_session.get_relevant.call_count == 2

        # Verify the calls were made with correct parameters
        calls = mock_session.get_relevant.call_args_list
        # First call: get total messages count
        assert calls[0] == unittest.mock.call(
            prompt="", distance_threshold=1.0, top_k=1000
        )
        # Second call: immediate test retrieval
        assert calls[1] == unittest.mock.call(
            prompt="Hello, how are you?", distance_threshold=0.1, top_k=5
        )

    @pytest.mark.asyncio
    async def test_get_relevant_conversation_context(self, session_manager):
        """Test retrieving relevant conversation context"""
        # Mock the session and its get_relevant method
        mock_session = MagicMock()

        # Configure side_effect to return different values for different calls
        def mock_get_relevant(prompt, distance_threshold, top_k):
            if prompt == "":  # Call for total messages count
                return [
                    {"role": "user", "content": "Message 1"},
                    {"role": "assistant", "content": "Response 1"},
                    {"role": "user", "content": "Previous question about API"},
                    {"role": "assistant", "content": "Previous answer about API"},
                ]
            else:  # Call for actual query
                return [
                    {
                        "role": "user",
                        "content": "Previous question about API",
                        "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
                    },
                    {
                        "role": "assistant",
                        "content": "Previous answer about API",
                        "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
                    },
                ]

        mock_session.get_relevant.side_effect = mock_get_relevant

        session_manager.user_sessions["test_user"] = mock_session

        venue_info = {"type": "dm", "channel_id": "D123456"}

        result = await session_manager.get_relevant_conversation_context(
            user_id="test_user",
            current_query="Tell me more about the API",
            venue_info=venue_info,
        )

        # Verify get_relevant was called twice (once for total count, once for actual query)
        assert mock_session.get_relevant.call_count == 2

        # Check the second call (actual query) parameters
        call_args = mock_session.get_relevant.call_args[1]
        assert call_args["prompt"] == "Tell me more about the API"
        assert call_args["distance_threshold"] == 0.35
        assert call_args["top_k"] == 10

        # Verify results
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "API" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_venue_privacy_filtering(self, session_manager):
        """Test that DM context is filtered out of channel responses"""
        # Mock session with mixed venue messages
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = [
            {
                "role": "user",
                "content": "Private DM message",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "user",
                "content": "Public channel message",
                "metadata": {"venue_type": "channel", "timestamp": 1234567891.0},
            },
        ]

        session_manager.user_sessions["test_user"] = mock_session

        # Request context for a channel venue
        venue_info = {"type": "channel", "channel_id": "C123456"}

        result = await session_manager.get_relevant_conversation_context(
            user_id="test_user",
            current_query="What's the status?",
            venue_info=venue_info,
        )

        # Should filter out DM messages when responding in channel
        assert len(result) == 1
        assert result[0]["content"] == "Public channel message"
        assert "Private DM message" not in [msg["content"] for msg in result]

    @pytest.mark.asyncio
    async def test_conversation_history_for_prompt(self, session_manager):
        """Test getting conversation history formatted for chat completion"""
        # Mock session with conversation history
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = [
            {
                "role": "user",
                "content": "What's the weather?",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
            },
        ]

        session_manager.user_sessions["test_user"] = mock_session

        venue_info = {"type": "dm", "channel_id": "D123456"}

        result = await session_manager.get_conversation_history_for_prompt(
            user_id="test_user", current_query="And tomorrow?", venue_info=venue_info
        )

        # Verify format is correct for OpenAI chat completion
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "What's the weather?"}
        assert result[1] == {"role": "assistant", "content": "It's sunny today!"}

        # Metadata should be stripped for the prompt
        for msg in result:
            assert "metadata" not in msg

    @pytest.mark.asyncio
    async def test_session_stats(self, session_manager):
        """Test getting session statistics"""
        # Mock session with stats data
        mock_session = MagicMock()
        mock_session.get_relevant.return_value = [
            {
                "role": "user",
                "content": "Test message 1",
                "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
            },
            {
                "role": "assistant",
                "content": "Test response 1",
                "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
            },
            {
                "role": "user",
                "content": "Test message 2",
                "metadata": {"venue_type": "channel", "timestamp": 1234567892.0},
            },
        ]

        session_manager.user_sessions["test_user"] = mock_session

        stats = await session_manager.get_session_stats("test_user")

        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 2
        assert stats["bot_messages"] == 1
        assert stats["venue_breakdown"]["dm"] == 2
        assert stats["venue_breakdown"]["channel"] == 1


if __name__ == "__main__":
    # Simple test runner for direct execution
    import sys
    import os

    # Set dummy environment variables for testing
    os.environ["OPENAI_API_KEY"] = "test_key"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    async def run_tests():
        with patch(
            "phillm.conversation.session_manager.EmbeddingService"
        ) as mock_embedding_class:
            # Mock the embedding service class
            mock_embedding_instance = MagicMock()
            mock_embedding_instance.create_embedding = AsyncMock(
                return_value=[0.1] * 3072
            )
            mock_embedding_class.return_value = mock_embedding_instance

            with patch(
                "redisvl.utils.vectorize.text.openai.OpenAITextVectorizer"
            ) as mock_vectorizer_class:
                # Create a proper mock that satisfies the BaseVectorizer type check
                from redisvl.utils.vectorize import BaseVectorizer

                mock_vectorizer = MagicMock(spec=BaseVectorizer)
                mock_vectorizer.dims = 3072  # OpenAI text-embedding-3-large dimensions
                mock_vectorizer.dtype = "float32"
                mock_vectorizer_class.return_value = mock_vectorizer

                manager = ConversationSessionManager()

                # Mock Redis dependencies
                manager.redis_client = AsyncMock()
                manager.sync_redis_client = MagicMock()
                manager.redis_client.ping = AsyncMock()

                test_instance = TestConversationSessionManager()

                print("🧪 Running conversation session tests...")

                # Test conversation turn addition
                print("  ✓ Testing conversation turn addition...")
                await test_instance.test_add_conversation_turn(manager)

                print("✅ All basic tests passed!")

    if sys.version_info >= (3, 7):
        asyncio.run(run_tests())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_tests())
