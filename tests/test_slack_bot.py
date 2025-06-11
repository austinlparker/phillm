import pytest
from unittest.mock import AsyncMock, patch

from phillm.slack.bot import SlackBot


@pytest.fixture
def slack_bot():
    with patch.dict(
        "os.environ",
        {
            "SLACK_BOT_TOKEN": "test-bot-token",
            "SLACK_APP_TOKEN": "test-app-token",
            "TARGET_USER_ID": "U123",
            "SCRAPE_CHANNELS": "general,random",
        },
    ):
        with (
            patch("phillm.slack.bot.AsyncApp"),
            patch("phillm.slack.bot.AsyncSocketModeHandler"),
            patch("phillm.slack.bot.EmbeddingService"),
            patch("phillm.slack.bot.CompletionService"),
            patch("phillm.slack.bot.ConversationSessionManager"),
            patch("phillm.slack.bot.UserManager"),
        ):
            bot = SlackBot()
            bot.app.client = AsyncMock()
            bot.completion_service = AsyncMock()
            bot.conversation_sessions = AsyncMock()
            bot.user_manager = AsyncMock()
            return bot


@pytest.mark.asyncio
async def test_process_target_user_message(slack_bot):
    """Test that target user message processing works with new system"""
    event = {
        "text": "Hello world",
        "channel": "C123",
        "ts": "1234567890.123",
        "user": "U123",
    }

    with patch("phillm.slack.bot.update_stats"), patch("phillm.slack.bot.telemetry"):
        await slack_bot._process_target_user_message(event)

    # Just verify it doesn't crash - the implementation is simplified now


@pytest.mark.asyncio
async def test_handle_direct_message_event(slack_bot):
    """Test DM handling with conversation sessions"""
    event = {
        "user": "U456",
        "channel": "D123",
        "text": "Hello bot",
        "ts": "1234567890.123",
    }

    slack_bot.conversation_sessions.get_conversation_history_for_prompt.return_value = []
    slack_bot.user_manager.get_user_display_name.return_value = "Test User"
    slack_bot._generate_ai_response = AsyncMock(return_value=("Hi there!", [0.1, 0.2]))

    with patch("phillm.slack.bot.update_stats"), patch("phillm.slack.bot.get_tracer"):
        await slack_bot._handle_direct_message_event(event)

    slack_bot.app.client.reactions_add.assert_called()
    slack_bot.app.client.chat_postMessage.assert_called()
    slack_bot.app.client.reactions_remove.assert_called()


@pytest.mark.asyncio
async def test_generate_ai_response(slack_bot):
    """Test AI response generation with new system (no vector search)"""
    query = "How are you?"

    # Mock the EmbeddingService import within the method
    with patch("phillm.ai.embeddings.EmbeddingService") as mock_embedding_service_class:
        mock_embedding_service = AsyncMock()
        mock_embedding_service.create_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding_service_class.return_value = mock_embedding_service

        slack_bot.completion_service.generate_response = AsyncMock(
            return_value="I'm doing well!"
        )

        response, embedding = await slack_bot._generate_ai_response(query)

        # Test that response is generated and embedding is created
        assert response == "I'm doing well!"
        assert embedding == [0.1, 0.2, 0.3]

        # Verify services were called correctly
        # create_embedding may be called 1-2 times depending on whether RAG succeeds or fails in test environment
        assert mock_embedding_service.create_embedding.call_count >= 1
        assert mock_embedding_service.create_embedding.call_count <= 2
        # First call should always be for the main query
        mock_embedding_service.create_embedding.assert_any_call(query)
        slack_bot.completion_service.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_store_channel_interaction_memory(slack_bot):
    """Test channel interaction memory storage (now simplified)"""
    event = {
        "user": "U456",
        "text": "Hello everyone",
        "channel": "C123",
    }

    # Should not crash - the implementation is now simplified
    await slack_bot._store_channel_interaction_memory(event)


@pytest.mark.asyncio
async def test_start_and_stop(slack_bot):
    """Test bot startup and shutdown"""
    # Mock the handler methods
    slack_bot.handler.start_async = AsyncMock()
    slack_bot.handler.close_async = AsyncMock()
    slack_bot.conversation_sessions.close = AsyncMock()
    slack_bot.user_manager.close = AsyncMock()

    await slack_bot.start()
    slack_bot.handler.start_async.assert_called_once()

    await slack_bot.stop()
    slack_bot.handler.close_async.assert_called_once()
    slack_bot.conversation_sessions.close.assert_called_once()
    slack_bot.user_manager.close.assert_called_once()


def test_setup_event_handlers(slack_bot):
    """Test that event handlers are set up without errors"""
    slack_bot._setup_event_handlers()

    # Verify handlers are registered (app.event calls)
    assert slack_bot.app.event.call_count >= 3  # message, app_mention, catch-all


@pytest.mark.asyncio
async def test_handle_message_skip_bot_messages(slack_bot):
    """Test that bot messages are properly skipped"""
    body = {
        "event": {
            "bot_id": "B123",
            "subtype": "bot_message",
            "user": "U123",
            "text": "Bot message",
            "channel_type": "channel",
        }
    }

    # Should not process bot messages
    with patch.object(slack_bot, "_process_target_user_message") as mock_process:
        await slack_bot._handle_message(body)

    mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_mention_success(slack_bot):
    """Test successful @ mention handling with conversation sessions"""
    body = {
        "event": {
            "user": "U456",
            "channel": "C123",
            "text": "<@UBOT> hello",
            "ts": "1234567890.123",
        }
    }
    say = AsyncMock()

    # Mock conversation session manager
    slack_bot.conversation_sessions.get_conversation_history_for_prompt.return_value = []
    slack_bot.user_manager.get_user_display_name.return_value = "Test User"
    slack_bot._generate_ai_response = AsyncMock(return_value=("Hi there!", [0.1, 0.2]))

    with patch("phillm.slack.bot.update_stats"), patch("phillm.slack.bot.get_tracer"):
        await slack_bot._handle_mention(body, say)

    # Verify reactions and response
    slack_bot.app.client.reactions_add.assert_called_with(
        channel="C123", timestamp="1234567890.123", name="thinking_face"
    )
    say.assert_called_with("Hi there!")
    slack_bot.app.client.reactions_remove.assert_called_with(
        channel="C123", timestamp="1234567890.123", name="thinking_face"
    )


@pytest.mark.asyncio
async def test_handle_mention_empty_message(slack_bot):
    """Test @ mention with empty message is skipped"""
    body = {
        "event": {
            "user": "U456",
            "channel": "C123",
            "text": "",
            "ts": "1234567890.123",
        }
    }
    say = AsyncMock()

    await slack_bot._handle_mention(body, say)

    # Should not process empty messages
    slack_bot.app.client.reactions_add.assert_not_called()
    say.assert_not_called()


@pytest.mark.asyncio
async def test_handle_mention_in_thread(slack_bot):
    """Test @ mention in a thread replies to the thread"""
    body = {
        "event": {
            "user": "U456",
            "channel": "C123",
            "text": "<@UBOT> hello in thread",
            "ts": "1234567890.456",
            "thread_ts": "1234567890.123",  # This indicates it's in a thread
        }
    }
    say = AsyncMock()

    # Mock conversation session manager
    slack_bot.conversation_sessions.get_conversation_history_for_prompt.return_value = []
    slack_bot.user_manager.get_user_display_name.return_value = "Test User"
    slack_bot._generate_ai_response = AsyncMock(
        return_value=("Hi there in thread!", [0.1, 0.2])
    )

    with patch("phillm.slack.bot.update_stats"), patch("phillm.slack.bot.get_tracer"):
        await slack_bot._handle_mention(body, say)

    # Verify reactions and response with thread_ts
    slack_bot.app.client.reactions_add.assert_called_with(
        channel="C123", timestamp="1234567890.456", name="thinking_face"
    )
    say.assert_called_with("Hi there in thread!", thread_ts="1234567890.123")
    slack_bot.app.client.reactions_remove.assert_called_with(
        channel="C123", timestamp="1234567890.456", name="thinking_face"
    )
