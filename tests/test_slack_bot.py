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
            patch("phillm.slack.bot.RedisVectorStore"),
            patch("phillm.slack.bot.ConversationMemory"),
            patch("phillm.slack.bot.UserManager"),
        ):
            bot = SlackBot()
            bot.app.client = AsyncMock()
            bot.vector_store = AsyncMock()
            bot.memory = AsyncMock()
            bot.completion_service = AsyncMock()
            bot.user_manager = AsyncMock()
            return bot


@pytest.mark.asyncio
async def test_process_target_user_message(slack_bot):
    event = {
        "text": "Hello world",
        "channel": "C123",
        "ts": "1234567890.123",
        "user": "U123",
    }

    # Mock async methods properly
    slack_bot.vector_store.message_exists = AsyncMock(return_value=False)
    slack_bot.vector_store.store_message = AsyncMock(return_value=None)
    slack_bot.embedding_service.create_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    with patch("phillm.slack.bot.update_stats"):
        await slack_bot._process_target_user_message(event)

    slack_bot.vector_store.store_message.assert_called_once()


@pytest.mark.asyncio
async def test_handle_direct_message_event(slack_bot):
    event = {
        "user": "U456",
        "channel": "D123",
        "text": "Hello bot",
        "ts": "1234567890.123",
    }

    slack_bot.memory.get_conversation_context.return_value = "Previous context"
    slack_bot.user_manager.get_user_display_name.return_value = "Test User"
    slack_bot._generate_ai_response = AsyncMock(return_value=("Hi there!", [0.1, 0.2]))

    with patch("phillm.slack.bot.update_stats"), patch("phillm.slack.bot.get_tracer"):
        await slack_bot._handle_direct_message_event(event)

    slack_bot.app.client.reactions_add.assert_called()
    slack_bot.app.client.chat_postMessage.assert_called()
    slack_bot.app.client.reactions_remove.assert_called()


@pytest.mark.asyncio
async def test_generate_ai_response(slack_bot):
    query = "How are you?"

    # Mock embedding service
    with patch("phillm.slack.bot.EmbeddingService") as mock_embedding_service:
        mock_service = AsyncMock()
        mock_service.create_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding_service.return_value = mock_service

        # Mock async methods
        slack_bot.vector_store.find_similar_messages = AsyncMock(return_value=[
            {"message": "I'm good", "similarity": 0.8}
        ])
        slack_bot.completion_service.generate_response = AsyncMock(return_value="I'm doing well!")

        response, embedding = await slack_bot._generate_ai_response(query)

        assert response == "I'm doing well!"
        assert embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_generate_ai_response_with_fallback(slack_bot):
    query = "How are you?"

    # Mock embedding service
    with patch("phillm.slack.bot.EmbeddingService") as mock_embedding_service:
        mock_service = AsyncMock()
        mock_service.create_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding_service.return_value = mock_service

        # No similar messages found, should use recent messages fallback
        slack_bot.vector_store.find_similar_messages = AsyncMock(side_effect=[[], []])
        slack_bot.vector_store.get_recent_messages = AsyncMock(return_value=[
            {"message": "Recent message"}
        ])
        slack_bot.completion_service.generate_response = AsyncMock(return_value="Fallback response")

        response, embedding = await slack_bot._generate_ai_response(query)

        assert response == "Fallback response"
        slack_bot.vector_store.get_recent_messages.assert_called()


@pytest.mark.asyncio
async def test_store_channel_interaction_memory(slack_bot):
    event = {
        "user": "U456",
        "text": "Hello everyone",
        "channel": "C123",
    }

    slack_bot.app.client.conversations_info.return_value = {
        "channel": {"name": "general"}
    }

    await slack_bot._store_channel_interaction_memory(event)

    slack_bot.memory.store_channel_interaction.assert_called_once_with(
        user_id="U456",
        message="Hello everyone",
        channel_id="C123",
        channel_name="general",
    )


@pytest.mark.asyncio
async def test_check_scraping_completeness_simple_complete(slack_bot):
    channel_id = "C123"

    # Mock oldest stored message
    slack_bot.vector_store.get_oldest_stored_message.return_value = {
        "timestamp": 1234567890.0,
        "message": "Oldest message",
    }

    # Mock API response - no older messages
    slack_bot.app.client.conversations_history.return_value = {"messages": []}

    result = await slack_bot.check_scraping_completeness_simple(channel_id)

    assert result["complete"] is True
    assert "No older messages exist" in result["reason"]


@pytest.mark.asyncio
async def test_check_scraping_completeness_simple_incomplete(slack_bot):
    channel_id = "C123"

    # Mock oldest stored message
    slack_bot.vector_store.get_oldest_stored_message.return_value = {
        "timestamp": 1234567890.0,
        "message": "Oldest message",
    }

    # Mock API response - found older messages from target user
    slack_bot.app.client.conversations_history.return_value = {
        "messages": [
            {"user": "U123", "text": "Older message", "ts": "1234567880.0"},
            {"user": "U999", "text": "Not target user", "ts": "1234567870.0"},
        ]
    }

    result = await slack_bot.check_scraping_completeness_simple(channel_id)

    assert result["complete"] is False
    assert result["needs_scraping"] is True
    assert "Found 1 older messages" in result["reason"]


@pytest.mark.asyncio
async def test_scrape_channel_history_with_rate_limit(slack_bot):
    channel_identifier = "general"

    # Mock channel lookup
    slack_bot.app.client.conversations_list.return_value = {
        "channels": [{"name": "general", "id": "C123"}]
    }

    # Mock scrape state
    slack_bot.vector_store.get_scrape_state.return_value = {
        "cursor": None,
        "last_message_ts": None,
        "oldest_processed": None,
        "updated_at": None,
    }

    # Mock first API call with rate limit
    slack_bot.app.client.conversations_history.side_effect = [
        Exception("ratelimited"),
        {
            "messages": [
                {"user": "U123", "text": "Test message", "ts": "1234567890.123"}
            ],
            "has_more": False,
            "response_metadata": {"next_cursor": None},
        },
    ]

    slack_bot.vector_store.message_exists.return_value = False
    slack_bot._process_target_user_message = AsyncMock()

    with patch("asyncio.sleep") as mock_sleep, patch("phillm.slack.bot.update_stats"):
        await slack_bot.scrape_channel_history(channel_identifier)

    # Should have slept due to rate limit
    mock_sleep.assert_called()
    slack_bot._process_target_user_message.assert_called_once()


@pytest.mark.asyncio
async def test_start_and_stop(slack_bot):
    # Mock the handler methods
    slack_bot.handler.start_async = AsyncMock()
    slack_bot.handler.close_async = AsyncMock()
    
    # The vector_store, memory, and user_manager are already AsyncMock from fixture
    # But let's ensure their close methods are properly mocked
    slack_bot.vector_store.close = AsyncMock()
    slack_bot.memory.close = AsyncMock()
    slack_bot.user_manager.close = AsyncMock()

    await slack_bot.start()
    slack_bot.handler.start_async.assert_called_once()

    await slack_bot.stop()
    slack_bot.handler.close_async.assert_called_once()
    slack_bot.vector_store.close.assert_called_once()
    slack_bot.memory.close.assert_called_once()
    slack_bot.user_manager.close.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_start_scraping_if_needed(slack_bot):
    # Mock channel resolution
    slack_bot.app.client.conversations_list.return_value = {
        "channels": [
            {"name": "general", "id": "C123"},
            {"name": "random", "id": "C456"},
        ]
    }

    # Mock completeness check - one complete, one needs scraping
    slack_bot.check_scraping_completeness_simple = AsyncMock()
    slack_bot.check_scraping_completeness_simple.side_effect = [
        {"needs_scraping": False, "reason": "Complete"},
        {"needs_scraping": True, "reason": "Missing messages"},
    ]

    # Mock scraping
    slack_bot.scrape_channel_history = AsyncMock()

    with patch("asyncio.create_task") as mock_create_task, patch("asyncio.sleep"):
        await slack_bot.check_and_start_scraping_if_needed()

    # Should only create task for the channel that needs scraping
    mock_create_task.assert_called_once()


def test_setup_event_handlers(slack_bot):
    # Test that event handlers are set up without errors
    slack_bot._setup_event_handlers()

    # Verify handlers are registered (app.event calls)
    assert slack_bot.app.event.call_count >= 3  # message, app_mention, catch-all


@pytest.mark.asyncio
async def test_handle_message_skip_bot_messages(slack_bot):
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
async def test_handle_mention_with_error(slack_bot):
    body = {
        "event": {
            "user": "U456",
            "channel": "C123",
            "text": "<@UBOT> hello",
        }
    }
    say = AsyncMock()

    # Mock error in response generation
    slack_bot._generate_ai_response = AsyncMock(side_effect=Exception("Test error"))

    await slack_bot._handle_mention(body, say)

    # Should send error message
    say.assert_called_with("Sorry, I'm having trouble generating a response right now.")
