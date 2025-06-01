import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from phillm.ai.completions import CompletionService


@pytest.fixture
def completion_service():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        return CompletionService()


@pytest.mark.asyncio
async def test_generate_response_success(completion_service):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response"
    mock_response.usage = MagicMock()
    mock_response.usage.completion_tokens = 10
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.total_tokens = 30

    # Mock telemetry tracer
    with (
        patch("phillm.ai.completions.get_tracer") as mock_get_tracer,
        patch.object(completion_service.client, "chat") as mock_chat,
    ):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        mock_chat.completions.create = AsyncMock(return_value=mock_response)

        similar_messages = [
            {"message": "Hey there! How's it going?", "similarity": 0.8},
            {"message": "I'm doing pretty well today", "similarity": 0.7},
        ]

        result = await completion_service.generate_response(
            query="How are you?",
            similar_messages=similar_messages,
            user_id="U123",
        )

        assert result == "This is a test response"
        mock_chat.completions.create.assert_called_once()

        # Check that system prompt was created
        call_args = mock_chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "How are you?"


@pytest.mark.asyncio
async def test_generate_scheduled_message_success(completion_service):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Good morning team!"

    with patch.object(completion_service.client, "chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(return_value=mock_response)

        result = await completion_service.generate_scheduled_message(
            context="Previous messages about morning greetings",
            user_id="U123",
            topic="morning greeting",
        )

        assert result == "Good morning team!"
        mock_chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_error_handling(completion_service):
    # Mock telemetry tracer
    with (
        patch("phillm.ai.completions.get_tracer") as mock_get_tracer,
        patch.object(completion_service.client, "chat") as mock_chat,
    ):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        mock_chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        similar_messages = [{"message": "test message", "similarity": 0.5}]

        with pytest.raises(Exception, match="API Error"):
            await completion_service.generate_response(
                query="test", similar_messages=similar_messages, user_id="U123"
            )


def test_build_system_prompt(completion_service):
    similar_messages = [
        {"message": "Hello there! How's it going?"},
        {"message": "I'm doing pretty well today, thanks for asking"},
    ]

    prompt = completion_service._build_system_prompt(
        "U123", similar_messages, "How are you?"
    )

    assert "Hello there! How's it going?" in prompt
    assert "I'm doing pretty well today" in prompt
    assert "style transfer" in prompt
    assert "PhiLLM" in prompt
    assert "Phillip's unique style" in prompt


def test_build_scheduled_prompt_with_topic(completion_service):
    context = "Previous messages"

    prompt = completion_service._build_scheduled_prompt(
        "U123", context, "morning update"
    )

    assert "U123" in prompt
    assert context in prompt
    assert "morning update" in prompt


def test_build_scheduled_prompt_without_topic(completion_service):
    context = "Previous messages"

    prompt = completion_service._build_scheduled_prompt("U123", context, None)

    assert "U123" in prompt
    assert context in prompt
    assert "morning update" not in prompt
