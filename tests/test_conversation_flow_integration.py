"""
Integration test for the new conversation flow using message history structure
"""

import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from phillm.ai.completions import CompletionService

# Set dummy environment variable for testing
os.environ["OPENAI_API_KEY"] = "test_key"


def test_completion_service_message_structure():
    """Test that completion service properly builds message structure"""

    completion_service = CompletionService()

    # Test the new _build_messages method
    system_prompt = "You are PhiLLM, an AI assistant."
    conversation_history = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have real-time weather data."},
        {"role": "user", "content": "Tell me about APIs then."},
    ]
    current_query = "How do REST APIs work?"

    messages = completion_service._build_messages(
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        current_query=current_query,
    )

    # Verify structure
    assert len(messages) == 5  # system + 3 history + current query
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt

    # Verify conversation history is preserved
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What's the weather like?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "I don't have real-time weather data."
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Tell me about APIs then."

    # Verify current query is last
    assert messages[4]["role"] == "user"
    assert messages[4]["content"] == "How do REST APIs work?"

    print("✅ Message structure test passed!")


def test_redisvl_llm_role_conversion():
    """Test that conversation history with RedisVL 'llm' roles are converted to 'assistant'"""

    # Simulate what ConversationSessionManager.get_conversation_history_for_prompt does
    relevant_messages = [
        {
            "role": "user",
            "content": "What's the weather like?",
            "metadata": {"venue_type": "dm", "timestamp": 1234567890.0},
        },
        {
            "role": "llm",  # RedisVL uses 'llm' for bot responses
            "content": "I don't have real-time weather data.",
            "metadata": {"venue_type": "dm", "timestamp": 1234567891.0},
        },
        {
            "role": "user",
            "content": "Tell me about APIs then.",
            "metadata": {"venue_type": "dm", "timestamp": 1234567892.0},
        },
    ]

    # Apply the role conversion logic from ConversationSessionManager
    formatted_messages = []
    for msg in relevant_messages:
        # Convert RedisVL "llm" role to OpenAI-compatible "assistant" role
        role = msg.get("role", "user")
        if role == "llm":
            role = "assistant"

        # Only include role and content for the chat completion
        formatted_messages.append({"role": role, "content": msg.get("content", "")})

    # Verify conversion worked correctly
    assert len(formatted_messages) == 3
    assert formatted_messages[0]["role"] == "user"
    assert formatted_messages[1]["role"] == "assistant"  # Converted from "llm"
    assert formatted_messages[2]["role"] == "user"

    # Verify all roles are valid for OpenAI
    valid_roles = {"system", "user", "assistant"}
    for msg in formatted_messages:
        assert msg["role"] in valid_roles, f"Invalid role for OpenAI: {msg['role']}"

    print("✅ RedisVL 'llm' role conversion test passed!")


def test_system_prompt_changes():
    """Test that system prompt no longer includes conversation context"""

    completion_service = CompletionService()

    similar_messages = [
        {"message": "hey what's up", "similarity": 0.8},
        {"message": "not much, just coding", "similarity": 0.7},
    ]

    # Build system prompt (no conversation context in new version)
    system_prompt = completion_service._build_system_prompt(
        user_id="test_user",
        similar_messages=similar_messages,
        is_dm=True,
        requester_display_name="John",
    )

    # Verify system prompt structure
    assert "Your task is to perform style transfer" in system_prompt
    assert "hey what's up" in system_prompt  # Style examples should be included
    assert "The person messaging you is John" in system_prompt

    # Verify NO conversation context in system prompt (moved to message history)
    assert "Recent conversation context:" not in system_prompt
    assert (
        "Use the conversation history provided in the message thread" in system_prompt
    )

    print("✅ System prompt structure test passed!")


async def test_conversation_flow_integration():
    """Test the complete conversation flow with mocked components"""

    with patch("openai.AsyncOpenAI") as mock_openai:
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yeah, that sounds cool!"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        completion_service = CompletionService()

        # Test data
        query = "What do you think about the new API?"
        similar_messages = [
            {"message": "APIs are pretty neat", "similarity": 0.8},
            {"message": "yeah i like working with them", "similarity": 0.7},
        ]
        conversation_history = [
            {"role": "user", "content": "Hey, how's it going?"},
            {"role": "assistant", "content": "going well, thanks!"},
            {"role": "user", "content": "Working on any interesting projects?"},
        ]

        # Generate response
        response = await completion_service.generate_response(
            query=query,
            similar_messages=similar_messages,
            user_id="test_user",
            is_dm=True,
            conversation_history=conversation_history,
            requester_display_name="Alice",
        )

        # Verify response
        assert response == "Yeah, that sounds cool!"

        # Verify OpenAI was called with correct message structure
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]

        messages = call_args["messages"]

        # Should have: system + conversation history + current query
        assert len(messages) == 5  # system + 3 history + current query
        assert messages[0]["role"] == "system"
        assert "style transfer" in messages[0]["content"].lower()

        # Verify conversation history is in messages
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hey, how's it going?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "going well, thanks!"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Working on any interesting projects?"

        # Verify current query is last
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == "What do you think about the new API?"

        print("✅ Complete conversation flow test passed!")


if __name__ == "__main__":
    # Run tests
    print("🧪 Testing new conversation flow implementation...")

    test_completion_service_message_structure()
    test_redisvl_llm_role_conversion()
    test_system_prompt_changes()

    # Run async test
    asyncio.run(test_conversation_flow_integration())

    print("✅ All conversation flow tests passed!")
    print()
    print("📋 Summary of Changes:")
    print("  • Conversation context moved from system prompt to message history")
    print("  • SemanticSessionManager integration for relevant context retrieval")
    print("  • Message structure now: [system, ...history, current_query]")
    print("  • Privacy-aware venue filtering (DM context not leaked to channels)")
    print("  • Semantic relevance instead of chronological recent messages")
