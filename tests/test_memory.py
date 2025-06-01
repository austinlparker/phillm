import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time
from datetime import datetime

from phillm.memory.conversation_memory import (
    ConversationMemory,
    Memory,
    MemoryType,
    MemoryImportance,
)


@pytest.fixture
def memory():
    with patch("phillm.memory.conversation_memory.redis.from_url") as mock_redis, patch(
        "phillm.memory.conversation_memory.get_tracer"
    ) as mock_tracer:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        
        # Mock tracer with a proper span context manager
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)
        
        mock_tracer_instance = MagicMock()
        mock_tracer_instance.start_as_current_span.return_value = mock_span
        mock_tracer.return_value = mock_tracer_instance
        
        memory_instance = ConversationMemory()
        memory_instance.redis_client = mock_client
        return memory_instance


@pytest.mark.asyncio
async def test_store_memory_success(memory):
    mock_embedding_service = AsyncMock()
    mock_embedding_service.create_embedding.return_value = [0.1, 0.2, 0.3]
    memory.set_embedding_service(mock_embedding_service)

    memory.redis_client.hset = AsyncMock()
    memory.redis_client.sadd = AsyncMock()
    memory._cleanup_old_memories = AsyncMock()

    memory_id = await memory.store_memory(
        user_id="U123",
        memory_type=MemoryType.DM_CONVERSATION,
        content="Test conversation",
        context={"channel_id": "C123"},
        importance=MemoryImportance.HIGH,
    )

    assert memory_id.startswith("U123:dm_conversation:")
    memory.redis_client.hset.assert_called_once()
    memory.redis_client.sadd.assert_called()


@pytest.mark.asyncio
async def test_store_dm_interaction(memory):
    memory.store_memory = AsyncMock(return_value="memory_123")

    await memory.store_dm_interaction(
        user_id="U123",
        user_message="Hello",
        bot_response="Hi there!",
        channel_id="D123",
        query_embedding=[0.1, 0.2, 0.3],
    )

    memory.store_memory.assert_called_once()
    args = memory.store_memory.call_args[1]
    assert args["user_id"] == "U123"
    assert args["memory_type"] == MemoryType.DM_CONVERSATION
    assert "Hello" in args["content"]
    assert "Hi there!" in args["content"]


@pytest.mark.asyncio
async def test_recall_memories_with_query(memory):
    # Mock existing memories
    memory_data = {
        "memory_id": "U123:dm_conversation:123",
        "user_id": "U123",
        "memory_type": "dm_conversation",
        "content": "Test memory content",
        "context": json.dumps({"channel_id": "D123"}),
        "importance": "3",
        "timestamp": str(time.time()),
        "embedding": json.dumps([0.1, 0.2, 0.3]),
        "decay_factor": "1.0",
        "access_count": "0",
        "last_accessed": "",
    }

    memory.redis_client.smembers.return_value = ["U123:dm_conversation:123"]
    memory.redis_client.hgetall.return_value = memory_data

    # Mock embedding service
    mock_embedding_service = AsyncMock()
    mock_embedding_service.create_embedding.return_value = [0.1, 0.2, 0.3]
    memory.set_embedding_service(mock_embedding_service)

    # Mock update memory access
    memory._update_memory_access = AsyncMock()

    with patch.object(memory, "_cosine_similarity", return_value=0.8):
        memories = await memory.recall_memories(
            user_id="U123", query="test query", limit=5
        )

    assert len(memories) == 1
    assert memories[0].content == "Test memory content"


@pytest.mark.asyncio
async def test_get_conversation_context(memory):
    memory.recall_memories = AsyncMock()

    # Create mock memory
    mock_memory = Memory(
        memory_id="test_123",
        user_id="U123",
        memory_type=MemoryType.DM_CONVERSATION,
        content="Previous conversation",
        context={"channel_id": "D123"},
        importance=MemoryImportance.HIGH,
        timestamp=time.time(),
    )
    memory.recall_memories.return_value = [mock_memory]

    context = await memory.get_conversation_context("U123", limit=3)

    assert "Previous conversation" in context
    memory.recall_memories.assert_called_once()


def test_memory_calculate_relevance_score():
    memory = Memory(
        memory_id="test_123",
        user_id="U123",
        memory_type=MemoryType.DM_CONVERSATION,
        content="Test content",
        context={},
        importance=MemoryImportance.HIGH,
        timestamp=time.time() - 86400,  # 1 day ago
        access_count=5,
        last_accessed=time.time() - 3600,  # 1 hour ago
    )

    score = memory.calculate_relevance_score(time.time(), base_similarity=0.5)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_get_memory_stats(memory):
    # Mock memories
    memory_data = [
        {
            "memory_id": "U123:dm_conversation:1",
            "user_id": "U123",
            "memory_type": "dm_conversation",
            "content": "Test content 1",
            "context": "{}",
            "importance": "3",
            "timestamp": str(time.time() - 100),
            "embedding": "",
            "decay_factor": "1.0",
            "access_count": "5",
            "last_accessed": "",
        },
        {
            "memory_id": "U123:channel_interaction:2",
            "user_id": "U123",
            "memory_type": "channel_interaction",
            "content": "Test content 2",
            "context": "{}",
            "importance": "2",
            "timestamp": str(time.time()),
            "embedding": "",
            "decay_factor": "1.0",
            "access_count": "2",
            "last_accessed": "",
        },
    ]

    memory.redis_client.smembers.return_value = ["mem1", "mem2"]
    memory.redis_client.hgetall.side_effect = memory_data

    stats = await memory.get_memory_stats("U123")

    assert stats["total_memories"] == 2
    assert "dm_conversation" in stats["by_type"]
    assert "channel_interaction" in stats["by_type"]
    assert stats["most_accessed"]["access_count"] == 5


@pytest.mark.asyncio
async def test_cleanup_old_memories(memory):
    # Create mock memories - more than max
    memory.max_memories_per_user = 2

    old_memory = Memory(
        memory_id="old_123",
        user_id="U123",
        memory_type=MemoryType.DM_CONVERSATION,
        content="Old content",
        context={},
        importance=MemoryImportance.LOW,
        timestamp=time.time() - 86400 * 10,  # 10 days ago
        access_count=0,
    )

    recent_memory = Memory(
        memory_id="recent_123",
        user_id="U123",
        memory_type=MemoryType.DM_CONVERSATION,
        content="Recent content",
        context={},
        importance=MemoryImportance.HIGH,
        timestamp=time.time(),
        access_count=10,
    )

    memory._get_user_memories = AsyncMock(return_value=[old_memory, recent_memory])
    memory._delete_memory = AsyncMock()

    await memory._cleanup_old_memories("U123")

    # Should not delete anything as we have exactly max_memories_per_user
    memory._delete_memory.assert_not_called()

    # Test with too many memories
    memory._get_user_memories.return_value = [old_memory, recent_memory, old_memory]
    await memory._cleanup_old_memories("U123")

    # Should delete the least relevant memory
    memory._delete_memory.assert_called()


@pytest.mark.asyncio
async def test_close(memory):
    await memory.close()
    memory.redis_client.close.assert_called_once()