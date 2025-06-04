import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

from phillm.memory.conversation_memory import (
    ConversationMemory,
    Memory,
    MemoryType,
    MemoryImportance,
)


@pytest.fixture
def memory():
    with (
        patch("phillm.memory.conversation_memory.redis.from_url") as mock_redis,
        patch("phillm.memory.conversation_memory.get_tracer") as mock_tracer,
        patch("phillm.memory.conversation_memory.SearchIndex") as mock_search_index,
        patch("redis.Redis.from_url") as mock_sync_redis,
    ):
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()
        mock_redis.return_value = mock_async_client
        mock_sync_redis.return_value = mock_sync_client

        # Mock tracer with a proper span context manager
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)

        mock_tracer_instance = MagicMock()
        mock_tracer_instance.start_as_current_span.return_value = mock_span
        mock_tracer.return_value = mock_tracer_instance

        # Mock search index
        mock_index = MagicMock()
        mock_search_index.return_value = mock_index

        memory_instance = ConversationMemory()
        memory_instance.redis_client = mock_async_client
        memory_instance.sync_redis_client = mock_sync_client
        memory_instance.index = mock_index
        memory_instance._redis_healthy = True

        # Mock the connection methods to prevent actual Redis connections
        memory_instance._ensure_redis_connection = AsyncMock()
        memory_instance.initialize_index = AsyncMock()

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

    assert isinstance(memory_id, str) and len(memory_id) == 32  # MD5 hash
    memory.redis_client.hset.assert_called_once()
    # Verify the document key uses the new format
    call_args = memory.redis_client.hset.call_args
    assert call_args[0][0].startswith("mem:")


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
    # Mock embedding service
    mock_embedding_service = AsyncMock()
    mock_embedding_service.create_embedding.return_value = [0.1, 0.2, 0.3]
    memory.set_embedding_service(mock_embedding_service)

    # Mock vector search results
    mock_search_result = {
        "memory_id": "test_memory_123",
        "user_id": "U123",
        "memory_type": "dm_conversation",
        "content": "Test memory content",
        "context": json.dumps({"channel_id": "D123"}),
        "importance": "3",
        "timestamp": str(time.time()),
        "decay_factor": "1.0",
        "access_count": "0",
        "last_accessed": "0.0",
        "vector_distance": 0.2,  # High similarity
    }

    # Mock the vector search methods
    memory._vector_search_memories = AsyncMock(return_value=[mock_search_result])
    memory._update_memory_access_vector = AsyncMock()

    memories = await memory.recall_memories(user_id="U123", query="test query", limit=5)

    assert len(memories) == 1
    assert memories[0].content == "Test memory content"
    memory._vector_search_memories.assert_called_once()


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
    # Mock the method calls to avoid connection issues
    memory._get_user_memory_count = AsyncMock(return_value=2)

    # Mock Redis search responses for stats
    memory.redis_client.execute_command = AsyncMock()

    # Mock responses for different search queries
    async def mock_execute_command(*args):
        if "LIMIT" in args and args[-1] == "0":
            # Count queries return count as first element
            if "dm_conversation" in args[2]:
                return [1]  # 1 DM conversation
            elif "channel_interaction" in args[2]:
                return [1]  # 1 channel interaction
            elif "importance:[3 3]" in args[2]:
                return [1]  # 1 high importance
            elif "importance:[2 2]" in args[2]:
                return [1]  # 1 medium importance
            else:
                return [2]  # Total count
        elif "SORTBY" in args:
            if "timestamp" in args and "ASC" in args:
                # Oldest memory
                return [1, "mem:old", str(time.time() - 100)]
            elif "timestamp" in args and "DESC" in args:
                # Newest memory
                return [1, "mem:new", str(time.time())]
            elif "access_count" in args:
                # Most accessed memory
                return [
                    1,
                    "mem:accessed",
                    "content",
                    "Test content",
                    "access_count",
                    "5",
                    "memory_type",
                    "dm_conversation",
                ]
        return [0]

    memory.redis_client.execute_command.side_effect = mock_execute_command

    stats = await memory.get_memory_stats("U123")

    assert stats["total_memories"] == 2
    assert "dm_conversation" in stats["by_type"]
    assert "channel_interaction" in stats["by_type"]
    assert stats["most_accessed"]["access_count"] == 5


@pytest.mark.asyncio
async def test_cleanup_old_memories(memory):
    # Create mock memories - more than max
    memory.max_memories_per_user = 2

    # Mock memory count and search results
    memory._get_user_memory_count = AsyncMock(return_value=2)
    memory._get_recent_memories = AsyncMock(return_value=[])
    memory._delete_memory_vector = AsyncMock()

    await memory._cleanup_old_memories("U123")

    # Should not delete anything as we have exactly max_memories_per_user
    memory._delete_memory_vector.assert_not_called()

    # Test with too many memories
    memory._get_user_memory_count.return_value = 3
    mock_memories = [
        {
            "memory_id": "old_123",
            "user_id": "U123",
            "memory_type": "dm_conversation",
            "content": "Old content",
            "context": "{}",
            "importance": "1",  # LOW
            "timestamp": str(time.time() - 86400 * 10),  # 10 days ago
            "decay_factor": "1.0",
            "access_count": "0",
            "last_accessed": "0.0",
        },
        {
            "memory_id": "recent_123",
            "user_id": "U123",
            "memory_type": "dm_conversation",
            "content": "Recent content",
            "context": "{}",
            "importance": "3",  # HIGH
            "timestamp": str(time.time()),
            "decay_factor": "1.0",
            "access_count": "10",
            "last_accessed": str(time.time()),
        },
        {
            "memory_id": "old2_123",
            "user_id": "U123",
            "memory_type": "dm_conversation",
            "content": "Old content 2",
            "context": "{}",
            "importance": "1",  # LOW
            "timestamp": str(time.time() - 86400 * 5),  # 5 days ago
            "decay_factor": "1.0",
            "access_count": "1",
            "last_accessed": "0.0",
        },
    ]
    memory._get_recent_memories.return_value = mock_memories

    await memory._cleanup_old_memories("U123")

    # Should delete the least relevant memory
    memory._delete_memory_vector.assert_called()


@pytest.mark.asyncio
async def test_close(memory):
    # Ensure sync_redis_client has a close method for the test
    memory.sync_redis_client.close = MagicMock()

    await memory.close()
    memory.redis_client.close.assert_called_once()
    memory.sync_redis_client.close.assert_called_once()
