import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from phillm.vector.redis_vector_store import RedisVectorStore


@pytest.fixture
def vector_store():
    with (
        patch("phillm.vector.redis_vector_store.redis.from_url") as mock_redis_async,
        patch("redis.Redis.from_url") as mock_redis_sync,
    ):
        # Mock async Redis client
        mock_async_client = AsyncMock()
        mock_redis_async.return_value = mock_async_client

        # Mock sync Redis client for RedisVL
        mock_sync_client = MagicMock()
        mock_redis_sync.return_value = mock_sync_client

        # Mock SearchIndex
        with patch("phillm.vector.redis_vector_store.SearchIndex") as mock_search_index:
            mock_index = MagicMock()
            mock_search_index.return_value = mock_index

            store = RedisVectorStore()
            store.redis_client = mock_async_client
            store.sync_redis_client = mock_sync_client
            store.index = mock_index

            return store


@pytest.mark.asyncio
async def test_store_message_success(vector_store):
    embedding = [0.1, 0.2, 0.3]

    # Mock the async methods
    vector_store._ensure_redis_connection = AsyncMock()
    vector_store.redis_client.hset = AsyncMock()

    result = await vector_store.store_message(
        user_id="U123",
        channel_id="C123",
        message="Hello world",
        embedding=embedding,
        timestamp="1234567890.123",
    )

    assert result is not None
    assert isinstance(result, str)
    # Should store as hash with msg: prefix
    vector_store.redis_client.hset.assert_called_once()

    # Check the call arguments
    call_args = vector_store.redis_client.hset.call_args
    doc_key = call_args[0][0]
    assert doc_key.startswith("msg:")

    # Check the stored data structure
    stored_data = call_args.kwargs["mapping"]
    assert stored_data["user_id"] == "U123"
    assert stored_data["channel_id"] == "C123"
    assert stored_data["message_text"] == "Hello world"
    assert stored_data["timestamp"] == 1234567890.123
    # Embedding should be stored as float32 bytes
    assert isinstance(stored_data["embedding"], bytes)


@pytest.mark.asyncio
async def test_find_similar_messages_with_text_query(vector_store):
    # Mock embedding service
    with patch("phillm.ai.embeddings.EmbeddingService") as mock_embedding_service:
        mock_service = AsyncMock()
        mock_service.create_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding_service.return_value = mock_service

        # Mock async methods
        vector_store._ensure_redis_connection = AsyncMock()

        # Mock RedisVL SearchIndex query results
        mock_results = [
            {
                "message_text": "Hello world",
                "channel_id": "C123",
                "timestamp": "1234567890.123",
                "message_id": "msg1",
                "vector_distance": 0.3,  # Lower distance = higher similarity
            },
            {
                "message_text": "Goodbye world",
                "channel_id": "C123",
                "timestamp": "1234567891.123",
                "message_id": "msg2",
                "vector_distance": 0.7,  # Higher distance = lower similarity
            },
        ]

        vector_store.index.query.return_value = mock_results

        results = await vector_store.find_similar_messages(
            "test query", user_id="U123", limit=5, threshold=0.2
        )

        assert len(results) <= 5
        assert len(results) == 2  # Both results above threshold

        # Check that RedisVL query was called
        vector_store.index.query.assert_called_once()

        # Check result format
        assert results[0]["message"] == "Hello world"
        assert (
            results[0]["similarity"] > results[1]["similarity"]
        )  # First should be more similar


@pytest.mark.asyncio
async def test_get_user_message_count(vector_store):
    # Mock async methods
    vector_store._ensure_redis_connection = AsyncMock()
    
    # Mock FT.SEARCH response - first element is count
    vector_store.redis_client.execute_command = AsyncMock(return_value=[42])

    count = await vector_store.get_user_message_count("U123")

    assert count == 42
    vector_store.redis_client.execute_command.assert_called_once_with(
        "FT.SEARCH", "phillm_messages", "@user_id:{U123}", "LIMIT", "0", "0"
    )


@pytest.mark.asyncio
async def test_get_recent_messages(vector_store):
    # Mock async methods
    vector_store._ensure_redis_connection = AsyncMock()
    
    # Mock FT.SEARCH response with SORTBY timestamp DESC
    # Format: [count, doc_id1, [field1, value1, field2, value2], doc_id2, [field3, value3, field4, value4]]
    vector_store.redis_client.execute_command = AsyncMock(return_value=[
        2,  # count
        "msg:newer_id",
        [
            "message_text",
            "Goodbye world",
            "channel_id",
            "C123",
            "timestamp",
            "1234567891.123",
            "message_id",
            "newer_id",
        ],
        "msg:older_id",
        [
            "message_text",
            "Hello world",
            "channel_id",
            "C123",
            "timestamp",
            "1234567890.123",
            "message_id",
            "older_id",
        ],
    ])

    messages = await vector_store.get_recent_messages("U123", limit=10)

    assert len(messages) == 2
    assert messages[0]["timestamp"] > messages[1]["timestamp"]  # Newest first
    assert messages[0]["message"] == "Goodbye world"
    assert messages[1]["message"] == "Hello world"

    vector_store.redis_client.execute_command.assert_called_once_with(
        "FT.SEARCH",
        "phillm_messages",
        "@user_id:{U123}",
        "SORTBY",
        "timestamp",
        "DESC",
        "LIMIT",
        "0",
        "10",
        "RETURN",
        "4",
        "message_text",
        "channel_id",
        "timestamp",
        "message_id",
    )


@pytest.mark.asyncio
async def test_delete_user_messages(vector_store):
    # Mock FT.SEARCH response for finding user documents
    vector_store.redis_client.execute_command.return_value = [
        3,  # count
        "msg:id1",
        "msg:id2",
        "msg:id3",  # document IDs
    ]

    deleted_count = await vector_store.delete_user_messages("U123")

    assert deleted_count == 3
    # Should call FT.SEARCH to find documents, then delete each one
    vector_store.redis_client.execute_command.assert_called_once_with(
        "FT.SEARCH", "phillm_messages", "@user_id:{U123}", "RETURN", "0"
    )
    assert vector_store.redis_client.delete.call_count == 3


@pytest.mark.asyncio
async def test_message_exists(vector_store):
    # Test message existence check using direct document lookup
    vector_store.redis_client.exists.return_value = True

    exists = await vector_store.message_exists("U123", "1234567890.123", "C123")

    assert exists is True
    # Should check for document with generated message ID
    vector_store.redis_client.exists.assert_called_once()
    call_args = vector_store.redis_client.exists.call_args[0][0]
    assert call_args.startswith("msg:")


@pytest.mark.asyncio
async def test_message_not_exists(vector_store):
    vector_store.redis_client.exists.return_value = False

    exists = await vector_store.message_exists("U123", "1234567890.123", "C123")

    assert exists is False


def test_generate_message_id(vector_store):
    message_id = vector_store._generate_message_id("U123", "C123", "1234567890.123")

    assert isinstance(message_id, str)
    assert len(message_id) == 32  # MD5 hash length


@pytest.mark.asyncio
async def test_find_similar_messages_with_embedding_vector(vector_store):
    # Test with direct embedding vector instead of text query
    query_embedding = [0.1, 0.2, 0.3]

    # Mock async methods
    vector_store._ensure_redis_connection = AsyncMock()

    # Mock RedisVL SearchIndex query results
    mock_results = [
        {
            "message_text": "Similar message",
            "channel_id": "C123",
            "timestamp": "1234567890.123",
            "message_id": "msg1",
            "vector_distance": 0.2,
        }
    ]

    vector_store.index.query.return_value = mock_results

    results = await vector_store.find_similar_messages(
        query_embedding, user_id="U123", limit=3, threshold=0.5
    )

    assert len(results) == 1
    assert results[0]["message"] == "Similar message"
    assert results[0]["similarity"] == 0.8  # 1.0 - 0.2 distance

    # Should not call embedding service since vector was provided directly
    vector_store.index.query.assert_called_once()


@pytest.mark.asyncio
async def test_get_oldest_stored_message(vector_store):
    # Mock FT.SEARCH response for oldest message
    vector_store.redis_client.execute_command.return_value = [
        1,  # count
        "msg:oldest_id",
        [
            "message_text",
            "First message",
            "channel_id",
            "C123",
            "timestamp",
            "1234567890.123",
            "message_id",
            "oldest_id",
        ],
    ]

    oldest = await vector_store.get_oldest_stored_message("U123", "C123")

    assert oldest is not None
    assert oldest["message"] == "First message"
    assert oldest["timestamp"] == 1234567890.123

    vector_store.redis_client.execute_command.assert_called_once_with(
        "FT.SEARCH",
        "phillm_messages",
        "@user_id:{U123} @channel_id:{C123}",
        "SORTBY",
        "timestamp",
        "ASC",
        "LIMIT",
        "0",
        "1",
        "RETURN",
        "4",
        "message_text",
        "channel_id",
        "timestamp",
        "message_id",
    )


@pytest.mark.asyncio
async def test_scrape_state_management(vector_store):
    # Test save scrape state
    await vector_store.save_scrape_state(
        "C123", "cursor123", "1234567890.123", "1234567800.123"
    )

    vector_store.redis_client.hset.assert_called()
    call_args = vector_store.redis_client.hset.call_args
    assert call_args[0][0] == "scrape_state:C123"

    # Test get scrape state
    vector_store.redis_client.hgetall.return_value = {
        "cursor": "cursor123",
        "last_message_ts": "1234567890.123",
        "oldest_processed": "1234567800.123",
    }

    state = await vector_store.get_scrape_state("C123")
    assert state["cursor"] == "cursor123"
    assert state["last_message_ts"] == "1234567890.123"

    # Test clear scrape state
    await vector_store.clear_scrape_state("C123")
    vector_store.redis_client.delete.assert_called_with("scrape_state:C123")
