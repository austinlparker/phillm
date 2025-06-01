import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from phillm.ai.embeddings import EmbeddingService


@pytest.fixture
def embedding_service():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        return EmbeddingService()


@pytest.mark.asyncio
async def test_create_embedding_success(embedding_service):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch.object(embedding_service.client, "embeddings") as mock_embeddings:
        mock_embeddings.create = AsyncMock(return_value=mock_response)

        result = await embedding_service.create_embedding("test message")

        assert result == [0.1, 0.2, 0.3]
        mock_embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input="test message"
        )


@pytest.mark.asyncio
async def test_create_embedding_strips_whitespace(embedding_service):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch.object(embedding_service.client, "embeddings") as mock_embeddings:
        mock_embeddings.create = AsyncMock(return_value=mock_response)

        await embedding_service.create_embedding("  test message  ")

        mock_embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input="test message"
        )


@pytest.mark.asyncio
async def test_create_embeddings_batch_success(embedding_service):
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]

    with patch.object(embedding_service.client, "embeddings") as mock_embeddings:
        mock_embeddings.create = AsyncMock(return_value=mock_response)

        texts = ["message 1", "message 2"]
        result = await embedding_service.create_embeddings_batch(texts)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input=["message 1", "message 2"]
        )


@pytest.mark.asyncio
async def test_create_embedding_error_handling(embedding_service):
    with patch.object(embedding_service.client, "embeddings") as mock_embeddings:
        mock_embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            await embedding_service.create_embedding("test message")
