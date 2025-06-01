import os
import hashlib
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import redis.asyncio as redis
import numpy as np
from loguru import logger
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from phillm.telemetry import telemetry


class RedisVectorStore:
    """Modern Redis vector store using RedisVL for semantic search"""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Connection configuration
        self.connection_timeout = 10  # seconds
        self.retry_delay = 5  # seconds between health check attempts
        
        # Initialize clients as None, will be created on demand
        self.redis_client = None
        self.sync_redis_client = None
        self._redis_healthy = False

        # Vector dimensions for text-embedding-3-large
        self.vector_dim = 3072

        # Index configuration
        self.index_name = "phillm_messages"
        self.index = None

        # Schema for vector search - separate metadata from vectors
        self.schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": self.index_name,
                    "prefix": "msg:",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "user_id", "type": "tag"},
                    {"name": "channel_id", "type": "tag"},
                    {"name": "message_text", "type": "text"},
                    {"name": "timestamp", "type": "numeric"},
                    {"name": "message_id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": self.vector_dim,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )

    async def _ensure_redis_connection(self):
        """Ensure Redis connection is established and healthy"""
        if self._redis_healthy and self.redis_client and self.sync_redis_client:
            try:
                # Quick health check
                await self.redis_client.ping()
                return
            except Exception:
                # Connection lost, need to reconnect
                self._redis_healthy = False
                
        logger.info("Connecting to Redis...")
        
        # Initialize async Redis client for regular operations
        self.redis_client = redis.from_url(
            self.redis_url, 
            password=self.redis_password, 
            decode_responses=True,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.connection_timeout
        )

        # Initialize sync Redis client for RedisVL operations
        import redis as sync_redis
        self.sync_redis_client = sync_redis.Redis.from_url(
            self.redis_url, 
            password=self.redis_password, 
            decode_responses=True,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.connection_timeout
        )
        
        # Test the connections
        await self.redis_client.ping()
        self.sync_redis_client.ping()
        
        self._redis_healthy = True
        logger.info("✅ Redis connection established successfully")
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy and available"""
        try:
            await self._ensure_redis_connection()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            self._redis_healthy = False
            return False

    async def initialize_index(self):
        """Initialize the vector search index"""
        # Ensure Redis connection first
        await self._ensure_redis_connection()
        
        # Create SearchIndex instance using the sync redis client
        # RedisVL works with standard redis-py clients
        self.index = SearchIndex(
            schema=self.schema, redis_client=self.sync_redis_client
        )

        # Check if index exists, create if not (using sync operations)
        if not self._index_exists_sync():
            logger.info(f"Creating vector index: {self.index_name}")
            self.index.create(overwrite=False)
            logger.info("✅ Vector index created successfully")
        else:
            logger.info(f"Vector index {self.index_name} already exists")
            # Connect to existing index
            self.index.connect()

    def _index_exists_sync(self) -> bool:
        """Check if the vector index exists (sync version for initialization)"""
        try:
            # Check if index exists by looking for it in the index list
            indices = self.sync_redis_client.execute_command("FT._LIST")
            # indices is a list of byte strings, so check for both string and bytes
            return (
                (self.index_name in indices) or (self.index_name.encode() in indices)
                if indices
                else False
            )
        except Exception:
            return False

    async def _index_exists(self) -> bool:
        """Check if the vector index exists"""
        try:
            await self.redis_client.ft(self.index_name).info()
            return True
        except Exception:
            return False

    async def store_message(
        self,
        user_id: str,
        channel_id: str,
        message: str,
        embedding: List[float],
        timestamp: str,
    ) -> str:
        """Store a message with its embedding in the vector database"""
        await self._ensure_redis_connection()
        
        message_id = self._generate_message_id(user_id, channel_id, timestamp)

        # Prepare data for storage with cleaner separation
        doc_key = f"msg:{message_id}"
        message_data = {
            "user_id": user_id,
            "channel_id": channel_id,
            "message_text": message,
            "timestamp": float(timestamp),
            "message_id": message_id,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        }

        # Store in Redis with new vector index prefix
        await self.redis_client.hset(doc_key, mapping=message_data)

        # Record metrics
        telemetry.record_message_scraped(channel_id, user_id)

        logger.debug(f"Stored vector message {message_id} for user {user_id}")
        return message_id

    async def find_similar_messages(
        self,
        query_embedding_or_text: str,
        user_id: str,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict]:
        """Find similar messages using Redis vector search"""
        await self._ensure_redis_connection()
        
        # Ensure index is initialized
        if not self.index:
            await self.initialize_index()

        # Get query embedding if text provided
        if isinstance(query_embedding_or_text, str):
            from phillm.ai.embeddings import EmbeddingService

            embedding_service = EmbeddingService()
            query_embedding = await embedding_service.create_embedding(
                query_embedding_or_text
            )
        else:
            query_embedding = query_embedding_or_text

        # Ensure consistent float32 format (same as storage)
        query_vector = np.array(query_embedding, dtype=np.float32)

        # Create vector query with user filter
        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=[
                "user_id",
                "channel_id",
                "message_text",
                "timestamp",
                "message_id",
            ],
            num_results=limit * 2,  # Get more results to filter
            filter_expression=Tag("user_id") == user_id,
        )

        # Execute search
        results = self.index.query(vector_query)

        # Process results and apply threshold
        similar_messages = []
        max_similarity = 0.0

        for result in results:
            # Redis vector search returns distance, convert to similarity
            distance = float(result.get("vector_distance", 1.0))
            similarity = 1.0 - distance  # Cosine distance to similarity
            max_similarity = max(max_similarity, similarity)

            if similarity >= threshold:
                similar_messages.append(
                    {
                        "message_id": result.get(
                            "message_id", result.get("id", "").replace("msg:", "")
                        ),
                        "message": result.get("message_text", ""),
                        "channel_id": result.get("channel_id", ""),
                        "timestamp": result.get("timestamp", ""),
                        "similarity": similarity,
                    }
                )

        # Sort by similarity and limit results
        similar_messages.sort(key=lambda x: x["similarity"], reverse=True)
        final_results = similar_messages[:limit]

        # Record metrics
        telemetry.record_similarity_search(
            len(query_embedding_or_text)
            if isinstance(query_embedding_or_text, str)
            else 0,
            len(final_results),
            threshold,
            max_similarity,
        )

        return final_results

    async def get_user_message_count(self, user_id: str) -> int:
        """Get total message count for a user using vector index"""
        await self._ensure_redis_connection()
        
        # Use FT.SEARCH to count documents with user_id tag
        result = await self.redis_client.execute_command(
            "FT.SEARCH",
            self.index_name,
            f"@user_id:{{{user_id}}}",
            "LIMIT",
            "0",
            "0",
        )
        # First element is the count
        return int(result[0]) if result else 0

    async def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get recent messages for a user using vector index"""
        try:
            # Use FT.SEARCH to get user messages sorted by timestamp (newest first)
            result = await self.redis_client.execute_command(
                "FT.SEARCH",
                self.index_name,
                f"@user_id:{{{user_id}}}",
                "SORTBY",
                "timestamp",
                "DESC",
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "4",
                "message_text",
                "channel_id",
                "timestamp",
                "message_id",
            )

            messages = []
            if len(result) > 1:
                # Skip count (first element) and process results in pairs
                for i in range(1, len(result), 2):
                    doc_id = result[i]
                    fields = result[i + 1] if i + 1 < len(result) else []

                    # Extract message ID from document key (msg:id)
                    message_id = (
                        doc_id.replace("msg:", "")
                        if doc_id.startswith("msg:")
                        else doc_id
                    )

                    # Parse field-value pairs
                    field_data = {}
                    for j in range(0, len(fields), 2):
                        if j + 1 < len(fields):
                            field_data[fields[j]] = fields[j + 1]

                    messages.append(
                        {
                            "message_id": field_data.get("message_id", message_id),
                            "message": field_data.get("message_text", ""),
                            "channel_id": field_data.get("channel_id", ""),
                            "timestamp": float(field_data.get("timestamp", 0)),
                        }
                    )

            return messages

        except Exception as e:
            logger.error(f"Error getting recent messages for user {user_id}: {e}")
            return []

    async def get_oldest_stored_message(
        self, user_id: str, channel_id: str = None
    ) -> Optional[Dict]:
        """Get the oldest stored message for a user (optionally in a specific channel)"""
        try:
            # Build search query
            query = f"@user_id:{{{user_id}}}"
            if channel_id:
                query += f" @channel_id:{{{channel_id}}}"

            # Search for oldest message (sorted by timestamp ascending)
            result = await self.redis_client.execute_command(
                "FT.SEARCH",
                self.index_name,
                query,
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

            if len(result) > 1:
                doc_id = result[1]
                fields = result[2] if len(result) > 2 else []

                # Extract message ID from document key
                message_id = (
                    doc_id.replace("msg:", "") if doc_id.startswith("msg:") else doc_id
                )

                # Parse field-value pairs
                field_data = {}
                for j in range(0, len(fields), 2):
                    if j + 1 < len(fields):
                        field_data[fields[j]] = fields[j + 1]

                return {
                    "message_id": field_data.get("message_id", message_id),
                    "message": field_data.get("message_text", ""),
                    "channel_id": field_data.get("channel_id", ""),
                    "timestamp": float(field_data.get("timestamp", 0)),
                }

            return None

        except Exception as e:
            logger.error(f"Error getting oldest stored message for user {user_id}: {e}")
            return None

    async def message_exists(
        self, user_id: str, timestamp: str, channel_id: str = None
    ) -> bool:
        """Check if a message already exists in the store by generating the expected message ID"""
        try:
            # Always use direct document lookup when possible
            if channel_id:
                # Use the same message ID generation logic as store_message
                message_id = self._generate_message_id(user_id, channel_id, timestamp)

                # Check if the document exists directly with new prefix
                doc_key = f"msg:{message_id}"
                exists = await self.redis_client.exists(doc_key)
                return bool(exists)

            # If no channel_id provided, just return False for now
            # This avoids complex queries that hit binary data
            return False

        except Exception as e:
            logger.error(f"Error checking message existence: {e}")
            return False

    async def delete_user_messages(self, user_id: str) -> int:
        """Delete all messages for a user using vector index"""
        try:
            # First, get all document IDs for this user
            result = await self.redis_client.execute_command(
                "FT.SEARCH",
                self.index_name,
                f"@user_id:{{{user_id}}}",
                "RETURN",
                "0",  # Return only document IDs, no fields
            )

            if len(result) <= 1:
                return 0

            deleted_count = 0
            # Skip count (first element) and delete each document
            for i in range(1, len(result)):
                doc_id = result[i]
                await self.redis_client.delete(doc_id)
                deleted_count += 1

            logger.info(f"Deleted {deleted_count} messages for user {user_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting messages for user {user_id}: {e}")
            raise

    # Scraping state management (unchanged from original)
    async def save_scrape_state(
        self,
        channel_id: str,
        cursor: str = None,
        last_message_ts: str = None,
        oldest_processed: str = None,
    ):
        """Save the current scraping state for a channel"""
        state_key = f"scrape_state:{channel_id}"
        state_data = {
            "cursor": cursor or "",
            "last_message_ts": last_message_ts or "",
            "oldest_processed": oldest_processed or "",
            "updated_at": str(datetime.now().timestamp()),
        }
        await self.redis_client.hset(state_key, mapping=state_data)
        logger.debug(f"Saved scrape state for channel {channel_id}: cursor={cursor}")

    async def get_scrape_state(self, channel_id: str) -> dict:
        """Get the saved scraping state for a channel"""
        state_key = f"scrape_state:{channel_id}"
        state_data = await self.redis_client.hgetall(state_key)

        if state_data:
            return {
                "cursor": state_data.get("cursor") or None,
                "last_message_ts": state_data.get("last_message_ts") or None,
                "oldest_processed": state_data.get("oldest_processed") or None,
                "updated_at": state_data.get("updated_at"),
            }
        return {
            "cursor": None,
            "last_message_ts": None,
            "oldest_processed": None,
            "updated_at": None,
        }

    async def clear_scrape_state(self, channel_id: str):
        """Clear the scraping state for a channel"""
        state_key = f"scrape_state:{channel_id}"
        await self.redis_client.delete(state_key)
        logger.info(f"Cleared scrape state for channel {channel_id}")

    def _generate_message_id(
        self, user_id: str, channel_id: str, timestamp: str
    ) -> str:
        """Generate a unique message ID"""
        content = f"{user_id}:{channel_id}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()

    async def close(self):
        """Close Redis connection"""
        await self.redis_client.close()
