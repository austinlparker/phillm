import os
import json
import hashlib
import time
from typing import List, Dict
from datetime import datetime
import redis.asyncio as redis
import numpy as np
from loguru import logger
from phillm.telemetry import get_tracer, telemetry


class VectorStore:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")

        self.redis_client = redis.from_url(
            self.redis_url, password=self.redis_password, decode_responses=True
        )

        self.vector_dim = 3072  # text-embedding-3-large dimension

    async def store_message(
        self,
        user_id: str,
        channel_id: str,
        message: str,
        embedding: List[float],
        timestamp: str,
    ) -> str:
        try:
            message_id = self._generate_message_id(user_id, channel_id, timestamp)

            message_data = {
                "user_id": user_id,
                "channel_id": channel_id,
                "message": message,
                "timestamp": timestamp,
                "embedding": json.dumps(embedding),
            }

            # Store message data
            await self.redis_client.hset(f"message:{message_id}", mapping=message_data)

            # Add to user's message index
            await self.redis_client.sadd(f"user_messages:{user_id}", message_id)

            # Add to channel index
            await self.redis_client.sadd(f"channel_messages:{channel_id}", message_id)

            logger.debug(f"Stored message {message_id} for user {user_id}")
            return message_id

        except Exception as e:
            logger.error(f"Error storing message: {e}")
            raise

    async def find_similar_messages(
        self,
        query_embedding_or_text: str,
        user_id: str,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict]:
        tracer = get_tracer()
        start_time = time.time()

        try:
            with tracer.start_as_current_span("find_similar_messages") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("limit", limit)
                span.set_attribute("threshold", threshold)

                if isinstance(query_embedding_or_text, str):
                    span.set_attribute("query.text", query_embedding_or_text[:100])
                    span.set_attribute("query.length", len(query_embedding_or_text))

                    from phillm.ai.embeddings import EmbeddingService

                    embedding_service = EmbeddingService()
                    query_embedding = await embedding_service.create_embedding(
                        query_embedding_or_text
                    )
                else:
                    query_embedding = query_embedding_or_text

                # Get all message IDs for the user
                message_ids = await self.redis_client.smembers(
                    f"user_messages:{user_id}"
                )
                span.set_attribute("total_messages", len(message_ids))

                if not message_ids:
                    return []

                similarities = []
                max_similarity = 0.0

                for message_id in message_ids:
                    message_data = await self.redis_client.hgetall(
                        f"message:{message_id}"
                    )

                    if not message_data or "embedding" not in message_data:
                        continue

                    stored_embedding = json.loads(message_data["embedding"])
                    similarity = self._cosine_similarity(
                        query_embedding, stored_embedding
                    )
                    max_similarity = max(max_similarity, similarity)

                    if similarity >= threshold:
                        similarities.append(
                            {
                                "message_id": message_id,
                                "message": message_data["message"],
                                "channel_id": message_data["channel_id"],
                                "timestamp": message_data["timestamp"],
                                "similarity": similarity,
                            }
                        )

                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                results = similarities[:limit]

                duration = time.time() - start_time
                span.set_attribute("results.count", len(results))
                span.set_attribute("max_similarity", max_similarity)
                span.set_attribute("duration_seconds", duration)

                # Record metrics
                telemetry.record_similarity_search(
                    len(query_embedding_or_text)
                    if isinstance(query_embedding_or_text, str)
                    else 0,
                    len(results),
                    threshold,
                    max_similarity,
                )

                return results

        except Exception as e:
            logger.error(f"Error finding similar messages: {e}")
            raise

    async def get_user_message_count(self, user_id: str) -> int:
        try:
            count = await self.redis_client.scard(f"user_messages:{user_id}")
            return count
        except Exception as e:
            logger.error(f"Error getting message count for user {user_id}: {e}")
            return 0

    async def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Dict]:
        try:
            message_ids = await self.redis_client.smembers(f"user_messages:{user_id}")

            messages = []
            for message_id in message_ids:
                message_data = await self.redis_client.hgetall(f"message:{message_id}")
                if message_data:
                    messages.append(
                        {
                            "message_id": message_id,
                            "message": message_data["message"],
                            "channel_id": message_data["channel_id"],
                            "timestamp": float(message_data["timestamp"]),
                        }
                    )

            # Sort by timestamp (newest first) and limit
            messages.sort(key=lambda x: x["timestamp"], reverse=True)
            return messages[:limit]

        except Exception as e:
            logger.error(f"Error getting recent messages for user {user_id}: {e}")
            return []

    def _generate_message_id(
        self, user_id: str, channel_id: str, timestamp: str
    ) -> str:
        content = f"{user_id}:{channel_id}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    async def delete_user_messages(self, user_id: str) -> int:
        try:
            message_ids = await self.redis_client.smembers(f"user_messages:{user_id}")

            if not message_ids:
                return 0

            # Delete all message data
            for message_id in message_ids:
                await self.redis_client.delete(f"message:{message_id}")

            # Delete user index
            await self.redis_client.delete(f"user_messages:{user_id}")

            logger.info(f"Deleted {len(message_ids)} messages for user {user_id}")
            return len(message_ids)

        except Exception as e:
            logger.error(f"Error deleting messages for user {user_id}: {e}")
            raise

    async def close(self):
        await self.redis_client.close()

    # Scraping state management
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
            "oldest_processed": oldest_processed
            or "",  # Track oldest message we've processed
            "updated_at": str(datetime.now().timestamp()),
        }
        await self.redis_client.hset(state_key, mapping=state_data)
        logger.debug(
            f"Saved scrape state for channel {channel_id}: cursor={cursor}, oldest={oldest_processed}"
        )

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
        """Clear the scraping state for a channel (when scraping is complete)"""
        state_key = f"scrape_state:{channel_id}"
        await self.redis_client.delete(state_key)
        logger.info(f"Cleared scrape state for channel {channel_id}")

    async def message_exists(self, user_id: str, timestamp: str) -> bool:
        """Check if a message already exists in the store"""
        # Check if any message with this timestamp exists for this user
        message_ids = await self.redis_client.smembers(f"user_messages:{user_id}")

        for existing_id in message_ids:
            message_data = await self.redis_client.hgetall(f"message:{existing_id}")
            if message_data and message_data.get("timestamp") == timestamp:
                return True
        return False

    async def get_oldest_stored_message(
        self, user_id: str, channel_id: str = None
    ) -> dict:
        """Get the oldest stored message for a user (optionally in a specific channel)"""
        try:
            message_ids = await self.redis_client.smembers(f"user_messages:{user_id}")

            if not message_ids:
                return None

            oldest_message = None
            oldest_timestamp = float("inf")

            for message_id in message_ids:
                message_data = await self.redis_client.hgetall(f"message:{message_id}")
                if not message_data:
                    continue

                # Filter by channel if specified
                if channel_id and message_data.get("channel_id") != channel_id:
                    continue

                timestamp = float(message_data.get("timestamp", 0))
                if timestamp < oldest_timestamp:
                    oldest_timestamp = timestamp
                    oldest_message = {
                        "message_id": message_id,
                        "message": message_data["message"],
                        "channel_id": message_data["channel_id"],
                        "timestamp": timestamp,
                    }

            return oldest_message

        except Exception as e:
            logger.error(f"Error getting oldest stored message for user {user_id}: {e}")
            return None
