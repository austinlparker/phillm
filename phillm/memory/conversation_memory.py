import os
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from loguru import logger
from phillm.telemetry import get_tracer


class MemoryType(Enum):
    """Types of memories the bot can store"""

    DM_CONVERSATION = "dm_conversation"
    CHANNEL_INTERACTION = "channel_interaction"
    USER_PREFERENCE = "user_preference"
    CONTEXT_SNIPPET = "context_snippet"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class MemoryImportance(Enum):
    """Importance levels for memory retention"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Memory:
    """A single memory entry"""

    memory_id: str
    user_id: str
    memory_type: MemoryType
    content: str
    context: Dict[str, Any]
    importance: MemoryImportance
    timestamp: float
    embedding: Optional[List[float]] = None
    decay_factor: float = 1.0  # Memories decay over time
    access_count: int = 0
    last_accessed: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage"""
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["importance"] = self.importance.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create memory from dictionary"""
        data["memory_type"] = MemoryType(data["memory_type"])
        data["importance"] = MemoryImportance(data["importance"])
        return cls(**data)

    def calculate_relevance_score(
        self, current_time: float, base_similarity: float = 0.0
    ) -> float:
        """Calculate how relevant this memory is right now"""
        # Time decay: memories become less relevant over time
        time_diff = current_time - self.timestamp
        days_old = time_diff / 86400

        # Decay based on importance level
        if self.importance == MemoryImportance.CRITICAL:
            time_decay = max(0.7, 1.0 - (days_old * 0.01))  # Very slow decay
        elif self.importance == MemoryImportance.HIGH:
            time_decay = max(0.5, 1.0 - (days_old * 0.02))  # Slow decay
        elif self.importance == MemoryImportance.MEDIUM:
            time_decay = max(0.3, 1.0 - (days_old * 0.05))  # Medium decay
        else:
            time_decay = max(0.1, 1.0 - (days_old * 0.1))  # Fast decay

        # Access frequency boost: frequently accessed memories stay relevant
        access_boost = min(0.3, self.access_count * 0.05)

        # Recency boost: recently accessed memories get a boost
        recency_boost = 0.0
        if self.last_accessed:
            hours_since_access = (current_time - self.last_accessed) / 3600
            recency_boost = max(0.0, 0.2 - (hours_since_access * 0.01))

        # Combine all factors
        relevance = (
            (base_similarity * 0.4)
            + (time_decay * 0.4)
            + (access_boost * 0.1)
            + (recency_boost * 0.1)
        )
        return min(1.0, relevance)


class ConversationMemory:
    """Manages conversation memory for users"""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")

        self.redis_client = redis.from_url(
            self.redis_url, password=self.redis_password, decode_responses=True
        )

        # Memory configuration
        self.max_memories_per_user = int(os.getenv("MAX_MEMORIES_PER_USER", "1000"))
        self.memory_retention_days = int(os.getenv("MEMORY_RETENTION_DAYS", "30"))
        self.embedding_service = None  # Will be injected

    def set_embedding_service(self, embedding_service):
        """Inject embedding service for memory vectorization"""
        self.embedding_service = embedding_service

    async def store_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        context: Dict[str, Any],
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        pre_computed_embedding: list = None,
    ) -> str:
        """Store a new memory"""
        tracer = get_tracer()

        try:
            # Generate memory ID
            memory_id = f"{user_id}:{memory_type.value}:{int(time.time() * 1000)}"

            # Use pre-computed embedding or create new one for semantic search
            embedding = pre_computed_embedding
            if not embedding and self.embedding_service and content.strip():
                try:
                    embedding = await self.embedding_service.create_embedding(content)
                except Exception as e:
                    logger.warning(f"Failed to create embedding for memory: {e}")

            # Create memory object
            memory = Memory(
                memory_id=memory_id,
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                context=context,
                importance=importance,
                timestamp=time.time(),
                embedding=embedding,
            )

            with tracer.start_as_current_span("store_memory") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("memory_type", memory_type.value)
                span.set_attribute("content_length", len(content))
                span.set_attribute("importance", importance.value)

                # Store memory data
                memory_data = memory.to_dict()
                # Convert complex objects to JSON strings for Redis storage
                if embedding:
                    memory_data["embedding"] = json.dumps(embedding)
                else:
                    memory_data["embedding"] = ""

                # Serialize context dict to JSON string
                memory_data["context"] = json.dumps(memory_data["context"])

                # Filter out None values and convert all values to strings for Redis
                redis_data = {}
                for key, value in memory_data.items():
                    if value is not None:
                        redis_data[key] = str(value)
                    else:
                        redis_data[key] = ""

                await self.redis_client.hset(f"memory:{memory_id}", mapping=redis_data)

                # Add to user's memory index
                await self.redis_client.sadd(f"user_memories:{user_id}", memory_id)

                # Add to type-specific index
                await self.redis_client.sadd(
                    f"memories_by_type:{memory_type.value}", memory_id
                )

                # Clean up old memories if needed
                await self._cleanup_old_memories(user_id)

                logger.debug(f"Stored memory {memory_id} for user {user_id}")
                return memory_id

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    async def recall_memories(
        self,
        user_id: str,
        query: str = None,
        memory_types: List[MemoryType] = None,
        limit: int = 10,
        min_relevance: float = 0.3,
    ) -> List[Memory]:
        """Recall relevant memories for a user"""
        tracer = get_tracer()
        current_time = time.time()

        try:
            with tracer.start_as_current_span("recall_memories") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("query_length", len(query) if query else 0)
                span.set_attribute("limit", limit)

                memories = await self._get_user_memories(user_id, memory_types)

                if not memories:
                    return []

                # If we have a query, compute semantic similarity
                query_embedding = None
                if query and self.embedding_service:
                    try:
                        query_embedding = await self.embedding_service.create_embedding(
                            query
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create query embedding: {e}")

                # Score and filter memories
                relevant_memories = []
                for memory in memories:
                    # Calculate base similarity
                    base_similarity = 0.0
                    if query_embedding and memory.embedding:
                        base_similarity = self._cosine_similarity(
                            query_embedding, memory.embedding
                        )
                    elif query and query.lower() in memory.content.lower():
                        base_similarity = 0.7  # Text match fallback

                    # Calculate overall relevance
                    relevance = memory.calculate_relevance_score(
                        current_time, base_similarity
                    )

                    if relevance >= min_relevance:
                        relevant_memories.append((memory, relevance))
                        # Update access count
                        memory.access_count += 1
                        memory.last_accessed = current_time
                        await self._update_memory_access(memory)

                # Sort by relevance and return top results
                relevant_memories.sort(key=lambda x: x[1], reverse=True)
                result = [mem for mem, score in relevant_memories[:limit]]

                span.set_attribute("memories_found", len(result))
                span.set_attribute("memories_considered", len(memories))

                return result

        except Exception as e:
            logger.error(f"Error recalling memories: {e}")
            return []

    async def get_conversation_context(self, user_id: str, limit: int = 5) -> str:
        """Get recent conversation context for a user"""
        try:
            # Get recent DM and interaction memories
            recent_memories = await self.recall_memories(
                user_id,
                memory_types=[
                    MemoryType.DM_CONVERSATION,
                    MemoryType.CHANNEL_INTERACTION,
                ],
                limit=limit,
                min_relevance=0.2,
            )

            logger.debug(
                f"💬 Found {len(recent_memories)} memories for user {user_id} (limit: {limit})"
            )

            if not recent_memories:
                logger.debug(f"💬 No conversation context found for user {user_id}")
                return ""

            # Build context string
            context_parts = []
            for i, memory in enumerate(recent_memories):
                time_str = datetime.fromtimestamp(memory.timestamp).strftime(
                    "%Y-%m-%d %H:%M"
                )
                context_parts.append(f"[{time_str}] {memory.content}")
                logger.debug(f"💬 Memory {i + 1}: {memory.content[:80]}...")

            context = "\n".join(context_parts)
            logger.debug(f"💬 Built conversation context: {len(context)} chars")
            return context

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return ""

    async def store_dm_interaction(
        self,
        user_id: str,
        user_message: str,
        bot_response: str,
        channel_id: str,
        query_embedding: list = None,
    ):
        """Store a DM interaction"""
        context = {
            "channel_id": channel_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "interaction_type": "dm",
        }

        content = f"User: {user_message}\nBot: {bot_response}"

        logger.debug(
            f"💾 Storing DM interaction for user {user_id}: {user_message[:50]}..."
        )

        memory_id = await self.store_memory(
            user_id=user_id,
            memory_type=MemoryType.DM_CONVERSATION,
            content=content,
            context=context,
            importance=MemoryImportance.HIGH,  # DMs are important
            pre_computed_embedding=query_embedding,
        )

        logger.debug(f"💾 Stored DM memory {memory_id} for user {user_id}")

    async def store_channel_interaction(
        self, user_id: str, message: str, channel_id: str, channel_name: str = None
    ):
        """Store a channel interaction"""
        context = {
            "channel_id": channel_id,
            "channel_name": channel_name or channel_id,
            "interaction_type": "channel_message",
        }

        await self.store_memory(
            user_id=user_id,
            memory_type=MemoryType.CHANNEL_INTERACTION,
            content=message,
            context=context,
            importance=MemoryImportance.MEDIUM,
        )

    async def store_user_preference(self, user_id: str, preference: str, value: str):
        """Store a user preference or behavior pattern"""
        context = {"preference_type": preference, "value": value}

        content = f"{preference}: {value}"

        await self.store_memory(
            user_id=user_id,
            memory_type=MemoryType.USER_PREFERENCE,
            content=content,
            context=context,
            importance=MemoryImportance.HIGH,
        )

    async def _get_user_memories(
        self, user_id: str, memory_types: List[MemoryType] = None
    ) -> List[Memory]:
        """Get all memories for a user, optionally filtered by type"""
        try:
            memory_ids = await self.redis_client.smembers(f"user_memories:{user_id}")

            memories = []
            for memory_id in memory_ids:
                memory_data = await self.redis_client.hgetall(f"memory:{memory_id}")
                if memory_data:
                    # Convert embedding back from JSON
                    if memory_data.get("embedding"):
                        try:
                            memory_data["embedding"] = json.loads(
                                memory_data["embedding"]
                            )
                        except (json.JSONDecodeError, TypeError):
                            memory_data["embedding"] = None
                    else:
                        memory_data["embedding"] = None

                    # Convert context back from JSON
                    if memory_data.get("context"):
                        try:
                            memory_data["context"] = json.loads(memory_data["context"])
                        except (json.JSONDecodeError, TypeError):
                            memory_data["context"] = {}
                    else:
                        memory_data["context"] = {}

                    # Convert numeric fields back from strings
                    memory_data["timestamp"] = (
                        float(memory_data.get("timestamp", 0))
                        if memory_data.get("timestamp")
                        else 0.0
                    )
                    memory_data["decay_factor"] = (
                        float(memory_data.get("decay_factor", 1.0))
                        if memory_data.get("decay_factor")
                        else 1.0
                    )
                    memory_data["access_count"] = (
                        int(memory_data.get("access_count", 0))
                        if memory_data.get("access_count")
                        else 0
                    )

                    # Handle last_accessed which might be empty string for None
                    last_accessed_str = memory_data.get("last_accessed", "")
                    if last_accessed_str and last_accessed_str != "":
                        memory_data["last_accessed"] = float(last_accessed_str)
                    else:
                        memory_data["last_accessed"] = None

                    # Convert enum values back to proper types
                    # MemoryImportance is stored as integer but retrieved as string
                    if memory_data.get("importance"):
                        memory_data["importance"] = int(memory_data["importance"])

                    # MemoryType should remain as string (enum value)

                    memory = Memory.from_dict(memory_data)

                    # Filter by type if specified
                    if memory_types is None or memory.memory_type in memory_types:
                        memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Error getting user memories: {e}")
            return []

    async def _update_memory_access(self, memory: Memory):
        """Update memory access statistics"""
        try:
            await self.redis_client.hset(
                f"memory:{memory.memory_id}",
                mapping={
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed or "",
                },
            )
        except Exception as e:
            logger.error(f"Error updating memory access: {e}")

    async def _cleanup_old_memories(self, user_id: str):
        """Clean up old, low-importance memories"""
        try:
            memories = await self._get_user_memories(user_id)

            if len(memories) <= self.max_memories_per_user:
                return

            # Sort by relevance (considering decay)
            current_time = time.time()
            scored_memories = []
            for memory in memories:
                score = memory.calculate_relevance_score(current_time, 0.0)
                scored_memories.append((memory, score))

            # Keep the most relevant memories
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            to_delete = [
                mem for mem, score in scored_memories[self.max_memories_per_user :]
            ]

            # Delete old memories
            for memory, _ in to_delete:
                await self._delete_memory(memory.memory_id, user_id)

            if to_delete:
                logger.info(
                    f"Cleaned up {len(to_delete)} old memories for user {user_id}"
                )

        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")

    async def _delete_memory(self, memory_id: str, user_id: str):
        """Delete a specific memory"""
        try:
            await self.redis_client.delete(f"memory:{memory_id}")
            await self.redis_client.srem(f"user_memories:{user_id}", memory_id)
            # Note: Not removing from type index for performance, it will be cleaned up eventually
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        try:
            memories = await self._get_user_memories(user_id)

            stats = {
                "total_memories": len(memories),
                "by_type": {},
                "by_importance": {},
                "oldest_memory": None,
                "newest_memory": None,
                "most_accessed": None,
            }

            if not memories:
                return stats

            # Count by type and importance
            for memory in memories:
                memory_type = memory.memory_type.value
                importance = memory.importance.value

                stats["by_type"][memory_type] = stats["by_type"].get(memory_type, 0) + 1
                stats["by_importance"][importance] = (
                    stats["by_importance"].get(importance, 0) + 1
                )

            # Find oldest and newest
            memories.sort(key=lambda m: m.timestamp)
            stats["oldest_memory"] = datetime.fromtimestamp(
                memories[0].timestamp
            ).isoformat()
            stats["newest_memory"] = datetime.fromtimestamp(
                memories[-1].timestamp
            ).isoformat()

            # Find most accessed
            most_accessed = max(memories, key=lambda m: m.access_count)
            stats["most_accessed"] = {
                "content": most_accessed.content[:100] + "..."
                if len(most_accessed.content) > 100
                else most_accessed.content,
                "access_count": most_accessed.access_count,
                "type": most_accessed.memory_type.value,
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close Redis connection"""
        await self.redis_client.close()
