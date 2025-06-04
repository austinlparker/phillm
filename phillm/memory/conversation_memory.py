import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import numpy as np
from loguru import logger
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
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
    """Manages conversation memory for users using Redis vector search"""

    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")

        # Connection configuration
        self.connection_timeout = 10  # seconds

        # Initialize clients as None, will be created on demand
        self.redis_client = None
        self.sync_redis_client = None
        self._redis_healthy = False

        # Vector dimensions for text-embedding-3-large
        self.vector_dim = 3072

        # Memory configuration
        self.max_memories_per_user = int(os.getenv("MAX_MEMORIES_PER_USER", "1000"))
        self.memory_retention_days = int(os.getenv("MEMORY_RETENTION_DAYS", "30"))
        self.embedding_service = None  # Will be injected

        # Index configuration for memories
        self.index_name = "phillm_memories"
        self.index = None

        # Schema for memory vector search
        self.schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": self.index_name,
                    "prefix": "mem:",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "user_id", "type": "tag"},
                    {"name": "memory_type", "type": "tag"},
                    {"name": "importance", "type": "numeric"},
                    {"name": "timestamp", "type": "numeric"},
                    {"name": "access_count", "type": "numeric"},
                    {"name": "last_accessed", "type": "numeric"},
                    {"name": "decay_factor", "type": "numeric"},
                    {"name": "content", "type": "text"},
                    {"name": "context", "type": "text"},
                    {"name": "memory_id", "type": "tag"},
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

    async def _ensure_redis_connection(self) -> None:
        """Ensure Redis connection is established and healthy"""
        if self._redis_healthy and self.redis_client and self.sync_redis_client:
            try:
                # Quick health check
                await self.redis_client.ping()
                return
            except Exception:
                # Connection lost, need to reconnect
                self._redis_healthy = False

        logger.info("Connecting to Redis for memory management...")

        # Initialize async Redis client for regular operations
        self.redis_client = redis.from_url(
            self.redis_url,
            password=self.redis_password,
            decode_responses=True,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.connection_timeout,
        )

        # Initialize sync Redis client for RedisVL operations
        import redis as sync_redis

        self.sync_redis_client = sync_redis.Redis.from_url(
            self.redis_url,
            password=self.redis_password,
            decode_responses=True,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.connection_timeout,
        )

        # Test the connections
        await self.redis_client.ping()  # type: ignore[attr-defined]
        self.sync_redis_client.ping()  # type: ignore[attr-defined]

        self._redis_healthy = True
        logger.info("✅ Redis connection established for memory management")

    async def initialize_index(self) -> None:
        """Initialize the memory vector search index"""
        # Ensure Redis connection first
        await self._ensure_redis_connection()

        # Create SearchIndex instance using the sync redis client
        self.index = SearchIndex(
            schema=self.schema, redis_client=self.sync_redis_client
        )

        # Check if index exists, create if not
        if not self._index_exists_sync():
            logger.info(f"Creating memory vector index: {self.index_name}")
            self.index.create(overwrite=False)  # type: ignore[attr-defined]
            logger.info("✅ Memory vector index created successfully")
        else:
            logger.info(f"Memory vector index {self.index_name} already exists")
            # Connect to existing index
            self.index.connect()  # type: ignore[attr-defined]

    def _index_exists_sync(self) -> bool:
        """Check if the vector index exists (sync version for initialization)"""
        try:
            # Check if index exists by looking for it in the index list
            indices = self.sync_redis_client.execute_command("FT._LIST")  # type: ignore[attr-defined]
            # indices is a list of byte strings, so check for both string and bytes
            return (
                (self.index_name in indices) or (self.index_name.encode() in indices)
                if indices
                else False
            )
        except Exception:
            return False

    def set_embedding_service(self, embedding_service: Any) -> None:
        """Inject embedding service for memory vectorization"""
        self.embedding_service = embedding_service

    def _generate_memory_id(
        self, user_id: str, memory_type: MemoryType, timestamp: float
    ) -> str:
        """Generate a unique memory ID"""
        content = f"{user_id}:{memory_type.value}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()

    async def store_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        context: Dict[str, Any],
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        pre_computed_embedding: Optional[List[float]] = None,
    ) -> str:
        """Store a new memory using vector search index"""
        tracer = get_tracer()
        await self._ensure_redis_connection()

        try:
            current_timestamp = time.time()

            # Generate memory ID using consistent format
            memory_id = self._generate_memory_id(
                user_id, memory_type, current_timestamp
            )

            # Use pre-computed embedding or create new one for semantic search
            embedding = pre_computed_embedding
            if not embedding and self.embedding_service and content.strip():
                try:
                    embedding = await self.embedding_service.create_embedding(content)
                except Exception as e:
                    logger.warning(f"Failed to create embedding for memory: {e}")
                    # Continue without embedding - we can still store the memory
                    embedding = None

            with tracer.start_as_current_span("store_memory") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("memory_type", memory_type.value)
                span.set_attribute("content_length", len(content))
                span.set_attribute("importance", importance.value)
                span.set_attribute("has_embedding", embedding is not None)

                # Ensure index is initialized
                if not self.index:
                    await self.initialize_index()

                # Prepare data for vector storage
                doc_key = f"mem:{memory_id}"
                memory_data: Dict[str, Any] = {
                    "user_id": user_id,
                    "memory_type": memory_type.value,
                    "content": content,
                    "context": json.dumps(context),
                    "importance": importance.value,
                    "timestamp": current_timestamp,
                    "memory_id": memory_id,
                    "decay_factor": 1.0,
                    "access_count": 0,
                    "last_accessed": 0.0,  # Use 0.0 instead of None for numeric field
                }

                # Add embedding if available
                if embedding:
                    memory_data["embedding"] = np.array(
                        embedding, dtype=np.float32
                    ).tobytes()
                else:
                    # Create zero vector for memories without embeddings
                    memory_data["embedding"] = np.zeros(
                        self.vector_dim, dtype=np.float32
                    ).tobytes()

                # Store in Redis with vector index prefix
                await self.redis_client.hset(doc_key, mapping=memory_data)  # type: ignore[attr-defined]

                # Clean up old memories if needed
                await self._cleanup_old_memories(user_id)

                logger.debug(f"Stored vector memory {memory_id} for user {user_id}")
                return memory_id

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    async def recall_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_relevance: float = 0.3,
    ) -> List[Memory]:
        """Recall relevant memories using Redis vector search"""
        tracer = get_tracer()
        current_time = time.time()
        await self._ensure_redis_connection()

        try:
            with tracer.start_as_current_span("recall_memories") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("query_length", len(query) if query else 0)
                span.set_attribute("limit", limit)
                span.set_attribute("has_memory_types_filter", memory_types is not None)

                # Ensure index is initialized
                if not self.index:
                    await self.initialize_index()

                if query and self.embedding_service:
                    # Use vector similarity search for semantic matching
                    try:
                        query_embedding = await self.embedding_service.create_embedding(
                            query
                        )
                        memories = await self._vector_search_memories(
                            user_id,
                            query_embedding,
                            memory_types,
                            limit * 2,  # Get more for filtering
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create query embedding, falling back to text search: {e}"
                        )
                        memories = await self._text_search_memories(
                            user_id, query, memory_types, limit * 2
                        )
                else:
                    # Use text search or get recent memories
                    if query:
                        memories = await self._text_search_memories(
                            user_id, query, memory_types, limit * 2
                        )
                    else:
                        memories = await self._get_recent_memories(
                            user_id, memory_types, limit * 2
                        )

                if not memories:
                    return []

                # Score memories with time decay and access frequency
                relevant_memories = []
                for memory_data in memories:
                    try:
                        memory = self._memory_from_search_result(memory_data)

                        # Calculate base similarity from search result
                        base_similarity = memory_data.get("similarity", 0.0)

                        # Calculate overall relevance with time decay
                        relevance = memory.calculate_relevance_score(
                            current_time, base_similarity
                        )

                        if relevance >= min_relevance:
                            relevant_memories.append((memory, relevance))

                            # Update access count asynchronously
                            memory.access_count += 1
                            memory.last_accessed = current_time
                            await self._update_memory_access_vector(memory)

                    except Exception as e:
                        logger.warning(f"Failed to process memory result: {e}")
                        continue

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
        query_embedding: Optional[List[float]] = None,
    ) -> None:
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
        self,
        user_id: str,
        message: str,
        channel_id: str,
        channel_name: Optional[str] = None,
    ) -> None:
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

    async def store_user_preference(
        self, user_id: str, preference: str, value: str
    ) -> None:
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

    async def _vector_search_memories(
        self,
        user_id: str,
        query_embedding: List[float],
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search memories using vector similarity"""
        try:
            # Ensure consistent float32 format
            query_vector = np.array(query_embedding, dtype=np.float32)

            # Build filter expression for user
            filter_expression = Tag("user_id") == user_id

            # Add memory type filter if specified
            if memory_types:
                type_values = [mt.value for mt in memory_types]
                if len(type_values) == 1:
                    filter_expression = filter_expression & (
                        Tag("memory_type") == type_values[0]
                    )
                else:
                    # For multiple types, use OR logic within the memory_type filter
                    type_filter = Tag("memory_type").in_set(type_values)
                    filter_expression = filter_expression & type_filter

            # Create vector query
            vector_query = VectorQuery(
                vector=query_vector,
                vector_field_name="embedding",
                return_fields=[
                    "user_id",
                    "memory_type",
                    "content",
                    "context",
                    "importance",
                    "timestamp",
                    "memory_id",
                    "decay_factor",
                    "access_count",
                    "last_accessed",
                ],
                num_results=limit,
                filter_expression=filter_expression,
            )

            # Execute search
            results = self.index.query(vector_query)  # type: ignore[attr-defined]

            # Process results and convert distance to similarity
            processed_results = []
            for result in results:
                # Redis vector search returns distance, convert to similarity
                distance = float(result.get("vector_distance", 1.0))
                similarity = 1.0 - distance  # Cosine distance to similarity

                result_data = dict(result)
                result_data["similarity"] = similarity
                processed_results.append(result_data)

            return processed_results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    async def _text_search_memories(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search memories using text search"""
        try:
            # Build search query
            search_query = f"@user_id:{{{user_id}}} @content:{query}"

            # Add memory type filter if specified
            if memory_types:
                type_values = [mt.value for mt in memory_types]
                if len(type_values) == 1:
                    search_query += f" @memory_type:{{{type_values[0]}}}"
                else:
                    type_query = "|".join([f"{{{t}}}" for t in type_values])
                    search_query += f" @memory_type:({type_query})"

            # Execute text search
            result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                "FT.SEARCH",
                self.index_name,
                search_query,
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "10",
                "user_id",
                "memory_type",
                "content",
                "context",
                "importance",
                "timestamp",
                "memory_id",
                "decay_factor",
                "access_count",
                "last_accessed",
            )

            return self._parse_search_results(
                result, similarity=0.7
            )  # Default text match similarity

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []

    async def _get_recent_memories(
        self,
        user_id: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get recent memories for a user, sorted by timestamp"""
        try:
            # Build search query for user
            search_query = f"@user_id:{{{user_id}}}"

            # Add memory type filter if specified
            if memory_types:
                type_values = [mt.value for mt in memory_types]
                if len(type_values) == 1:
                    search_query += f" @memory_type:{{{type_values[0]}}}"
                else:
                    type_query = "|".join([f"{{{t}}}" for t in type_values])
                    search_query += f" @memory_type:({type_query})"

            # Execute search sorted by timestamp (newest first)
            result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                "FT.SEARCH",
                self.index_name,
                search_query,
                "SORTBY",
                "timestamp",
                "DESC",
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "10",
                "user_id",
                "memory_type",
                "content",
                "context",
                "importance",
                "timestamp",
                "memory_id",
                "decay_factor",
                "access_count",
                "last_accessed",
            )

            return self._parse_search_results(
                result, similarity=0.0
            )  # No similarity for recent memories

        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []

    def _parse_search_results(
        self, result: List[Any], similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Parse Redis search results into memory data"""
        memories = []
        if len(result) > 1:
            # Skip count (first element) and process results in pairs
            for i in range(1, len(result), 2):
                doc_id = result[i]
                fields = result[i + 1] if i + 1 < len(result) else []

                # Extract memory ID from document key (mem:id)
                memory_id = (
                    doc_id.replace("mem:", "") if doc_id.startswith("mem:") else doc_id
                )

                # Parse field-value pairs
                field_data = {"similarity": similarity}
                for j in range(0, len(fields), 2):
                    if j + 1 < len(fields):
                        field_data[fields[j]] = fields[j + 1]

                # Ensure memory_id is set
                if "memory_id" not in field_data:
                    field_data["memory_id"] = memory_id

                memories.append(field_data)

        return memories

    def _memory_from_search_result(self, result_data: Dict[str, Any]) -> Memory:
        """Convert search result to Memory object"""
        # Parse context JSON
        context = {}
        if result_data.get("context"):
            try:
                context = json.loads(result_data["context"])
            except (json.JSONDecodeError, TypeError):
                context = {}

        # Convert numeric fields
        timestamp = float(result_data.get("timestamp", 0))
        importance = int(result_data.get("importance", MemoryImportance.MEDIUM.value))
        decay_factor = float(result_data.get("decay_factor", 1.0))
        access_count = int(result_data.get("access_count", 0))

        # Handle last_accessed
        last_accessed = None
        if result_data.get("last_accessed"):
            try:
                last_accessed_val = float(result_data["last_accessed"])
                if last_accessed_val > 0:
                    last_accessed = last_accessed_val
            except (ValueError, TypeError):
                pass

        return Memory(
            memory_id=result_data.get("memory_id", ""),
            user_id=result_data.get("user_id", ""),
            memory_type=MemoryType(
                result_data.get("memory_type", MemoryType.DM_CONVERSATION.value)
            ),
            content=result_data.get("content", ""),
            context=context,
            importance=MemoryImportance(importance),
            timestamp=timestamp,
            embedding=None,  # Don't load embeddings for recall (performance)
            decay_factor=decay_factor,
            access_count=access_count,
            last_accessed=last_accessed,
        )

    async def _update_memory_access_vector(self, memory: Memory) -> None:
        """Update memory access statistics in vector store"""
        try:
            doc_key = f"mem:{memory.memory_id}"
            await self.redis_client.hset(  # type: ignore[attr-defined]
                doc_key,
                mapping={
                    "access_count": memory.access_count,
                    "last_accessed": memory.last_accessed or 0.0,
                },
            )
        except Exception as e:
            logger.error(f"Error updating memory access: {e}")

    # Legacy method for backwards compatibility
    async def _update_memory_access(self, memory: Memory) -> None:
        """Legacy update method - redirects to vector version"""
        await self._update_memory_access_vector(memory)

    async def _cleanup_old_memories(self, user_id: str) -> None:
        """Clean up old, low-importance memories using vector index"""
        try:
            # Get total count of user memories
            memory_count = await self._get_user_memory_count(user_id)

            if memory_count <= self.max_memories_per_user:
                return

            # Get all user memories sorted by timestamp (oldest first)
            memories_to_score = await self._get_recent_memories(
                user_id, None, memory_count  # Get all memories
            )

            if len(memories_to_score) <= self.max_memories_per_user:
                return

            # Convert to Memory objects and score them
            current_time = time.time()
            scored_memories = []

            for memory_data in memories_to_score:
                try:
                    memory = self._memory_from_search_result(memory_data)
                    score = memory.calculate_relevance_score(current_time, 0.0)
                    scored_memories.append((memory, score))
                except Exception as e:
                    logger.warning(f"Failed to score memory for cleanup: {e}")
                    continue

            # Keep the most relevant memories
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            to_delete = [
                mem for mem, score in scored_memories[self.max_memories_per_user :]
            ]

            # Delete old memories
            for memory in to_delete:
                await self._delete_memory_vector(memory.memory_id)

            if to_delete:
                logger.info(
                    f"Cleaned up {len(to_delete)} old memories for user {user_id}"
                )

        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")

    async def _get_user_memory_count(self, user_id: str) -> int:
        """Get total memory count for a user using vector index"""
        try:
            result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                "FT.SEARCH",
                self.index_name,
                f"@user_id:{{{user_id}}}",
                "LIMIT",
                "0",
                "0",
            )
            # First element is the count
            return int(result[0]) if result else 0
        except Exception as e:
            logger.error(f"Error getting user memory count: {e}")
            return 0

    async def _delete_memory_vector(self, memory_id: str) -> None:
        """Delete a specific memory from vector store"""
        try:
            doc_key = f"mem:{memory_id}"
            await self.redis_client.delete(doc_key)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")

    # Legacy method for backwards compatibility
    async def _delete_memory(self, memory_id: str, user_id: str) -> None:
        """Legacy delete method - redirects to vector version"""
        await self._delete_memory_vector(memory_id)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors (legacy method)"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user using vector index queries"""
        try:
            await self._ensure_redis_connection()

            # Get total count
            total_memories = await self._get_user_memory_count(user_id)

            stats: Dict[str, Any] = {
                "total_memories": total_memories,
                "by_type": {},
                "by_importance": {},
                "oldest_memory": None,
                "newest_memory": None,
                "most_accessed": None,
            }

            if total_memories == 0:
                return stats

            # Get aggregated stats using Redis search
            try:
                # Get count by type
                for memory_type in MemoryType:
                    type_result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                        "FT.SEARCH",
                        self.index_name,
                        f"@user_id:{{{user_id}}} @memory_type:{{{memory_type.value}}}",
                        "LIMIT",
                        "0",
                        "0",
                    )
                    count = int(type_result[0]) if type_result else 0
                    if count > 0:
                        stats["by_type"][memory_type.value] = count

                # Get count by importance
                for importance in MemoryImportance:
                    importance_result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                        "FT.SEARCH",
                        self.index_name,
                        f"@user_id:{{{user_id}}} @importance:[{importance.value} {importance.value}]",
                        "LIMIT",
                        "0",
                        "0",
                    )
                    count = int(importance_result[0]) if importance_result else 0
                    if count > 0:
                        stats["by_importance"][importance.value] = count

                # Get oldest memory
                oldest_result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                    "FT.SEARCH",
                    self.index_name,
                    f"@user_id:{{{user_id}}}",
                    "SORTBY",
                    "timestamp",
                    "ASC",
                    "LIMIT",
                    "0",
                    "1",
                    "RETURN",
                    "1",
                    "timestamp",
                )
                if len(oldest_result) > 2:
                    oldest_timestamp = float(oldest_result[2])
                    stats["oldest_memory"] = datetime.fromtimestamp(
                        oldest_timestamp
                    ).isoformat()

                # Get newest memory
                newest_result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                    "FT.SEARCH",
                    self.index_name,
                    f"@user_id:{{{user_id}}}",
                    "SORTBY",
                    "timestamp",
                    "DESC",
                    "LIMIT",
                    "0",
                    "1",
                    "RETURN",
                    "1",
                    "timestamp",
                )
                if len(newest_result) > 2:
                    newest_timestamp = float(newest_result[2])
                    stats["newest_memory"] = datetime.fromtimestamp(
                        newest_timestamp
                    ).isoformat()

                # Get most accessed memory
                most_accessed_result = await self.redis_client.execute_command(  # type: ignore[attr-defined]
                    "FT.SEARCH",
                    self.index_name,
                    f"@user_id:{{{user_id}}}",
                    "SORTBY",
                    "access_count",
                    "DESC",
                    "LIMIT",
                    "0",
                    "1",
                    "RETURN",
                    "4",
                    "content",
                    "access_count",
                    "memory_type",
                )
                if len(most_accessed_result) > 2:
                    # Parse the result fields
                    fields = most_accessed_result[2:]
                    field_data = {}
                    for j in range(0, len(fields), 2):
                        if j + 1 < len(fields):
                            field_data[fields[j]] = fields[j + 1]

                    content = field_data.get("content", "")
                    stats["most_accessed"] = {
                        "content": (
                            content[:100] + "..." if len(content) > 100 else content
                        ),
                        "access_count": int(field_data.get("access_count", 0)),
                        "type": field_data.get("memory_type", ""),
                    }

            except Exception as e:
                logger.warning(f"Error getting detailed stats, using basic count: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()  # type: ignore[attr-defined]
        if self.sync_redis_client:
            self.sync_redis_client.close()  # type: ignore[attr-defined]
