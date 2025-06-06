import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from redisvl.extensions.message_history import SemanticMessageHistory
import redis.asyncio as redis
from phillm.ai.embeddings import EmbeddingService
from phillm.telemetry import get_tracer


class ConversationSessionManager:
    """Manages user conversation sessions using RedisVL SemanticSessionManager"""

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.connection_timeout = 10

        # Initialize Redis connections
        self.redis_client = None
        self.sync_redis_client = None

        # Cache of user sessions
        self.user_sessions: Dict[str, SemanticMessageHistory] = {}

        # Session configuration
        self.distance_threshold = float(
            os.getenv("CONVERSATION_DISTANCE_THRESHOLD", "0.35")
        )
        self.max_context_messages = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))

        # Initialize embedding service for session management
        self.embedding_service = EmbeddingService()

    async def _ensure_redis_connection(self) -> None:
        """Ensure Redis connection is established"""
        if self.redis_client and self.sync_redis_client:
            try:
                await self.redis_client.ping()
                return
            except Exception:
                pass

        logger.info("Establishing Redis connection for conversation sessions...")

        # Async client
        self.redis_client = redis.from_url(
            self.redis_url,
            password=self.redis_password,
            decode_responses=True,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.connection_timeout,
        )

        # Sync client for RedisVL
        import redis as sync_redis

        self.sync_redis_client = sync_redis.Redis.from_url(
            self.redis_url,
            password=self.redis_password,
            decode_responses=True,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.connection_timeout,
        )

        # Test connections
        await self.redis_client.ping()
        self.sync_redis_client.ping()

        logger.info("✅ Redis connection established for conversation sessions")

    def _get_user_session(self, user_id: str) -> SemanticMessageHistory:
        """Get or create a semantic session for a user"""
        if user_id not in self.user_sessions:
            session_name = f"user_session_{user_id}"

            # Use custom vectorizer to integrate with our existing embedding service
            from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

            # Create session with OpenAI vectorizer to match our embedding service
            session = SemanticMessageHistory(
                name=session_name,
                redis_client=self.sync_redis_client,
                distance_threshold=self.distance_threshold,
                vectorizer=OpenAITextVectorizer(
                    model="text-embedding-3-large",  # Match our embedding service
                    api_config={"api_key": os.getenv("OPENAI_API_KEY")},
                ),
            )

            self.user_sessions[user_id] = session
            logger.debug(f"Created new conversation session for user {user_id}")

        return self.user_sessions[user_id]

    async def add_conversation_turn(
        self,
        user_id: str,
        user_message: str,
        bot_response: str,
        venue_info: Dict[str, Any],
    ) -> None:
        """Add a complete conversation turn (user message + bot response)"""
        tracer = get_tracer()

        try:
            with tracer.start_as_current_span("add_conversation_turn") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("user_message.length", len(user_message))
                span.set_attribute("bot_response.length", len(bot_response))
                span.set_attribute("venue_type", venue_info.get("type", "unknown"))

                await self._ensure_redis_connection()
                session = self._get_user_session(user_id)

                # Generate session key for debugging consistency
                session_key = f"user_session_{user_id}"
                span.set_attribute("session_key", session_key)

                # Create metadata for both messages
                timestamp = time.time()
                base_metadata = {
                    "timestamp": timestamp,
                    "venue_type": venue_info.get("type", "unknown"),
                    "channel_id": venue_info.get("channel_id"),
                    "user_id": user_id,
                }

                # Use the store method to add the conversation turn
                session.store(
                    prompt=user_message, response=bot_response, metadata=base_metadata
                )

                # Get total messages after adding for debugging
                try:
                    all_messages_after = session.get_relevant(
                        prompt="",  # Empty query to get all messages
                        distance_threshold=1.0,  # Max threshold to get everything
                        top_k=1000,  # Large limit
                    )
                    total_messages_after_add = len(all_messages_after)
                    span.set_attribute(
                        "total_messages_after_add", total_messages_after_add
                    )

                    # Log detailed storage verification
                    logger.info(
                        f"🔍 STORAGE VERIFICATION - User {user_id}: {total_messages_after_add} total messages after adding"
                    )
                    if all_messages_after:
                        latest_msg = all_messages_after[-1]  # Get most recent
                        logger.info(
                            f"🔍 Latest stored message role: {latest_msg.get('role')}, content preview: {latest_msg.get('content', '')[:50]}..."
                        )

                        # Test immediate retrieval with the exact same user message
                        test_retrieval = session.get_relevant(
                            prompt=user_message,  # Use exact same message
                            distance_threshold=0.1,  # Very lenient threshold
                            top_k=5,
                        )
                        span.set_attribute(
                            "immediate_test_retrieval_count", len(test_retrieval)
                        )
                        logger.info(
                            f"🔍 IMMEDIATE TEST - Retrieved {len(test_retrieval)} messages for exact same query with 0.1 threshold"
                        )

                        if test_retrieval:
                            for i, msg in enumerate(test_retrieval):
                                similarity_info = "unknown"
                                if isinstance(msg, dict) and "metadata" in msg:
                                    metadata = msg["metadata"]
                                    if "similarity" in metadata:
                                        similarity_info = (
                                            f"sim:{metadata['similarity']:.3f}"
                                        )
                                    elif "distance" in metadata:
                                        similarity_info = (
                                            f"dist:{metadata['distance']:.3f}"
                                        )
                                logger.info(
                                    f"🔍   Retrieved #{i+1}: role={msg.get('role')}, {similarity_info}, content={msg.get('content', '')[:30]}..."
                                )

                except Exception as e:
                    logger.warning(f"Failed to get total messages count after add: {e}")
                    span.set_attribute("total_messages_after_add", -1)

                logger.debug(
                    f"Stored conversation turn for user {user_id} in {venue_info.get('type', 'unknown')} venue"
                )

        except Exception as e:
            logger.error(f"Error adding conversation turn for user {user_id}: {e}")
            raise

    async def get_relevant_conversation_context(
        self,
        user_id: str,
        current_query: str,
        venue_info: Dict[str, Any],
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get semantically relevant conversation context for current query"""
        tracer = get_tracer()

        try:
            with tracer.start_as_current_span(
                "get_relevant_conversation_context"
            ) as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("current_query.length", len(current_query))
                span.set_attribute("venue_type", venue_info.get("type", "unknown"))

                await self._ensure_redis_connection()
                session = self._get_user_session(user_id)

                # Generate session key for debugging
                session_key = f"user_session_{user_id}"
                span.set_attribute("session_key", session_key)
                span.set_attribute("distance_threshold", self.distance_threshold)

                # Get total messages in session first for debugging
                try:
                    all_messages = session.get_relevant(
                        prompt="",  # Empty query to get all messages
                        distance_threshold=1.0,  # Max threshold to get everything
                        top_k=1000,  # Large limit
                    )
                    total_messages_in_session = len(all_messages)
                    span.set_attribute(
                        "total_messages_in_session", total_messages_in_session
                    )
                    span.set_attribute("session_exists", total_messages_in_session > 0)

                    # Enhanced debugging for retrieval
                    logger.info(
                        f"🔍 RETRIEVAL DEBUG - User {user_id}, Query: '{current_query[:50]}...', Session has {total_messages_in_session} total messages"
                    )
                    if all_messages:
                        logger.info("🔍 Sample stored messages:")
                        for i, msg in enumerate(
                            all_messages[-3:]
                        ):  # Show last 3 messages
                            logger.info(
                                f"🔍   Stored #{i+1}: role={msg.get('role')}, content='{msg.get('content', '')[:50]}...'"
                            )

                except Exception as e:
                    logger.warning(f"Failed to get total messages count: {e}")
                    span.set_attribute("total_messages_in_session", -1)
                    span.set_attribute("session_exists", False)

                # Get relevant messages based on semantic similarity
                top_k = max_messages or self.max_context_messages
                logger.info(
                    f"🔍 SEMANTIC SEARCH - Searching for '{current_query[:50]}...' with threshold {self.distance_threshold}, top_k={top_k}"
                )

                relevant_messages = session.get_relevant(
                    prompt=current_query,
                    distance_threshold=self.distance_threshold,
                    top_k=top_k,
                )

                # Count messages found before distance filtering
                span.set_attribute(
                    "messages_found_before_distance_filter", len(relevant_messages)
                )
                logger.info(
                    f"🔍 SEMANTIC SEARCH RESULT - Found {len(relevant_messages)} relevant messages"
                )

                # Find the closest similarity score for debugging
                closest_similarity_score = 0.0
                if relevant_messages:
                    # SemanticMessageHistory includes similarity scores in the metadata
                    similarity_scores = []
                    logger.info(
                        f"🔍 ANALYZING {len(relevant_messages)} relevant messages:"
                    )
                    for i, msg in enumerate(relevant_messages):
                        # Check if the message has similarity metadata
                        if isinstance(msg, dict) and "metadata" in msg:
                            metadata = msg["metadata"]
                            score_info = "no_score"
                            if "similarity" in metadata:
                                score = float(metadata["similarity"])
                                similarity_scores.append(score)
                                score_info = f"sim:{score:.3f}"
                            elif "distance" in metadata:
                                # Convert distance to similarity (assuming cosine distance)
                                distance = float(metadata["distance"])
                                score = 1.0 - distance
                                similarity_scores.append(score)
                                score_info = f"dist:{distance:.3f}/sim:{score:.3f}"

                            logger.info(
                                f"🔍   #{i+1}: {score_info}, role={msg.get('role')}, content='{msg.get('content', '')[:40]}...'"
                            )

                    if similarity_scores:
                        closest_similarity_score = max(similarity_scores)
                        logger.info(
                            f"🔍 BEST SIMILARITY SCORE: {closest_similarity_score:.3f}"
                        )
                    else:
                        logger.info("🔍 NO SIMILARITY SCORES FOUND in message metadata")

                span.set_attribute("closest_similarity_score", closest_similarity_score)

                # Filter for venue appropriateness if needed
                filtered_messages = self._filter_for_venue_privacy(
                    relevant_messages,
                    current_venue_type=venue_info.get("type", "unknown"),
                )

                span.set_attribute("relevant_messages.count", len(filtered_messages))

                logger.debug(
                    f"Retrieved {len(filtered_messages)} relevant conversation messages for user {user_id}"
                )

                return filtered_messages

        except Exception as e:
            logger.error(
                f"Error getting relevant conversation context for user {user_id}: {e}"
            )
            return []

    def _filter_for_venue_privacy(
        self, messages: List[Dict[str, Any]], current_venue_type: str
    ) -> List[Dict[str, Any]]:
        """Filter conversation context based on venue privacy rules"""
        filtered = []

        for message in messages:
            metadata = message.get("metadata", {})
            message_venue = metadata.get("venue_type", "unknown")

            # Privacy rule: Don't leak DM context into public channels
            if current_venue_type == "channel" and message_venue == "dm":
                # Skip DM messages when responding in public channels
                # unless explicitly marked as public-safe
                if not metadata.get("public_safe", False):
                    continue

            # All other combinations are allowed
            filtered.append(message)

        return filtered

    async def get_conversation_history_for_prompt(
        self,
        user_id: str,
        current_query: str,
        venue_info: Dict[str, Any],
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get conversation history formatted for chat completion messages"""
        relevant_messages = await self.get_relevant_conversation_context(
            user_id, current_query, venue_info, max_messages
        )

        # Convert to OpenAI chat format
        formatted_messages = []
        for msg in relevant_messages:
            # Only include role and content for the chat completion
            formatted_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )

        return formatted_messages

    async def clear_user_session(self, user_id: str) -> None:
        """Clear all conversation history for a user"""
        try:
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                session.clear()
                logger.info(f"Cleared conversation session for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing session for user {user_id}: {e}")

    async def get_session_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's conversation session"""
        try:
            await self._ensure_redis_connection()
            session = self._get_user_session(user_id)

            # Get all messages to analyze
            all_messages = session.get_relevant(
                prompt="",  # Empty query to get all messages
                distance_threshold=1.0,  # Max threshold to get everything
                top_k=1000,  # Large limit
            )

            stats: Dict[str, Any] = {
                "total_messages": len(all_messages),
                "user_messages": len(
                    [m for m in all_messages if m.get("role") == "user"]
                ),
                "bot_messages": len(
                    [m for m in all_messages if m.get("role") == "assistant"]
                ),
                "venue_breakdown": {},
                "oldest_message": None,
                "newest_message": None,
            }

            # Analyze venues and timestamps
            timestamps = []
            for msg in all_messages:
                metadata = msg.get("metadata", {})
                venue = metadata.get("venue_type", "unknown")

                if venue in stats["venue_breakdown"]:
                    stats["venue_breakdown"][venue] += 1
                else:
                    stats["venue_breakdown"][venue] = 1

                if "timestamp" in metadata:
                    timestamps.append(metadata["timestamp"])

            if timestamps:
                stats["oldest_message"] = datetime.fromtimestamp(
                    min(timestamps)
                ).isoformat()
                stats["newest_message"] = datetime.fromtimestamp(
                    max(timestamps)
                ).isoformat()

            return stats

        except Exception as e:
            logger.error(f"Error getting session stats for user {user_id}: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close Redis connections and cleanup"""
        if self.redis_client:
            await self.redis_client.close()
        if self.sync_redis_client:
            self.sync_redis_client.close()

        self.user_sessions.clear()
        logger.info("Conversation session manager closed")
