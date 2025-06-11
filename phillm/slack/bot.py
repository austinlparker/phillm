import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from loguru import logger

from phillm.ai.embeddings import EmbeddingService
from phillm.ai.completions import CompletionService
from phillm.conversation import ConversationSessionManager
from phillm.user import UserManager
from phillm.vector.redis_vector_store import RedisVectorStore
from phillm.api.debug import update_stats, add_error
from phillm.telemetry import get_tracer, telemetry


class SlackBot:
    def __init__(self):
        self.app = AsyncApp(
            token=os.getenv("SLACK_BOT_TOKEN"),
            signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
        )
        self.handler = AsyncSocketModeHandler(self.app, os.getenv("SLACK_APP_TOKEN"))

        self.target_user_id = os.getenv("TARGET_USER_ID")
        self.scrape_channels = os.getenv("SCRAPE_CHANNELS", "").split(",")

        self.embedding_service = EmbeddingService()
        self.completion_service = CompletionService()

        # Initialize conversation session manager
        self.conversation_sessions = ConversationSessionManager()

        # Initialize user manager
        self.user_manager = UserManager(self.app.client)

        # Initialize vector store for RAG
        self.vector_store = RedisVectorStore()

        self._setup_event_handlers()

    def _setup_event_handlers(self):
        @self.app.event("message")
        async def handle_message_events(body, logger):
            # Log all incoming events for debugging
            event = body.get("event", {})
            logger.info(
                f"🔍 RECEIVED EVENT: type=message, channel_type={event.get('channel_type')}, user={event.get('user')}, subtype={event.get('subtype')}, bot_id={event.get('bot_id')}"
            )
            await self._handle_message(body)

        @self.app.event("app_mention")
        async def handle_mentions(body, say):
            await self._handle_mention(body, say)

        # Add explicit handler for im_created events
        @self.app.event("im_created")
        async def handle_im_created(body, logger):
            logger.info(f"🔍 IM CREATED EVENT: {body}")

        # Catch-all event handler for debugging
        @self.app.event({"type": lambda x: True})  # type: ignore[dict-item]
        async def handle_all_events(body, logger):
            event_type = body.get("event", {}).get("type", "unknown")
            if event_type not in ["message", "app_mention"]:
                logger.info(f"🔍 OTHER EVENT: {event_type} - {body.get('event', {})}")

    async def _handle_message(self, body):
        event = body.get("event", {})
        channel_type = event.get("channel_type")
        user_id = event.get("user")
        message_text = event.get("text", "")

        # Debug logging
        logger.debug(
            f"Received message: user={user_id}, channel_type={channel_type}, text={message_text[:50]}..."
        )

        # Skip messages from bots (including ourselves) and all system messages with subtypes
        if event.get("bot_id") or event.get("subtype"):
            logger.debug(
                f"Skipping bot/system message with subtype: {event.get('subtype')}"
            )
            return

        # Handle target user messages for scraping (in channels)
        if user_id == self.target_user_id and channel_type != "im":
            logger.debug("Processing target user message for scraping")
            await self._process_target_user_message(event)
            return

        # Handle direct messages to the bot
        if channel_type == "im":
            logger.info(f"Processing DM from user {user_id}")
            await self._handle_direct_message_event(event)
        else:
            # Store memory of channel interactions for context
            if (
                user_id and user_id != self.target_user_id
            ):  # Don't store target user's own messages
                await self._store_channel_interaction_memory(event)
            logger.debug(f"Ignoring non-DM message in channel_type={channel_type}")

    async def _process_target_user_message(self, event):
        message_text = event.get("text", "")
        channel_id = event.get("channel")

        if not self.target_user_id:
            logger.error("Target user ID not configured")
            return

        try:
            update_stats(last_api_call=datetime.now().isoformat())

            # Note: Target user message scraping is no longer needed as we're using
            # conversation sessions for memory management

            # Record metrics
            telemetry.record_message_scraped(channel_id, self.target_user_id)

            # Update stats
            update_stats(total_messages_processed=1)
            logger.info(f"Processed message from target user: {message_text[:50]}...")

        except Exception as e:
            error_msg = f"Error processing target user message: {e}"
            logger.error(error_msg)
            add_error(error_msg)

    async def _handle_mention(self, body, say):
        """Handle @ mentions in public channels"""
        tracer = get_tracer()
        event = body.get("event", {})
        message_text = event.get("text", "")
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get(
            "thread_ts"
        )  # Get thread timestamp if this is in a thread

        logger.debug(
            f"🎯 Mention Event - user_id: '{user_id}', channel_id: '{channel_id}', thread_ts: '{thread_ts}', text: '{message_text[:50]}...'"
        )

        # Skip empty messages
        if not message_text or message_text.strip() == "":
            return

        try:
            with tracer.start_as_current_span("handle_mention") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("channel_id", channel_id)
                span.set_attribute("message.length", len(message_text))
                span.set_attribute("message.preview", message_text[:100])

                logger.info(
                    f"Received @ mention from user {user_id} in channel {channel_id}: {message_text[:50]}..."
                )

                # Update stats
                update_stats(
                    mention_conversations=1,
                    last_mention_received=datetime.now().isoformat(),
                )

                # Get conversation context from session manager for mentions
                conversation_history = []
                user_display_name = None
                if user_id:
                    venue_info = {
                        "type": "channel",
                        "channel_id": channel_id,
                        "timestamp": time.time(),
                    }

                    conversation_history = await self.conversation_sessions.get_conversation_history_for_prompt(
                        user_id=user_id,
                        current_query=message_text,
                        venue_info=venue_info,
                        max_messages=4,  # Fewer messages for public channels
                    )

                    user_display_name = await self.user_manager.get_user_display_name(
                        user_id
                    )

                # Add thinking reaction to show we're processing
                await self.app.client.reactions_add(
                    channel=channel_id, timestamp=event.get("ts"), name="thinking_face"
                )

                # Generate AI response (not a DM, but in channel)
                response, query_embedding = await self._generate_ai_response(
                    message_text,
                    is_dm=False,  # This is a channel mention, not a DM
                    conversation_history=conversation_history,
                    requester_display_name=user_display_name,
                )

                span.set_attribute("response.length", len(response))
                span.set_attribute("response.preview", response[:100])

                # Send the response using say() for channel mentions
                # If this mention is in a thread, reply in the thread
                if thread_ts:
                    await say(response, thread_ts=thread_ts)
                    logger.debug(f"Replied in thread {thread_ts}")
                else:
                    await say(response)
                    logger.debug("Replied in channel (not a thread)")

                # Remove thinking reaction
                await self.app.client.reactions_remove(
                    channel=channel_id, timestamp=event.get("ts"), name="thinking_face"
                )

                # Store the mention interaction in session manager
                if user_id:
                    mention_venue_info = {
                        "type": "channel",
                        "channel_id": channel_id,
                        "timestamp": time.time(),
                    }

                    asyncio.create_task(
                        self.conversation_sessions.add_conversation_turn(
                            user_id=user_id,
                            user_message=message_text,
                            bot_response=response,
                            venue_info=mention_venue_info,
                        )
                    )

                # Record metrics
                telemetry.record_mention_processed(user_id, channel_id, len(response))

                logger.info(
                    f"Sent mention response to user {user_id} in channel {channel_id}"
                )

        except Exception as e:
            error_msg = f"Error handling mention: {e}"
            logger.error(error_msg)
            add_error(error_msg)

            with tracer.start_as_current_span("handle_mention_error") as span:
                span.set_attribute("error", str(e))
                span.set_attribute("user_id", user_id)
                span.set_attribute("channel_id", channel_id)

                try:
                    await say(
                        "Sorry, I'm having trouble generating a response right now."
                    )
                except Exception:
                    pass  # Don't let error responses fail

    async def _handle_direct_message_event(self, event):
        """Handle direct messages to the bot (from message event)"""
        tracer = get_tracer()
        user_id = event.get("user")
        channel_id = event.get("channel")
        message_text = event.get("text", "")

        logger.debug(
            f"🎯 DM Event - user_id: '{user_id}' (type: {type(user_id)}), channel_id: '{channel_id}'"
        )

        # Skip empty messages
        if not message_text or message_text.strip() == "":
            return

        try:
            with tracer.start_as_current_span("handle_dm") as span:
                span.set_attribute("user_id", user_id)
                span.set_attribute("channel_id", channel_id)
                span.set_attribute("message.length", len(message_text))
                span.set_attribute("message.preview", message_text[:100])

                logger.info(f"Received DM from user {user_id}: {message_text[:50]}...")

                # Update stats
                update_stats(
                    dm_conversations=1, last_dm_received=datetime.now().isoformat()
                )

                # Get relevant conversation history from session manager
                venue_info = {
                    "type": "dm",
                    "channel_id": channel_id,
                    "timestamp": time.time(),
                }

                conversation_history = await self.conversation_sessions.get_conversation_history_for_prompt(
                    user_id=user_id,
                    current_query=message_text,
                    venue_info=venue_info,
                    max_messages=6,  # Get up to 6 relevant conversation messages
                )

                # Get user's display name
                logger.debug(f"🔍 Fetching display name for user {user_id}")
                user_display_name = await self.user_manager.get_user_display_name(
                    user_id
                )
                logger.debug(
                    f"🔍 Got display name for user {user_id}: '{user_display_name}'"
                )

                # Add thinking reaction to show we're processing
                await self.app.client.reactions_add(
                    channel=channel_id, timestamp=event.get("ts"), name="thinking_face"
                )

                # Generate AI response using new conversation history
                response, query_embedding = await self._generate_ai_response(
                    message_text,
                    is_dm=True,
                    conversation_history=conversation_history,
                    requester_display_name=user_display_name,
                )

                span.set_attribute("response.length", len(response))
                span.set_attribute("response.preview", response[:100])

                # Send the response message
                await self.app.client.chat_postMessage(
                    channel=channel_id, text=response
                )

                # Remove thinking reaction
                await self.app.client.reactions_remove(
                    channel=channel_id, timestamp=event.get("ts"), name="thinking_face"
                )

                # Store the conversation turn in new session manager
                asyncio.create_task(
                    self.conversation_sessions.add_conversation_turn(
                        user_id=user_id,
                        user_message=message_text,
                        bot_response=response,
                        venue_info=venue_info,
                    )
                )

                # Record metrics
                telemetry.record_dm_processed(user_id, len(response))

                logger.info(f"Sent DM response to user {user_id}")

        except Exception as e:
            error_msg = f"Error handling DM: {e}"
            logger.error(error_msg)
            add_error(error_msg)

            with tracer.start_as_current_span("handle_dm_error") as span:
                span.set_attribute("error", str(e))
                span.set_attribute("user_id", user_id)

                try:
                    await self.app.client.chat_postMessage(
                        channel=channel_id,
                        text="Sorry, I'm having trouble processing your message right now. Please try again later.",
                    )
                except Exception:
                    pass  # Don't let error responses fail

    async def _get_bot_user_id(self):
        """Get the bot's user ID"""
        try:
            auth_response = await self.app.client.auth_test()
            return auth_response["user_id"]
        except Exception as e:
            logger.error(f"Error getting bot user ID: {e}")
            return None

    async def _generate_ai_response(
        self,
        query: str,
        is_dm: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        requester_display_name: Optional[str] = None,
    ) -> Tuple[str, List[float]]:
        # Create embedding once and reuse it
        from phillm.ai.embeddings import EmbeddingService

        embedding_service = EmbeddingService()
        query_embedding = await embedding_service.create_embedding(query)

        if not self.target_user_id:
            logger.error("Target user ID not configured")
            return "", []

        # Find similar historical messages for style examples via RAG
        try:
            similar_messages = await self.vector_store.find_similar_messages(
                query,  # Pass the original query text, vector store will handle embedding
                user_id=self.target_user_id,
                limit=10,
                threshold=0.5,  # Lower threshold to get more examples
            )
            logger.info(
                f"🔍 Found {len(similar_messages)} similar historical messages for style reference"
            )
        except Exception as e:
            logger.warning(
                f"Failed to retrieve similar messages: {e}, proceeding without RAG"
            )
            similar_messages = []

        logger.info(
            f"🔍 Generating response for query: {query[:50]}... using conversation context + {len(similar_messages)} style examples"
        )

        # Pass conversation history for context-aware responses
        response = await self.completion_service.generate_response(
            query=query,
            similar_messages=similar_messages,
            user_id=self.target_user_id,
            is_dm=is_dm,
            conversation_history=conversation_history,
            requester_display_name=requester_display_name,
        )

        return response, query_embedding

    async def _store_channel_interaction_memory(self, event):
        """Store memory of channel interactions for context"""
        try:
            user_id = event.get("user")
            message_text = event.get("text", "")

            if not user_id or not message_text.strip():
                return

            # Note: Channel interaction memory is now handled by conversation sessions
            # This method is kept for backwards compatibility but no longer stores data

            logger.debug(
                f"Channel interaction noted for user {user_id} (handled by conversation sessions)"
            )

        except Exception as e:
            logger.error(f"Error processing channel interaction: {e}")

    async def check_scraping_completeness_simple(
        self, channel_id: str
    ) -> Dict[str, Any]:
        """Simple completeness check - see if there are older messages than what we have stored"""
        try:
            if not self.target_user_id:
                return {
                    "complete": False,
                    "reason": "Target user ID not configured",
                    "needs_scraping": True,
                }

            # Note: Completeness checking disabled as we no longer use vector store for scraping
            # Conversation sessions handle memory automatically
            oldest_stored = None

            if not oldest_stored:
                return {
                    "complete": False,
                    "reason": "No messages stored yet",
                    "oldest_stored": None,
                    "needs_scraping": True,
                }

            # Check if there are any messages from the target user older than our oldest stored message
            oldest_stored_ts = str(oldest_stored["timestamp"])

            try:
                # Look for messages older than our oldest stored message
                older_check = await self.app.client.conversations_history(
                    channel=channel_id,
                    latest=oldest_stored_ts,  # Messages before our oldest
                    limit=50,  # Check a reasonable batch
                    inclusive=False,  # Don't include the boundary message
                )

                if not older_check.get("messages"):
                    return {
                        "complete": True,
                        "reason": "No older messages exist in channel",
                        "oldest_stored": oldest_stored["timestamp"],
                        "needs_scraping": False,
                    }

                # Check if any of these older messages are from our target user
                messages_list: List[Dict[str, Any]] = older_check.get("messages", [])
                target_user_older_messages: List[Dict[str, Any]] = [
                    msg
                    for msg in messages_list
                    if msg.get("user") == self.target_user_id
                    and msg.get("text")
                    and not msg.get("subtype")
                ]

                if target_user_older_messages:
                    return {
                        "complete": False,
                        "reason": f"Found {len(target_user_older_messages)} older messages from target user",
                        "oldest_stored": oldest_stored["timestamp"],
                        "needs_scraping": True,
                    }
                else:
                    return {
                        "complete": True,
                        "reason": "No older messages from target user found",
                        "oldest_stored": oldest_stored["timestamp"],
                        "needs_scraping": False,
                    }

            except Exception as e:
                error_str = str(e).lower()
                if "ratelimited" in error_str or "429" in error_str:
                    logger.warning("Hit rate limit during completeness check")
                    return {
                        "complete": False,
                        "reason": "Rate limited during check - assuming incomplete",
                        "oldest_stored": oldest_stored["timestamp"],
                        "needs_scraping": True,
                    }
                else:
                    logger.warning(f"API error during completeness check: {e}")
                    return {
                        "complete": False,
                        "reason": f"API error during check: {e}",
                        "oldest_stored": oldest_stored["timestamp"],
                        "needs_scraping": True,
                    }

        except Exception as e:
            logger.error(f"Error in simple completeness check: {e}")
            return {
                "complete": False,
                "reason": f"Check failed: {e}",
                "oldest_stored": None,
                "needs_scraping": True,
            }

    async def check_scraping_completeness(self, channel_id: str) -> Dict[str, Any]:
        """Check if we have scraped all available message history for the target user in a channel"""
        try:
            if not self.target_user_id:
                return {
                    "complete": False,
                    "reason": "Target user ID not configured",
                    "needs_scraping": True,
                }

            # Note: Completeness checking disabled as we no longer use vector store for scraping
            # Conversation sessions handle memory automatically
            oldest_stored = None

            if not oldest_stored:
                return {
                    "complete": False,
                    "reason": "No messages stored yet",
                    "oldest_stored": None,
                    "oldest_available": None,
                }

            # Fetch the very oldest message in the channel (limit=1, oldest possible)
            try:
                # Get the oldest message in the entire channel history
                oldest_result = await self.app.client.conversations_history(
                    channel=channel_id,
                    limit=1,
                    oldest="0",  # Start from Unix epoch
                    inclusive=True,
                )

                if not oldest_result.get("messages"):
                    return {
                        "complete": True,
                        "reason": "No messages in channel",
                        "oldest_stored": oldest_stored["timestamp"],
                        "oldest_available": None,
                    }

                # Find the oldest message from the target user in the channel
                # We need to paginate through to find their first message
                target_user_oldest = None
                cursor = None
                iterations = 0
                max_iterations = 50  # Limit to prevent infinite loops in huge channels

                logger.info(
                    f"🔍 Searching for oldest message from user {self.target_user_id} in channel {channel_id}"
                )

                while iterations < max_iterations:
                    kwargs = {
                        "channel": channel_id,
                        "limit": 15,  # Use recommended batch size for rate limits
                        "oldest": "0",
                        "inclusive": True,
                    }
                    if cursor:
                        kwargs["cursor"] = cursor

                    try:
                        batch_result = await self.app.client.conversations_history(
                            **kwargs  # type: ignore[arg-type]
                        )
                    except Exception as e:
                        error_str = str(e).lower()
                        if (
                            "ratelimited" in error_str
                            or "rate_limited" in error_str
                            or "429" in error_str
                        ):
                            logger.warning(
                                "Hit rate limit during completeness check, waiting 65 seconds..."
                            )
                            await asyncio.sleep(65)
                            continue
                        else:
                            # For other errors, return incomplete status rather than failing
                            logger.warning(f"API error during completeness check: {e}")
                            return {
                                "complete": False,
                                "reason": f"API error prevented full check: {e}",
                                "oldest_stored": oldest_stored["timestamp"],
                                "oldest_available": None,
                            }

                    messages: List[Dict[str, Any]] = batch_result.get("messages", [])

                    # Look for target user messages in this batch
                    target_messages = [
                        msg
                        for msg in messages
                        if msg.get("user") == self.target_user_id
                        and msg.get("text")
                        and not msg.get("subtype")
                    ]

                    if target_messages:
                        # Found some target user messages, keep the oldest one
                        batch_oldest = min(
                            target_messages, key=lambda m: float(m["ts"])
                        )
                        if not target_user_oldest or float(batch_oldest["ts"]) < float(
                            target_user_oldest["ts"]
                        ):
                            target_user_oldest = batch_oldest

                    # Check if we should continue
                    if not batch_result.get("has_more") or not batch_result.get(  # type: ignore[call-overload]
                        "response_metadata", {}
                    ).get("next_cursor"):
                        break

                    cursor = batch_result["response_metadata"]["next_cursor"]
                    iterations += 1

                    # Log progress every 10 iterations
                    if iterations % 10 == 0:
                        logger.info(
                            f"🔍 Completeness check progress: {iterations}/{max_iterations} iterations, oldest found: {target_user_oldest['ts'] if target_user_oldest else 'None'}"
                        )

                    # Respect rate limits - same as scraping
                    await asyncio.sleep(65)

                # Handle timeout case
                if iterations >= max_iterations:
                    logger.warning(
                        f"⏰ Completeness check timed out after {max_iterations} iterations"
                    )
                    if target_user_oldest:
                        # We found something, but may not be the absolute oldest
                        oldest_available_ts = float(target_user_oldest["ts"])
                        oldest_stored_ts = oldest_stored["timestamp"]
                        is_complete = abs(oldest_stored_ts - oldest_available_ts) <= 1.0

                        return {
                            "complete": is_complete,
                            "reason": (
                                f"Partial check completed (timeout after {max_iterations} iterations)"
                                if is_complete
                                else "May be missing older messages (check timed out)"
                            ),
                            "oldest_stored": oldest_stored_ts,
                            "oldest_available": oldest_available_ts,
                            "oldest_stored_preview": oldest_stored["message"][:50]
                            + "...",
                            "oldest_available_preview": target_user_oldest["text"][:50]
                            + "...",
                            "note": f"Check stopped after {max_iterations} API calls to respect rate limits",
                        }
                    else:
                        return {
                            "complete": False,
                            "reason": f"Check timed out after {max_iterations} iterations without finding target user messages",
                            "oldest_stored": oldest_stored["timestamp"],
                            "oldest_available": None,
                            "note": "May need manual verification for very large channels",
                        }

                if not target_user_oldest:
                    return {
                        "complete": True,
                        "reason": "Target user has no messages in this channel",
                        "oldest_stored": oldest_stored["timestamp"],
                        "oldest_available": None,
                    }

                # Compare timestamps
                oldest_available_ts = float(target_user_oldest["ts"])
                oldest_stored_ts = oldest_stored["timestamp"]

                # Consider complete if we have the oldest message (within 1 second tolerance)
                is_complete = abs(oldest_stored_ts - oldest_available_ts) <= 1.0

                return {
                    "complete": is_complete,
                    "reason": (
                        "Comparison completed"
                        if is_complete
                        else "Missing older messages"
                    ),
                    "oldest_stored": oldest_stored_ts,
                    "oldest_available": oldest_available_ts,
                    "oldest_stored_preview": oldest_stored["message"][:50] + "...",
                    "oldest_available_preview": target_user_oldest["text"][:50] + "...",
                }

            except Exception as e:
                logger.error(f"Error checking oldest message in channel: {e}")
                return {
                    "complete": False,
                    "reason": f"Error checking channel history: {e}",
                    "oldest_stored": oldest_stored["timestamp"],
                    "oldest_available": None,
                }

        except Exception as e:
            logger.error(f"Error checking scraping completeness: {e}")
            return {
                "complete": False,
                "reason": f"Error: {e}",
                "oldest_stored": None,
                "oldest_available": None,
            }

    async def scrape_channel_history(
        self, channel_identifier: str, days_back: Optional[int] = None
    ) -> None:
        if not self.target_user_id:
            logger.error("Target user ID not configured")
            return

        try:
            # Check if it's already a channel ID (starts with C)
            if channel_identifier.startswith("C"):
                channel_id = channel_identifier
                logger.info(f"Using channel ID directly: {channel_id}")
            else:
                # Look up by name
                channels_result = await self.app.client.conversations_list()
                channel_id = None

                for channel in channels_result["channels"]:
                    if channel["name"] == channel_identifier:
                        channel_id = channel["id"]
                        break

                if not channel_id:
                    logger.warning(f"Channel {channel_identifier} not found")
                    return

            # Check for existing scraping state
            assert channel_id is not None  # Should be resolved by this point
            # Note: Scraping state management disabled as vector store is no longer used
            cursor = None
            oldest = None
            if days_back:
                oldest = str((datetime.now() - timedelta(days=days_back)).timestamp())

            total_messages_fetched = 0
            total_target_messages_processed = 0
            skipped_existing_messages = 0

            if cursor:
                logger.info(
                    f"Resuming scrape from saved position for channel {channel_id}"
                )
            else:
                logger.info(f"Starting fresh scrape for channel {channel_id}")

            logger.info(
                f"📝 Starting to scrape and process channel history for {channel_id}"
            )
            update_stats(
                last_scrape_started=datetime.now().isoformat(),
                current_scraping_status="scraping",
            )

            while True:
                # Use recommended batch size for non-Marketplace apps (max 15)
                batch_limit = int(os.getenv("SLACK_BATCH_SIZE", "15"))

                kwargs = {
                    "channel": channel_id,
                    "limit": batch_limit,
                    "inclusive": True,  # Include boundary messages for better continuity
                }
                if oldest:
                    kwargs["oldest"] = oldest
                if cursor:
                    kwargs["cursor"] = cursor

                try:
                    messages_result = await self.app.client.conversations_history(
                        **kwargs  # type: ignore[arg-type]
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if "ratelimited" in error_str or "rate_limited" in error_str:
                        # Handle new stricter rate limits for non-Marketplace apps (1 req/min)
                        logger.warning(
                            "Hit rate limit (1 req/min), waiting 65 seconds..."
                        )
                        await asyncio.sleep(65)  # Wait a bit longer than 60s for safety
                        continue
                    elif "429" in error_str:
                        logger.warning("HTTP 429 - Rate limited, waiting 65 seconds...")
                        await asyncio.sleep(65)
                        continue
                    else:
                        raise e

                messages: List[Dict[str, Any]] = messages_result.get("messages", [])
                total_messages_fetched += len(messages)

                # Log pagination info for debugging
                has_more = messages_result.get("has_more", False)
                next_cursor = messages_result.get("response_metadata", {}).get(  # type: ignore[call-overload]
                    "next_cursor"
                )
                logger.debug(
                    f"📄 Fetched {len(messages)} messages, has_more: {has_more}, cursor: {next_cursor[:20] + '...' if next_cursor else 'None'}"
                )

                # Process target user messages from this batch immediately
                # Skip messages with subtypes (system messages like channel_join, channel_leave, etc.)
                target_messages_in_batch = [
                    msg
                    for msg in messages
                    if msg.get("user") == self.target_user_id
                    and msg.get("text")
                    and not msg.get("subtype")
                ]

                if target_messages_in_batch:
                    logger.info(
                        f"Processing {len(target_messages_in_batch)} target messages from batch of {len(messages)}"
                    )

                    # Process in smaller sub-batches to avoid overwhelming APIs
                    sub_batch_size = 3
                    for i in range(0, len(target_messages_in_batch), sub_batch_size):
                        sub_batch = target_messages_in_batch[i : i + sub_batch_size]

                        for message in sub_batch:
                            # Note: Message existence checking disabled as vector store is no longer used
                            # All messages will be processed

                            try:
                                await self._process_target_user_message(
                                    {
                                        "text": message["text"],
                                        "channel": channel_id,
                                        "ts": message["ts"],
                                    }
                                )
                                total_target_messages_processed += 1
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                                continue

                        # Small delay between sub-batches
                        if i + sub_batch_size < len(target_messages_in_batch):
                            await asyncio.sleep(1)

                logger.info(
                    f"Batch complete: {total_target_messages_processed} processed, {skipped_existing_messages} skipped, from {total_messages_fetched} total messages"
                )

                # Note: Scraping state saving disabled as vector store is no longer used
                pass

                # Improved pagination: use has_more AND cursor presence
                if not has_more or not next_cursor:
                    logger.info(
                        f"📄 Pagination complete: has_more={has_more}, cursor_present={bool(next_cursor)}"
                    )
                    break

                cursor = next_cursor

                # Respect 1 request per minute rate limit for non-Marketplace apps
                delay = int(os.getenv("SLACK_REQUEST_DELAY", "65"))
                logger.info(
                    f"⏳ Waiting {delay} seconds before next API request (1 req/min rate limit)..."
                )
                await asyncio.sleep(delay)

            logger.info(
                f"✅ Scraping complete! Processed {total_target_messages_processed} new messages, skipped {skipped_existing_messages} existing messages, from {total_messages_fetched} total messages"
            )

            # Note: Scraping state clearing disabled as vector store is no longer used
            pass

            update_stats(
                last_scrape_completed=datetime.now().isoformat(),
                current_scraping_status="idle",
            )

        except Exception as e:
            error_msg = f"Error scraping channel {channel_identifier}: {e}"
            logger.error(error_msg)
            add_error(error_msg)
            update_stats(current_scraping_status="error")

    async def start(self):
        logger.info("Starting Slack bot...")
        update_stats(bot_started_at=datetime.now().isoformat())

        # Start Socket Mode handler first to enable real-time event processing
        await self.handler.start_async()

    async def check_and_start_scraping_if_needed(self):
        """Check completeness for each channel and only start scraping if needed"""
        if not self.scrape_channels or not self.target_user_id:
            logger.info("No channels configured for scraping")
            return

        logger.info(f"📋 Checking completeness for channels: {self.scrape_channels}")

        for channel in self.scrape_channels:
            if not channel.strip():
                continue

            channel_name = channel.strip()
            try:
                # Resolve channel name to ID if needed
                if channel_name.startswith("C"):
                    channel_id = channel_name
                    logger.info(f"Using channel ID directly: {channel_id}")
                else:
                    channels_result = await self.app.client.conversations_list()
                    channel_id = None
                    for ch in channels_result["channels"]:
                        if ch["name"] == channel_name:
                            channel_id = ch["id"]
                            break

                    if not channel_id:
                        logger.warning(f"Channel {channel_name} not found, skipping")
                        continue

                # Check if we have complete history for this channel using simple check
                assert (
                    channel_id is not None
                )  # Help mypy understand the None check above
                logger.info(
                    f"🔍 Checking completeness for channel {channel_name} ({channel_id})"
                )
                completeness_check = await self.check_scraping_completeness_simple(
                    channel_id
                )

                if not completeness_check.get("needs_scraping", True):
                    logger.info(
                        f"✅ Channel {channel_name} already has complete message history for user {self.target_user_id}"
                    )
                    logger.info(f"📊 Reason: {completeness_check['reason']}")
                    if completeness_check.get("oldest_stored"):
                        oldest_date = datetime.fromtimestamp(
                            completeness_check["oldest_stored"]
                        ).isoformat()
                        logger.info(f"📅 Oldest message: {oldest_date}")
                else:
                    logger.info(
                        f"📝 Channel {channel_name} needs scraping: {completeness_check['reason']}"
                    )
                    if completeness_check.get("oldest_stored"):
                        oldest_date = datetime.fromtimestamp(
                            completeness_check["oldest_stored"]
                        ).isoformat()
                        logger.info(f"📅 Will resume from: {oldest_date}")

                    # Start scraping in background for this channel
                    logger.info(f"🚀 Starting background scraping for {channel_name}")
                    asyncio.create_task(self.scrape_channel_history(channel_name))

                # Add delay between channel checks to avoid rate limits
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(
                    f"Error checking completeness for channel {channel_name}: {e}"
                )
                # If we can't check completeness, err on the side of scraping
                logger.info(
                    f"🔄 Starting scraping for {channel_name} due to check error"
                )
                asyncio.create_task(self.scrape_channel_history(channel_name))

                # Add delay after error too
                await asyncio.sleep(2)

    async def start_scraping(self):
        """Start background scraping after checking completeness"""
        await self.check_and_start_scraping_if_needed()

    async def stop(self):
        logger.info("Stopping Slack bot...")
        await self.handler.close_async()
        await self.conversation_sessions.close()
        await self.user_manager.close()
