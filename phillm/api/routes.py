import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger

from phillm.vector.redis_vector_store import RedisVectorStore
from phillm.ai.completions import CompletionService

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    user_id: str
    limit: Optional[int] = 5


class QueryResponse(BaseModel):
    response: str
    similar_messages: List[Dict[str, Any]]


class MessageCountResponse(BaseModel):
    user_id: str
    message_count: int


class RecentMessagesResponse(BaseModel):
    user_id: str
    messages: List[Dict[str, Any]]


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    try:
        # Test Redis connection
        vector_store = RedisVectorStore()
        redis_healthy = await vector_store.health_check()
        await vector_store.close()

        if redis_healthy:
            return {"status": "healthy", "service": "PhiLLM", "redis": "connected"}
        else:
            raise HTTPException(
                status_code=503,
                detail={"status": "unhealthy", "service": "PhiLLM", "redis": "disconnected"}
            )
    except HTTPException:
        raise  # Re-raise HTTPException to preserve status code
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "service": "PhiLLM", "error": str(e)}
        )


@router.post("/query", response_model=QueryResponse)
async def query_ai_twin(request: QueryRequest) -> QueryResponse:
    try:
        vector_store = RedisVectorStore()
        completion_service = CompletionService()

        # Find similar messages
        similar_messages = await vector_store.find_similar_messages(
            request.query, user_id=request.user_id, limit=request.limit or 5
        )

        if not similar_messages:
            raise HTTPException(
                status_code=404, detail=f"No messages found for user {request.user_id}"
            )

        # Generate AI response using many-shot learning
        response = await completion_service.generate_response(
            query=request.query,
            similar_messages=similar_messages,
            user_id=request.user_id,
        )

        await vector_store.close()

        return QueryResponse(response=response, similar_messages=similar_messages)

    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/messages/count", response_model=MessageCountResponse)
async def get_user_message_count(user_id: str) -> MessageCountResponse:
    try:
        vector_store = RedisVectorStore()
        count = await vector_store.get_user_message_count(user_id)
        await vector_store.close()

        return MessageCountResponse(user_id=user_id, message_count=count)

    except Exception as e:
        logger.error(f"Error getting message count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/messages/recent", response_model=RecentMessagesResponse)
async def get_recent_messages(user_id: str, limit: int = 20) -> RecentMessagesResponse:
    try:
        vector_store = RedisVectorStore()
        messages = await vector_store.get_recent_messages(user_id, limit)
        await vector_store.close()

        return RecentMessagesResponse(user_id=user_id, messages=messages)

    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}/messages")
async def delete_user_messages(user_id: str) -> Dict[str, Any]:
    try:
        vector_store = RedisVectorStore()
        deleted_count = await vector_store.delete_user_messages(user_id)
        await vector_store.close()

        return {
            "user_id": user_id,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} messages for user {user_id}",
        }

    except Exception as e:
        logger.error(f"Error deleting user messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/status/{channel_id}")
async def get_scraping_status(channel_id: str) -> Dict[str, Any]:
    """Get scraping status for a channel"""
    try:
        vector_store = RedisVectorStore()
        scrape_state = await vector_store.get_scrape_state(channel_id)
        await vector_store.close()

        return {
            "channel_id": channel_id,
            "scrape_state": scrape_state,
            "has_saved_progress": scrape_state["cursor"] is not None,
        }

    except Exception as e:
        logger.error(f"Error getting scraping status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scraping/reset/{channel_id}")
async def reset_scraping_state(channel_id: str) -> Dict[str, Any]:
    """Reset scraping state for a channel (start from beginning)"""
    try:
        vector_store = RedisVectorStore()
        await vector_store.clear_scrape_state(channel_id)
        await vector_store.close()

        return {
            "channel_id": channel_id,
            "message": "Scraping state reset. Next scrape will start from the beginning.",
        }

    except Exception as e:
        logger.error(f"Error resetting scraping state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/completeness/{channel_id}")
async def check_scraping_completeness(channel_id: str, detailed: bool = False) -> Dict[str, Any]:
    """Check if we have scraped all available message history for the target user in a channel"""
    try:
        from phillm.slack.bot import SlackBot
        import os

        # We need the SlackBot instance to access the Slack API
        bot = SlackBot()
        target_user_id = os.getenv("TARGET_USER_ID")

        if not target_user_id:
            raise HTTPException(status_code=400, detail="TARGET_USER_ID not configured")

        # Use simple check by default, detailed only if requested
        if detailed:
            completeness_check = await bot.check_scraping_completeness(channel_id)
        else:
            completeness_check = await bot.check_scraping_completeness_simple(
                channel_id
            )

        # Add some additional context
        total_stored = await bot.vector_store.get_user_message_count(target_user_id)
        oldest_stored = await bot.vector_store.get_oldest_stored_message(
            target_user_id, channel_id
        )

        result = {
            **completeness_check,
            "target_user_id": target_user_id,
            "total_messages_stored": total_stored,
            "oldest_message_date": None,
            "check_type": "detailed" if detailed else "simple",
        }

        if oldest_stored:
            from datetime import datetime

            result["oldest_message_date"] = datetime.fromtimestamp(
                oldest_stored["timestamp"]
            ).isoformat()

        if not completeness_check.get("complete", True):
            result["recommendation"] = "Run scraping to collect more message history"

        return result

    except Exception as e:
        logger.error(f"Error checking scraping completeness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    is_dm: bool


@router.get("/debug/vector-search")
async def debug_vector_search() -> Dict[str, Any]:
    """Debug vector similarity search"""
    try:
        from phillm.slack.bot import SlackBot
        import os

        bot = SlackBot()
        target_user_id = os.getenv("TARGET_USER_ID")

        # Get total message count
        total_messages = await bot.vector_store.get_user_message_count(target_user_id or "")

        # Test search with different thresholds
        test_query = "mcdonalds"
        results = {}

        for threshold in [0.0, 0.3, 0.5, 0.7, 0.9]:
            similar = await bot.vector_store.find_similar_messages(
                test_query, user_id=target_user_id or "", limit=3, threshold=threshold
            )
            results[f"threshold_{threshold}"] = {
                "count": len(similar),
                "messages": [
                    {
                        "similarity": msg.get("similarity", 0),
                        "preview": msg["message"][:80] + "..."
                        if len(msg["message"]) > 80
                        else msg["message"],
                    }
                    for msg in similar
                ],
            }

        # Get a few recent messages to verify storage
        recent = await bot.vector_store.get_recent_messages(target_user_id or "", limit=5)

        return {
            "target_user_id": target_user_id,
            "total_stored_messages": total_messages,
            "test_query": test_query,
            "similarity_results": results,
            "recent_messages_sample": [
                {
                    "preview": msg["message"][:80] + "..."
                    if len(msg["message"]) > 80
                    else msg["message"],
                    "timestamp": msg.get("timestamp"),
                }
                for msg in recent
            ],
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@router.get("/slack/test-dm-access")
async def test_dm_access() -> Dict[str, Any]:
    """Test if bot can access DM conversations and check detailed permissions"""
    try:
        from phillm.slack.bot import SlackBot

        bot = SlackBot()

        results: Dict[str, Any] = {}

        # Try to list conversations (including DMs)
        try:
            conversations = await bot.app.client.conversations_list(types="im")
            ch: List[Dict[str, Any]] = conversations.get("channels", [])  # type: ignore[assignment]
            dm_count = len(ch)
            results["dm_conversations_found"] = dm_count
            results["dm_channels"] = [
                {
                    "id": channel.get("id"),
                    "user": channel.get("user"),
                    "created": channel.get("created"),
                }
                for channel in ch[:5]  # Show first 5
            ]
        except Exception as e:
            results["dm_list_error"] = str(e)

        # Try to get bot info and team info for scopes
        try:
            auth_info = await bot.app.client.auth_test()
            bot_user_id = auth_info.get("user_id")  # type: ignore[assignment]
            team_id = auth_info.get("team_id")  # type: ignore[assignment]
            results["bot_user_id"] = bot_user_id
            results["team_id"] = team_id

            # Try to get team info which sometimes includes scopes
            team_info = await bot.app.client.team_info()  # type: ignore[assignment]
            results["team_info"] = team_info
        except Exception as e:
            results["auth_error"] = str(e)

        # Test if we can actually read a DM conversation
        dm_channels_list = results.get("dm_channels")
        if dm_channels_list:
            try:
                dm_id = dm_channels_list[0]["id"]  # type: ignore[index]
                history = await bot.app.client.conversations_history(
                    channel=dm_id, limit=1
                )
                results["can_read_dm_history"] = True
                dm_history_sample = len(history.get("messages", []))  # type: ignore[assignment]
                results["dm_history_sample"] = dm_history_sample
            except Exception as e:
                results["dm_history_error"] = str(e)
                results["can_read_dm_history"] = False

        # Test sending a message to a DM (we'll catch the error to see what permission we're missing)
        if dm_channels_list:
            try:
                dm_id = dm_channels_list[0]["id"]  # type: ignore[index]
                # Don't actually send, just test the API call structure
                results["can_post_to_dm"] = "not_tested"  # We don't want to spam
            except Exception as e:
                results["dm_post_error"] = str(e)

        return results

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai_twin(request: ChatRequest) -> ChatResponse:
    """Chat with the AI twin directly via API (simulates DM)"""
    try:
        from phillm.memory import ConversationMemory
        from phillm.ai.embeddings import EmbeddingService

        completion_service = CompletionService()
        vector_store = RedisVectorStore()
        memory = ConversationMemory()
        embedding_service = EmbeddingService()
        memory.set_embedding_service(embedding_service)

        # Use target user or provided user ID
        user_id = request.user_id or os.getenv("TARGET_USER_ID")
        if not user_id:
            raise HTTPException(status_code=400, detail="No target user configured")

        # Get conversation context from memory
        conversation_context = await memory.get_conversation_context(user_id, limit=3)

        # Find similar messages for context
        similar_messages = await vector_store.find_similar_messages(
            request.message, user_id=user_id, limit=5
        )

        if not similar_messages:
            raise HTTPException(
                status_code=404, detail=f"No messages found for user {user_id}"
            )

        # Get requester's display name if available
        requester_display_name = None
        if request.user_id:
            from phillm.user import UserManager
            from phillm.slack.bot import SlackBot

            bot = SlackBot()
            user_manager = UserManager(bot.app.client)
            requester_display_name = await user_manager.get_user_display_name(
                request.user_id
            )
            await user_manager.close()

        # Generate AI response with memory context
        response = await completion_service.generate_response(
            query=request.message,
            similar_messages=similar_messages,
            user_id=user_id,
            is_dm=True,
            conversation_context=conversation_context,
            requester_display_name=requester_display_name,
        )

        # Store this interaction in memory (simulate API user)
        api_user_id = request.user_id or "api_user"
        await memory.store_dm_interaction(api_user_id, request.message, response, "api")

        await vector_store.close()
        await memory.close()

        return ChatResponse(response=response, is_dm=True)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{user_id}/stats")
async def get_user_memory_stats(user_id: str) -> Dict[str, Any]:
    """Get memory statistics for a user"""
    try:
        from phillm.memory import ConversationMemory

        memory = ConversationMemory()
        stats = await memory.get_memory_stats(user_id)
        await memory.close()

        return {"user_id": user_id, "memory_stats": stats}

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{user_id}/recall")
async def recall_user_memories(
    user_id: str, query: Optional[str] = None, limit: int = 10, min_relevance: float = 0.3
) -> Dict[str, Any]:
    """Recall memories for a user"""
    try:
        from phillm.memory import ConversationMemory
        from phillm.ai.embeddings import EmbeddingService

        memory = ConversationMemory()
        embedding_service = EmbeddingService()
        memory.set_embedding_service(embedding_service)

        memories = await memory.recall_memories(
            user_id=user_id, query=query, limit=limit, min_relevance=min_relevance
        )

        # Convert memories to serializable format
        memory_data = []
        for mem in memories:
            memory_data.append(
                {
                    "memory_id": mem.memory_id,
                    "memory_type": mem.memory_type.value,
                    "content": mem.content,
                    "context": mem.context,
                    "importance": mem.importance.value,
                    "timestamp": mem.timestamp,
                    "access_count": mem.access_count,
                    "last_accessed": mem.last_accessed,
                }
            )

        await memory.close()

        return {
            "user_id": user_id,
            "query": query,
            "memories_found": len(memory_data),
            "memories": memory_data,
        }

    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{user_id}/context")
async def get_conversation_context(user_id: str, limit: int = 5) -> Dict[str, Any]:
    """Get recent conversation context for a user"""
    try:
        from phillm.memory import ConversationMemory
        from phillm.ai.embeddings import EmbeddingService

        memory = ConversationMemory()
        embedding_service = EmbeddingService()
        memory.set_embedding_service(embedding_service)

        context = await memory.get_conversation_context(user_id, limit)
        await memory.close()

        return {"user_id": user_id, "context": context, "context_length": len(context)}

    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/info")
async def get_user_info(user_id: str) -> Dict[str, Any]:
    """Get user information including display name"""
    try:
        from phillm.user import UserManager
        from phillm.slack.bot import SlackBot

        # We need the SlackBot instance to access the Slack client
        bot = SlackBot()
        user_manager = UserManager(bot.app.client)

        user_info = await user_manager.get_user_info(user_id)
        await user_manager.close()

        return {"user_id": user_id, "user_info": user_info}

    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/cache/stats")
async def get_user_cache_stats() -> Dict[str, Any]:
    """Get user cache statistics"""
    try:
        from phillm.user import UserManager

        user_manager = UserManager()
        stats = await user_manager.get_cache_stats()
        await user_manager.close()

        return {"cache_stats": stats}

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/cache/invalidate")
async def invalidate_user_cache(user_id: str) -> Dict[str, Any]:
    """Invalidate cached user information"""
    try:
        from phillm.user import UserManager

        user_manager = UserManager()
        await user_manager.invalidate_user_cache(user_id)
        await user_manager.close()

        return {"user_id": user_id, "message": "User cache invalidated successfully"}

    except Exception as e:
        logger.error(f"Error invalidating user cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
