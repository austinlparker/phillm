from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger

from phillm.conversation import ConversationSessionManager
# from phillm.vector.redis_vector_store import RedisVectorStore

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    user_id: str
    limit: Optional[int] = 5


class QueryResponse(BaseModel):
    response: str
    similar_messages: List[Dict[str, Any]]


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    try:
        # Test conversation session manager connection
        session_manager = ConversationSessionManager()
        await session_manager._ensure_redis_connection()
        await session_manager.close()

        return {"status": "healthy", "service": "PhiLLM", "redis": "connected"}
    except HTTPException:
        raise  # Re-raise HTTPException to preserve status code
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "service": "PhiLLM", "error": str(e)},
        )


@router.post("/query", response_model=QueryResponse)
async def query_ai_twin(request: QueryRequest) -> QueryResponse:
    """DISABLED: This endpoint used the old vector store system"""
    # TODO: Reimplement using ConversationSessionManager
    raise HTTPException(
        status_code=503, detail="Endpoint temporarily disabled during system migration"
    )


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    is_dm: bool


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
        raise HTTPException(
            status_code=503,
            detail="Endpoint temporarily disabled during system migration",
        )
        # from phillm.memory import ConversationMemory
        # from phillm.ai.embeddings import EmbeddingService

        # completion_service = CompletionService()
        # vector_store = RedisVectorStore()
        # memory = ConversationMemory()
        # embedding_service = EmbeddingService()
        # memory.set_embedding_service(embedding_service)

        # # Use target user or provided user ID
        # user_id = request.user_id or os.getenv("TARGET_USER_ID")
        # if not user_id:
        #     raise HTTPException(status_code=400, detail="No target user configured")

        # # Get conversation context from memory (convert to message format)
        # conversation_context_str = await memory.get_conversation_context(
        #     user_id, limit=3
        # )

        # # Convert string context to message format for compatibility
        # conversation_history = []
        # if conversation_context_str and conversation_context_str.strip():
        #     # Simple conversion - treat as user message for now
        #     conversation_history = [
        #         {"role": "user", "content": conversation_context_str}
        #     ]

        # # Find similar messages for context
        # similar_messages = await vector_store.find_similar_messages(
        #     request.message, user_id=user_id, limit=5
        # )

        # if not similar_messages:
        #     raise HTTPException(
        #         status_code=404, detail=f"No messages found for user {user_id}"
        #     )

        # # Get requester's display name if available
        # requester_display_name = None
        # if request.user_id:
        #     from phillm.user import UserManager
        #     from phillm.slack.bot import SlackBot

        #     bot = SlackBot()
        #     user_manager = UserManager(bot.app.client)
        #     requester_display_name = await user_manager.get_user_display_name(
        #         request.user_id
        #     )
        #     await user_manager.close()

        # # Generate AI response with memory context
        # response = await completion_service.generate_response(
        #     query=request.message,
        #     similar_messages=similar_messages,
        #     user_id=user_id,
        #     is_dm=True,
        #     conversation_history=conversation_history,
        #     requester_display_name=requester_display_name,
        # )

        # # Store this interaction in memory (simulate API user)
        # api_user_id = request.user_id or "api_user"
        # await memory.store_dm_interaction(api_user_id, request.message, response, "api")

        # await vector_store.close()
        # await memory.close()

        # return ChatResponse(response=response, is_dm=True)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{user_id}/stats")
async def get_user_memory_stats(user_id: str) -> Dict[str, Any]:
    """Get memory statistics for a user"""
    try:
        raise HTTPException(
            status_code=503,
            detail="Endpoint temporarily disabled during system migration",
        )
        # from phillm.memory import ConversationMemory

        # memory = ConversationMemory()
        # stats = await memory.get_memory_stats(user_id)
        # await memory.close()

        # return {"user_id": user_id, "memory_stats": stats}

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
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
