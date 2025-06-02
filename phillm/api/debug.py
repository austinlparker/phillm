from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import os
from datetime import datetime
from typing import Any, Dict

from phillm.vector.redis_vector_store import RedisVectorStore

debug_router = APIRouter()

# Global stats tracking
debug_stats: Dict[str, Any] = {
    "bot_started_at": None,
    "last_scrape_started": None,
    "last_scrape_completed": None,
    "total_messages_processed": 0,
    "current_scraping_status": "idle",
    "last_api_call": None,
    "errors": [],
    "channels_scraped": [],
    "dm_conversations": 0,
    "last_dm_received": None,
    "mention_conversations": 0,
    "last_mention_received": None,
}


def update_stats(**kwargs: Any) -> None:
    """Update debug stats"""
    for key, value in kwargs.items():
        if key in debug_stats:
            if (
                key
                in [
                    "total_messages_processed",
                    "dm_conversations",
                    "mention_conversations",
                ]
                and isinstance(value, int)
                and value == 1
            ):
                # Increment counter
                current_val = debug_stats[key]
                if isinstance(current_val, int):
                    debug_stats[key] = current_val + value
            else:
                debug_stats[key] = value


def add_error(error_msg: str) -> None:
    """Add error to debug log"""
    errors_list = debug_stats["errors"]
    if isinstance(errors_list, list):
        errors_list.append(
            {"timestamp": datetime.now().isoformat(), "error": str(error_msg)}
        )
        # Keep only last 10 errors
        if len(errors_list) > 10:
            debug_stats["errors"] = errors_list[-10:]


@debug_router.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request) -> str:
    """Debug dashboard showing bot status and stats"""

    # Get Redis stats
    try:
        vector_store = RedisVectorStore()
        user_id = os.getenv("TARGET_USER_ID", "unknown")
        channel_id = (
            os.getenv("SCRAPE_CHANNELS", "").split(",")[0]
            if os.getenv("SCRAPE_CHANNELS")
            else "unknown"
        )

        message_count = await vector_store.get_user_message_count(user_id)
        recent_messages = await vector_store.get_recent_messages(user_id, limit=5)
        scrape_state = await vector_store.get_scrape_state(channel_id)
        redis_status = "✅ Connected"
        await vector_store.close()
    except Exception as e:
        message_count = 0
        recent_messages = []
        scrape_state = {"cursor": None, "last_message_ts": None, "updated_at": None}
        redis_status = f"❌ Error: {str(e)}"

    # Get environment info
    env_info = {
        "TARGET_USER_ID": os.getenv("TARGET_USER_ID", "Not set"),
        "SCRAPE_CHANNELS": os.getenv("SCRAPE_CHANNELS", "Not set"),
        "OPENAI_API_KEY": "Set" if os.getenv("OPENAI_API_KEY") else "Not set",
        "SLACK_BOT_TOKEN": "Set" if os.getenv("SLACK_BOT_TOKEN") else "Not set",
        "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
    }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PhiLLM Debug Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
            .metric h4 {{ margin: 0 0 10px 0; color: #333; }}
            .metric p {{ margin: 0; font-size: 18px; font-weight: bold; color: #007bff; }}
            .error {{ border-left-color: #dc3545; }}
            .error p {{ color: #dc3545; }}
            .success {{ border-left-color: #28a745; }}
            .success p {{ color: #28a745; }}
            .log {{ background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }}
            .refresh-btn {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; float: right; }}
            .refresh-btn:hover {{ background: #0056b3; }}
            h1, h2 {{ color: #333; }}
            .timestamp {{ color: #666; font-size: 12px; }}
        </style>
        <script>
            function refreshPage() {{ window.location.reload(); }}
            setInterval(refreshPage, 30000); // Auto-refresh every 30 seconds
            
            async function testChat() {{
                const message = prompt("Enter a test message for the AI twin:");
                if (!message) return;
                
                try {{
                    const response = await fetch('/api/v1/chat', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{message: message}})
                    }});
                    
                    if (response.ok) {{
                        const data = await response.json();
                        alert('AI Twin Response:\\n\\n' + data.response);
                    }} else {{
                        const error = await response.json();
                        alert('Error: ' + error.detail);
                    }}
                }} catch (e) {{
                    alert('Error: ' + e.message);
                }}
            }}
            
            async function clearStore(storeType, userId) {{
                const confirmMsg = `Are you sure you want to clear the ${{storeType}} store for user ${{userId}}? This action cannot be undone.`;
                if (!confirm(confirmMsg)) return;
                
                try {{
                    const endpoint = storeType === 'all' 
                        ? `/debug/stores/clear-all/${{userId}}`
                        : storeType === 'cache' 
                            ? '/debug/stores/cache/clear'
                            : `/debug/stores/${{storeType}}/clear/${{userId}}`;
                    
                    const response = await fetch(endpoint, {{ method: 'POST' }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        alert(`Success: ${{data.message}}`);
                        refreshPage();
                    }} else {{
                        alert(`Error: ${{data.error}}`);
                    }}
                }} catch (e) {{
                    alert('Error: ' + e.message);
                }}
            }}
            
            async function resetScraping(channelId) {{
                if (!confirm(`Reset scraping state for channel ${{channelId}}?`)) return;
                
                try {{
                    const response = await fetch(`/debug/stores/scraping/reset/${{channelId}}`, {{ method: 'POST' }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        alert(`Success: ${{data.message}}`);
                        refreshPage();
                    }} else {{
                        alert(`Error: ${{data.error}}`);
                    }}
                }} catch (e) {{
                    alert('Error: ' + e.message);
                }}
            }}
            
            async function getStoresOverview() {{
                try {{
                    const response = await fetch('/debug/stores/overview');
                    const data = await response.json();
                    
                    if (data.error) {{
                        alert('Error: ' + data.error);
                    }} else {{
                        const overview = JSON.stringify(data, null, 2);
                        const popup = window.open('', '_blank', 'width=800,height=600,scrollbars=yes');
                        popup.document.write('<pre>' + overview + '</pre>');
                    }}
                }} catch (e) {{
                    alert('Error: ' + e.message);
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🤖 PhiLLM Debug Dashboard</h1>
            <button class="refresh-btn" onclick="refreshPage()">🔄 Refresh</button>
            <div style="clear: both;"></div>
            
            <div class="card">
                <h2>📊 Current Status</h2>
                <div class="status-grid">
                    <div class="metric {"success" if redis_status.startswith("✅") else "error"}">
                        <h4>Redis Connection</h4>
                        <p>{redis_status}</p>
                    </div>
                    <div class="metric">
                        <h4>Messages Stored</h4>
                        <p>{message_count:,} messages</p>
                    </div>
                    <div class="metric">
                        <h4>Scraping Status</h4>
                        <p>{str(debug_stats["current_scraping_status"]).title()}</p>
                    </div>
                    <div class="metric">
                        <h4>Total Processed</h4>
                        <p>{debug_stats["total_messages_processed"]:,} messages</p>
                    </div>
                    <div class="metric {"success" if scrape_state["cursor"] else ""}">
                        <h4>Scraping Progress</h4>
                        <p>{"✅ Can Resume" if scrape_state["cursor"] else "🔄 Complete/Fresh"}</p>
                    </div>
                    <div class="metric">
                        <h4>DM Conversations</h4>
                        <p>{debug_stats["dm_conversations"]:,} messages</p>
                    </div>
                    <div class="metric">
                        <h4>@ Mention Responses</h4>
                        <p>{debug_stats["mention_conversations"]:,} messages</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>⚙️ Configuration</h2>
                <div class="status-grid">
                    <div class="metric">
                        <h4>Target User ID</h4>
                        <p>{env_info["TARGET_USER_ID"]}</p>
                    </div>
                    <div class="metric">
                        <h4>Scrape Channels</h4>
                        <p>{env_info["SCRAPE_CHANNELS"]}</p>
                    </div>
                    <div class="metric {"success" if env_info["OPENAI_API_KEY"] == "Set" else "error"}">
                        <h4>OpenAI API Key</h4>
                        <p>{env_info["OPENAI_API_KEY"]}</p>
                    </div>
                    <div class="metric {"success" if env_info["SLACK_BOT_TOKEN"] == "Set" else "error"}">
                        <h4>Slack Bot Token</h4>
                        <p>{env_info["SLACK_BOT_TOKEN"]}</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📅 Timeline</h2>
                <div class="log">
                    <div><strong>Bot Started:</strong> {debug_stats["bot_started_at"] or "Not started"}</div>
                    <div><strong>Last Scrape Started:</strong> {debug_stats["last_scrape_started"] or "Never"}</div>
                    <div><strong>Last Scrape Completed:</strong> {debug_stats["last_scrape_completed"] or "Never"}</div>
                    <div><strong>Last API Call:</strong> {debug_stats["last_api_call"] or "Never"}</div>
                    <div><strong>Last DM Received:</strong> {debug_stats["last_dm_received"] or "Never"}</div>
                    <div><strong>Last @ Mention Received:</strong> {debug_stats["last_mention_received"] or "Never"}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>💬 Recent Messages (Last 5)</h2>
                <div class="log">
    """

    if recent_messages:
        for msg in recent_messages:
            timestamp = datetime.fromtimestamp(float(msg.get("timestamp", 0))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            message_preview = msg.get("message", "")[:100] + (
                "..." if len(msg.get("message", "")) > 100 else ""
            )
            html_content += f"<div><span class='timestamp'>[{timestamp}]</span> {message_preview}</div>"
    else:
        html_content += "<div>No messages found</div>"

    html_content += """
                </div>
            </div>
            
            <div class="card">
                <h2>❌ Recent Errors</h2>
                <div class="log">
    """

    errors_list = debug_stats["errors"]
    if isinstance(errors_list, list) and errors_list:
        for error in reversed(errors_list):  # Show newest first
            html_content += f"<div><span class='timestamp'>[{error['timestamp']}]</span> {error['error']}</div>"
    else:
        html_content += "<div>No recent errors</div>"

    html_content += f"""
                </div>
            </div>
            
            <div class="card">
                <h2>🔧 Actions</h2>
                <div class="status-grid">
                    <div class="metric">
                        <h4>API Health Check</h4>
                        <p><a href="/api/v1/health" target="_blank">Check API Status</a></p>
                    </div>
                    <div class="metric">
                        <h4>Message Count</h4>
                        <p><a href="/api/v1/users/{env_info["TARGET_USER_ID"]}/messages/count" target="_blank">Get Count</a></p>
                    </div>
                    <div class="metric">
                        <h4>Recent Messages</h4>
                        <p><a href="/api/v1/users/{env_info["TARGET_USER_ID"]}/messages/recent" target="_blank">View JSON</a></p>
                    </div>
                    <div class="metric">
                        <h4>API Documentation</h4>
                        <p><a href="/docs" target="_blank">OpenAPI Docs</a></p>
                    </div>
                    <div class="metric">
                        <h4>Scraping Status</h4>
                        <p><a href="/api/v1/scraping/status/{channel_id}" target="_blank">Check Progress</a></p>
                    </div>
                    <div class="metric">
                        <h4>Reset Scraping</h4>
                        <p><button onclick="resetScraping('{channel_id}')">🔄 Reset State</button></p>
                    </div>
                    <div class="metric">
                        <h4>Test Chat API</h4>
                        <p><button onclick="testChat()">💬 Send Test Message</button></p>
                    </div>
                    <div class="metric">
                        <h4>Stores Overview</h4>
                        <p><button onclick="getStoresOverview()">📊 View All Stats</button></p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🗂️ Store Management</h2>
                <div style="margin-bottom: 15px;">
                    <strong>⚠️ Warning:</strong> These actions will permanently delete data and cannot be undone!
                </div>
                <div class="status-grid">
                    <div class="metric error">
                        <h4>Clear Vector Store</h4>
                        <p><button onclick="clearStore('vector', '{env_info["TARGET_USER_ID"]}')">🗑️ Clear Messages</button></p>
                        <small>Deletes all scraped messages and embeddings</small>
                    </div>
                    <div class="metric error">
                        <h4>Clear Memory Store</h4>
                        <p><button onclick="clearStore('memory', '{env_info["TARGET_USER_ID"]}')">🧠 Clear Memories</button></p>
                        <small>Deletes all conversation memories</small>
                    </div>
                    <div class="metric error">
                        <h4>Clear User Cache</h4>
                        <p><button onclick="clearStore('cache', 'all')">👥 Clear Cache</button></p>
                        <small>Clears all cached user information</small>
                    </div>
                    <div class="metric error">
                        <h4>Clear ALL Stores</h4>
                        <p><button onclick="clearStore('all', '{env_info["TARGET_USER_ID"]}')" style="background: #dc3545;">💥 CLEAR ALL</button></p>
                        <small>⚠️ Deletes EVERYTHING for this user</small>
                    </div>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>🤖 PhiLLM Slack AI Twin Debug Dashboard</p>
                <p class="timestamp">Auto-refreshes every 30 seconds | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


@debug_router.get("/debug/stats")
async def get_debug_stats() -> Dict[str, Any]:
    """Get debug stats as JSON"""
    return debug_stats


@debug_router.get("/debug/stores/overview")
async def get_stores_overview() -> Dict[str, Any]:
    """Get overview of all stores and their statistics"""
    try:
        from phillm.memory import ConversationMemory
        from phillm.user import UserManager
        from phillm.ai.embeddings import EmbeddingService

        user_id = os.getenv("TARGET_USER_ID", "unknown")

        # Vector store stats
        vector_store = RedisVectorStore()
        vector_stats = {
            "total_messages": await vector_store.get_user_message_count(user_id),
            "recent_messages_count": len(
                await vector_store.get_recent_messages(user_id, limit=10)
            ),
            "status": "connected",
        }
        await vector_store.close()

        # Memory store stats
        memory = ConversationMemory()
        embedding_service = EmbeddingService()
        memory.set_embedding_service(embedding_service)
        memory_stats = await memory.get_memory_stats(user_id)
        await memory.close()

        # User cache stats
        user_manager = UserManager()
        user_cache_stats = await user_manager.get_cache_stats()
        await user_manager.close()

        return {
            "vector_store": vector_stats,
            "memory_store": memory_stats,
            "user_cache": user_cache_stats,
            "target_user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


@debug_router.post("/debug/stores/vector/clear/{user_id}")
async def clear_vector_store(user_id: str) -> Dict[str, Any]:
    """Clear all vector store data for a specific user"""
    try:
        vector_store = RedisVectorStore()
        deleted_count = await vector_store.delete_user_messages(user_id)
        await vector_store.close()

        add_error(
            f"Vector store cleared for user {user_id}: {deleted_count} messages deleted"
        )

        return {
            "success": True,
            "user_id": user_id,
            "deleted_messages": deleted_count,
            "message": f"Cleared {deleted_count} messages from vector store",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        add_error(f"Error clearing vector store for {user_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@debug_router.post("/debug/stores/memory/clear/{user_id}")
async def clear_memory_store(user_id: str) -> Dict[str, Any]:
    """Clear all memory store data for a specific user"""
    try:
        from phillm.memory import ConversationMemory
        from phillm.ai.embeddings import EmbeddingService

        memory = ConversationMemory()
        embedding_service = EmbeddingService()
        memory.set_embedding_service(embedding_service)

        # Get current memory count before clearing
        memories = await memory._get_user_memories(user_id)
        initial_count = len(memories)

        # Clear memories by deleting from Redis
        for mem in memories:
            await memory._delete_memory(mem.memory_id, user_id)

        await memory.close()

        add_error(
            f"Memory store cleared for user {user_id}: {initial_count} memories deleted"
        )

        return {
            "success": True,
            "user_id": user_id,
            "deleted_memories": initial_count,
            "message": f"Cleared {initial_count} memories from memory store",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        add_error(f"Error clearing memory store for {user_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@debug_router.post("/debug/stores/cache/clear")
async def clear_user_cache() -> Dict[str, Any]:
    """Clear all user cache data"""
    try:
        from phillm.user import UserManager

        user_manager = UserManager()

        # Get all cache keys
        cache_keys = await user_manager.redis_client.keys("user_info:*")
        deleted_count = 0

        # Delete all cache entries
        for key in cache_keys:
            await user_manager.redis_client.delete(key)
            deleted_count += 1

        await user_manager.close()

        add_error(f"User cache cleared: {deleted_count} entries deleted")

        return {
            "success": True,
            "deleted_cache_entries": deleted_count,
            "message": f"Cleared {deleted_count} entries from user cache",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        add_error(f"Error clearing user cache: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@debug_router.post("/debug/stores/scraping/reset/{channel_id}")
async def reset_scraping_state(channel_id: str) -> Dict[str, Any]:
    """Reset scraping state for a channel"""
    try:
        vector_store = RedisVectorStore()
        await vector_store.clear_scrape_state(channel_id)
        await vector_store.close()

        add_error(f"Scraping state reset for channel {channel_id}")

        return {
            "success": True,
            "channel_id": channel_id,
            "message": f"Scraping state reset for channel {channel_id}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        add_error(f"Error resetting scraping state for {channel_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@debug_router.post("/debug/stores/clear-all/{user_id}")
async def clear_all_stores(user_id: str) -> Dict[str, Any]:
    """Clear all stores for a specific user (DANGEROUS!)"""
    try:
        results = {}

        # Clear vector store
        try:
            vector_store = RedisVectorStore()
            vector_deleted = await vector_store.delete_user_messages(user_id)
            await vector_store.close()
            results["vector_store"] = {"success": True, "deleted": vector_deleted}
        except Exception as e:
            results["vector_store"] = {"success": False, "error": str(e)}  # type: ignore[dict-item]

        # Clear memory store
        try:
            from phillm.memory import ConversationMemory
            from phillm.ai.embeddings import EmbeddingService

            memory = ConversationMemory()
            embedding_service = EmbeddingService()
            memory.set_embedding_service(embedding_service)

            memories = await memory._get_user_memories(user_id)
            for mem in memories:
                await memory._delete_memory(mem.memory_id, user_id)
            await memory.close()

            results["memory_store"] = {"success": True, "deleted": len(memories)}
        except Exception as e:
            results["memory_store"] = {"success": False, "error": str(e)}  # type: ignore[dict-item]

        # Clear user cache (specific user only)
        try:
            from phillm.user import UserManager

            user_manager = UserManager()
            await user_manager.invalidate_user_cache(user_id)
            await user_manager.close()
            results["user_cache"] = {"success": True, "deleted": 1}
        except Exception as e:
            results["user_cache"] = {"success": False, "error": str(e)}  # type: ignore[dict-item]

        add_error(f"ALL STORES CLEARED for user {user_id}")

        return {
            "success": True,
            "user_id": user_id,
            "results": results,
            "message": f"All stores cleared for user {user_id}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        add_error(f"Error clearing all stores for {user_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
