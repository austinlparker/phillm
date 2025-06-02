from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
from loguru import logger

from phillm.slack.bot import SlackBot
from phillm.api.routes import router
from phillm.api.debug import debug_router
from phillm.telemetry import telemetry

load_dotenv()

slack_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize OpenTelemetry only in the worker process
    telemetry.setup_telemetry()

    # Startup
    global slack_bot
    slack_bot = SlackBot()

    # Try to initialize vector index with retries
    import asyncio

    logger.info("🚀 Starting PhiLLM application...")

    async def initialize_with_retries():
        max_attempts = 10
        retry_delay = 5

        for attempt in range(max_attempts):
            try:
                logger.info(
                    f"Attempting to initialize Redis vector index (attempt {attempt + 1}/{max_attempts})"
                )
                await slack_bot.vector_store.initialize_index()
                logger.info("✅ Redis vector index initialized successfully")
                break
            except Exception as e:
                logger.error(f"Failed to initialize Redis (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        "❌ All Redis initialization attempts failed - application will be unhealthy"
                    )
                    # Don't raise - let the app start but be unhealthy
                    break

    # Initialize in background so app can start
    asyncio.create_task(initialize_with_retries())

    # Start bot in background - don't wait for initial scraping
    asyncio.create_task(slack_bot.start())
    # Start scraping in background after bot is connected
    asyncio.create_task(slack_bot.start_scraping())

    yield

    # Shutdown
    if slack_bot:
        await slack_bot.stop()
        await slack_bot.vector_store.close()
        await slack_bot.memory.close()
        await slack_bot.user_manager.close()


app = FastAPI(
    title="PhiLLM - Slack AI Twin",
    description="Create AI personas from Slack message history",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")
app.include_router(debug_router)


# Add root-level health check for ALB
@app.get("/health")
async def root_health_check():
    """Root-level health check for load balancer"""
    try:
        # Import here to avoid circular imports
        from phillm.vector.redis_vector_store import RedisVectorStore
        from fastapi import HTTPException

        vector_store = RedisVectorStore()
        redis_healthy = await vector_store.health_check()
        await vector_store.close()

        if redis_healthy:
            return {"status": "healthy", "service": "PhiLLM", "redis": "connected"}
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "service": "PhiLLM",
                    "redis": "disconnected",
                },
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "service": "PhiLLM", "error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run("phillm.main:app", host="0.0.0.0", port=3000, reload=True)
