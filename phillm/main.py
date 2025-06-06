from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
from loguru import logger
from typing import AsyncGenerator, Dict, Any

from phillm.slack.bot import SlackBot
from phillm.api.routes import router
from phillm.api.debug import debug_router
from phillm.telemetry import telemetry

load_dotenv()

slack_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize OpenTelemetry only in the worker process
    telemetry.setup_telemetry()

    # Startup
    global slack_bot
    slack_bot = SlackBot()

    # Initialize conversation sessions and bot
    import asyncio

    logger.info("🚀 Starting PhiLLM application...")

    # Start bot in background - don't wait for initial scraping
    asyncio.create_task(slack_bot.start())
    # Start scraping in background after bot is connected
    asyncio.create_task(slack_bot.start_scraping())

    yield

    # Shutdown
    if slack_bot:
        await slack_bot.stop()


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
async def root_health_check() -> Dict[str, Any]:
    """Root-level health check for load balancer"""
    try:
        # Check conversation session manager health
        if slack_bot and slack_bot.conversation_sessions:
            return {"status": "healthy", "service": "PhiLLM", "redis": "connected"}
        else:
            from fastapi import HTTPException

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
