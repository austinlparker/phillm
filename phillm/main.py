from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

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

    # Initialize vector index
    await slack_bot.vector_store.initialize_index()

    # Start bot in background - don't wait for initial scraping
    import asyncio

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


if __name__ == "__main__":
    uvicorn.run("phillm.main:app", host="0.0.0.0", port=3000, reload=True)
