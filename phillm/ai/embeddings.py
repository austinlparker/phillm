import os
import time
from typing import List
import openai
from loguru import logger
from redisvl.utils.vectorize import OpenAITextVectorizer
from phillm.telemetry import get_tracer, telemetry


class EmbeddingService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-large"

        # Initialize RedisVL OpenAI vectorizer for better integration
        self.vectorizer = OpenAITextVectorizer(
            model=self.model,
            api_config={
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        )

    async def create_embedding(self, text: str) -> List[float]:
        tracer = get_tracer()
        start_time = time.time()

        try:
            with tracer.start_as_current_span("create_embedding") as span:
                span.set_attribute("text.length", len(text))
                span.set_attribute("model", self.model)
                span.set_attribute("text.preview", text[:100])

                response = await self.client.embeddings.create(
                    model=self.model, input=text.strip()
                )

                embedding = response.data[0].embedding
                duration = time.time() - start_time

                span.set_attribute("embedding.dimension", len(embedding))
                span.set_attribute("duration_seconds", duration)

                # Record metrics
                telemetry.record_embedding_created(len(text), duration)

                logger.debug(f"Created embedding for text: {text[:50]}...")
                return embedding

        except Exception as e:
            duration = time.time() - start_time
            with tracer.start_as_current_span("create_embedding_error") as span:
                span.set_attribute("error", str(e))
                span.set_attribute("duration_seconds", duration)
            logger.error(f"Error creating embedding: {e}")
            raise

    async def create_embedding_with_vectorizer(self, text: str) -> List[float]:
        """Create embedding using RedisVL vectorizer (alternative method)"""
        tracer = get_tracer()
        start_time = time.time()

        try:
            with tracer.start_as_current_span("create_embedding_vectorizer") as span:
                span.set_attribute("text.length", len(text))
                span.set_attribute("model", self.model)
                span.set_attribute("text.preview", text[:100])

                # Use RedisVL vectorizer
                embedding = self.vectorizer.embed(text.strip())

                duration = time.time() - start_time
                span.set_attribute("embedding.dimension", len(embedding))
                span.set_attribute("duration_seconds", duration)

                # Record metrics
                telemetry.record_embedding_created(len(text), duration)

                logger.debug(
                    f"Created embedding with vectorizer for text: {text[:50]}..."
                )
                return embedding

        except Exception as e:
            duration = time.time() - start_time
            with tracer.start_as_current_span(
                "create_embedding_vectorizer_error"
            ) as span:
                span.set_attribute("error", str(e))
                span.set_attribute("duration_seconds", duration)
            logger.error(f"Error creating embedding with vectorizer: {e}")
            raise

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.model, input=[text.strip() for text in texts]
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Created {len(embeddings)} embeddings in batch")

            return embeddings

        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}")
            raise
