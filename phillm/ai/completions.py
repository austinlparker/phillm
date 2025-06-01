import os
import time
from typing import Optional
import openai
from loguru import logger
from phillm.telemetry import get_tracer, telemetry


class CompletionService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # Configurable model

    async def generate_response(
        self,
        query: str,
        similar_messages: list,
        user_id: str,
        is_dm: bool = False,
        temperature: float = 0.8,
        conversation_context: str = None,
        requester_display_name: str = None,
    ) -> str:
        tracer = get_tracer()
        start_time = time.time()

        try:
            with tracer.start_as_current_span("generate_response") as span:
                span.set_attribute("query.length", len(query))
                span.set_attribute("query.preview", query[:100])
                span.set_attribute("user_id", user_id)
                span.set_attribute("is_dm", is_dm)
                span.set_attribute("temperature", temperature)
                span.set_attribute("model", self.model)
                span.set_attribute("examples.count", len(similar_messages))

                system_prompt = self._build_system_prompt(
                    user_id,
                    similar_messages,
                    query,
                    is_dm,
                    conversation_context,
                    requester_display_name,
                )
                span.set_attribute("prompt.length", len(system_prompt))
                span.set_attribute("system_prompt.full", system_prompt)
                span.set_attribute("user_query", query)

                # Add all examples to telemetry for debugging
                for i, msg in enumerate(
                    similar_messages[:10]
                ):  # Limit to top 10 for telemetry
                    span.set_attribute(
                        f"example.{i}.text", msg["message"][:100]
                    )  # First 100 chars
                    span.set_attribute(
                        f"example.{i}.similarity", msg.get("similarity", 0)
                    )
                    span.set_attribute(f"example.{i}.length", len(msg["message"]))

                # Add conversation context and requester info to telemetry
                span.set_attribute(
                    "conversation_context.length",
                    len(conversation_context) if conversation_context else 0,
                )
                span.set_attribute(
                    "requester_display_name", requester_display_name or ""
                )
                if conversation_context:
                    span.set_attribute(
                        "conversation_context.preview", conversation_context[:200]
                    )

                # Note: Full prompt and response content will be automatically captured
                # by OpenAI instrumentation when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

                # Debug: Log example count and top examples
                logger.debug(
                    f"🎯 Using {len(similar_messages)} examples for user {user_id}"
                )

                # Debug: Log conversation context and requester info
                logger.debug(
                    f"💬 Conversation context length: {len(conversation_context) if conversation_context else 0}"
                )
                logger.debug(f"👤 Requester display name: '{requester_display_name}'")
                if conversation_context:
                    logger.debug(f"💬 Context preview: {conversation_context[:100]}...")

                if similar_messages:
                    top_examples = [
                        f"({msg.get('similarity', 0):.3f}) {msg['message'][:50]}..."
                        for msg in similar_messages[:3]
                    ]
                    logger.debug(f"📝 Top examples: {top_examples}")

                    # Log all examples used in the prompt for debugging
                    logger.debug("🔍 All examples for prompt debugging:")
                    for i, msg in enumerate(similar_messages[:5]):
                        logger.debug(
                            f"  {i + 1}. (sim: {msg.get('similarity', 0):.3f}) {msg['message'][:80]}..."
                        )

                # Create non-streaming response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=temperature,
                    max_tokens=300,
                    top_p=0.9,  # Slightly more focused for style consistency
                    frequency_penalty=0.1,  # Reduce repetition but allow natural patterns
                    presence_penalty=0.1,  # Light penalty to maintain style consistency
                )

                generated_response = response.choices[0].message.content

                duration = time.time() - start_time

                span.set_attribute("response.length", len(generated_response))
                span.set_attribute("response.preview", generated_response[:100])
                span.set_attribute("duration_seconds", duration)

                # Record metrics
                telemetry.record_completion_generated(
                    len(query), len(generated_response), duration, temperature
                )

                logger.info(f"Generated AI response for query: {query[:50]}...")
                return generated_response

        except Exception as e:
            duration = time.time() - start_time
            with tracer.start_as_current_span("generate_response_error") as span:
                span.set_attribute("error", str(e))
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("user_id", user_id)
            logger.error(f"Error generating completion: {e}")
            raise

    def _build_system_prompt(
        self,
        user_id: str,
        similar_messages: list,
        query: str,
        is_dm: bool = False,
        conversation_context: str = None,
        requester_display_name: str = None,
    ) -> str:
        """Build prompt using many-shot in-context learning with actual examples"""

        # Format similar messages as natural conversation examples
        examples = []
        for msg in similar_messages[:12]:  # Use top 12 examples for better focus
            message_text = msg["message"].strip()

            # Skip very short messages (less than 10 chars) as they're not good examples
            if len(message_text) < 10:
                continue

            # Truncate very long messages but keep more content for style transfer
            if len(message_text) > 500:
                message_text = message_text[:500] + "..."

            # Present as natural conversation, not formal examples
            examples.append(f'"{message_text}"')

        # Format examples to show your communication patterns
        if len(examples) < 3:
            examples_text = (
                "Your communication style (limited examples):\n"
                + "\n".join(examples)
                + "\n\nNote: Few examples available - focus on natural, authentic responses."
            )
        else:
            examples_text = "Your authentic communication patterns:\n" + "\n".join(
                examples
            )

        context_note = "DM conversation" if is_dm else "public channel conversation"

        # Add conversation context if available
        context_section = ""
        if conversation_context and conversation_context.strip():
            context_section = f"""

Recent conversation context:
{conversation_context}

This context shows your recent interactions. Use it to maintain conversation continuity and context awareness."""

        # Add requester name information
        requester_section = ""
        if requester_display_name:
            # Always include the requester info if we have a name, even if it's just a user ID
            requester_section = f"""

The person messaging you is {requester_display_name}. You can refer to them by name in your response to make it more personal and natural."""

        prompt = f"""Your task is to perform style transfer while limiting topic transfer in conversations. As PhiLLM, your goal is to mimic Phillip's unique style while keeping topics consistent when responding to messages.

Analyze the provided communication samples to understand Phillip's style:

{examples_text}

# Style Analysis

Based on these examples, identify Phillip's patterns:
- **Vocabulary**: Look for casual, modern slang, fragmented expressions
- **Sentence Length**: Note if responses are short and concise
- **Punctuation**: Observe informal, conversational punctuation
- **Emotional Expression**: Identify if tone is relaxed, laid-back
- **Conversation Flow**: Notice if style is easygoing, humorous, or ironic

Respond to messages using this style without altering the underlying topic.

# Critical Guidelines

- Maintain Phillip's casual and fragmented style if observed in examples
- Avoid adding information or formality unless indicated by Phillip's style
- Use short responses that align with Phillip's typical message length
- Match Phillip's personality and energy exactly as shown in the examples
- Stay focused on the conversation topic without introducing new subjects

{context_section}{requester_section}

# Notes

- This is a {context_note}
- Refer to people by name when appropriate for a personal touch
- Stay aware of the conversation context provided
- Ensure responses maintain Phillip's style without shifting the conversation's topic

Respond exactly as Phillip would:"""

        # Debug: Log sections of the prompt for debugging
        logger.debug(
            f"🎭 Built prompt with context_section: {bool(context_section)}, requester_section: {bool(requester_section)}"
        )
        logger.debug(f"🎭 Final prompt length: {len(prompt)} chars")

        return prompt

    # Old style analysis methods removed - now using many-shot learning approach

    async def generate_scheduled_message(
        self, context: str, user_id: str, topic: Optional[str] = None
    ) -> str:
        try:
            prompt = self._build_scheduled_prompt(user_id, context, topic)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300,
            )

            generated_message = response.choices[0].message.content
            logger.info(f"Generated scheduled message for user {user_id}")

            return generated_message

        except Exception as e:
            logger.error(f"Error generating scheduled message: {e}")
            raise

    def _build_scheduled_prompt(
        self, user_id: str, context: str, topic: Optional[str]
    ) -> str:
        base_prompt = f"""Based on the communication style of user {user_id} shown in these messages:

{context}

Generate a message that this person might naturally send in a Slack channel."""

        if topic:
            base_prompt += f" The message should be about: {topic}"

        base_prompt += """

The message should:
- Sound natural and authentic to their voice
- Be appropriate for a workplace Slack channel
- Be conversational and engaging
- Not be too long (1-3 sentences typically)"""

        return base_prompt
