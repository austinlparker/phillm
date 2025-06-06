import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any
import openai
from loguru import logger
from phillm.telemetry import get_tracer, telemetry


class CompletionService:
    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # Configurable model
        self.max_response_tokens = int(
            os.getenv("MAX_RESPONSE_TOKENS", "3000")
        )  # Configurable response length

    async def generate_response(
        self,
        query: str,
        similar_messages: List[Dict[str, Any]],
        user_id: str,
        is_dm: bool = False,
        temperature: float = 0.8,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        requester_display_name: Optional[str] = None,
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
                span.set_attribute("max_response_tokens", self.max_response_tokens)
                span.set_attribute("examples.count", len(similar_messages))

                # Build system prompt (style instructions only)
                system_prompt = self._build_system_prompt(
                    user_id,
                    similar_messages,
                    is_dm,
                    requester_display_name,
                )

                # Build message history for conversation context
                messages = self._build_messages(
                    system_prompt, conversation_history or [], query
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

                # Add conversation history and requester info to telemetry
                span.set_attribute(
                    "conversation_history.length",
                    len(conversation_history) if conversation_history else 0,
                )
                span.set_attribute(
                    "requester_display_name", requester_display_name or ""
                )
                span.set_attribute("messages.total_count", len(messages))

                # Note: Full prompt and response content will be automatically captured
                # by OpenAI instrumentation when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

                # Debug: Log example count and top examples
                logger.debug(
                    f"🎯 Using {len(similar_messages)} examples for user {user_id}"
                )

                # Debug: Log conversation history and requester info
                logger.debug(
                    f"💬 Conversation history messages: {len(conversation_history) if conversation_history else 0}"
                )
                logger.debug(f"💬 Total messages in request: {len(messages)}")
                logger.debug(f"👤 Requester display name: '{requester_display_name}'")

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

                # Create non-streaming response with conversation history
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=self.max_response_tokens,
                    top_p=0.9,  # Slightly more focused for style consistency
                    frequency_penalty=0.1,  # Reduce repetition but allow natural patterns
                    presence_penalty=0.1,  # Light penalty to maintain style consistency
                )

                generated_response = response.choices[0].message.content or ""

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
        similar_messages: List[Dict[str, Any]],
        is_dm: bool = False,
        requester_display_name: Optional[str] = None,
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

        # Get current Pacific time
        pacific_tz = ZoneInfo("America/Los_Angeles")
        current_time = datetime.now(pacific_tz)
        time_info = current_time.strftime("%A, %B %d, %Y at %I:%M %p PT")

        # Add requester name information (conversation context now handled via message history)
        requester_section = ""
        if requester_display_name:
            requester_section = f"""

The person messaging you is {requester_display_name}. Use their name naturally only when it feels appropriate, not in every response."""

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

{requester_section}

# Notes

- This is a {context_note}
- Current time: {time_info}
- Use names naturally and sparingly - not in every response unless Phillip's style shows frequent name usage
- Use the conversation history provided in the message thread to maintain context
- Ensure responses maintain Phillip's style without shifting the conversation's topic

Respond exactly as Phillip would to continue the conversation naturally."""

        # Debug: Log sections of the prompt for debugging
        logger.debug(
            f"🎭 Built prompt with requester_section: {bool(requester_section)}"
        )
        logger.debug(f"🎭 Final prompt length: {len(prompt)} chars")

        return prompt

    def _build_messages(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        current_query: str,
    ) -> List[Dict[str, str]]:
        """Build the complete message structure for chat completion"""

        # Start with system prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (already filtered for relevance and privacy)
        messages.extend(conversation_history)

        # Add current user query
        messages.append({"role": "user", "content": current_query})

        return messages

    async def generate_scheduled_message(
        self, context: str, user_id: str, topic: Optional[str] = None
    ) -> str:
        try:
            prompt = self._build_scheduled_prompt(user_id, context, topic)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=self.max_response_tokens,
            )

            generated_message = response.choices[0].message.content or ""
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
