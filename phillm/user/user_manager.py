import os
import time
from typing import Dict, Optional
import redis.asyncio as redis
from loguru import logger
from phillm.telemetry import get_tracer


class UserManager:
    """Manages user information and display name mapping"""

    def __init__(self, slack_client=None):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")

        self.redis_client = redis.from_url(
            self.redis_url, password=self.redis_password, decode_responses=True
        )

        self.slack_client = slack_client

        # Cache settings
        self.cache_ttl = int(os.getenv("USER_CACHE_TTL", "86400"))  # 24 hours
        self.batch_size = 50  # Process users in batches

    def set_slack_client(self, slack_client):
        """Set the Slack client for API calls"""
        self.slack_client = slack_client

    async def get_user_display_name(self, user_id: str) -> str:
        """Get display name for a user, with caching"""
        tracer = get_tracer()

        logger.debug(f"👤 get_user_display_name called for user {user_id}")

        try:
            with tracer.start_as_current_span("get_user_display_name") as span:
                span.set_attribute("user_id", user_id)

                # Check cache first
                cached_name = await self._get_cached_user_info(user_id)
                if cached_name:
                    span.set_attribute("cache_hit", True)
                    span.set_attribute("display_name", cached_name)
                    logger.debug(
                        f"👤 Resolved user {user_id} to display name: '{cached_name}' (from cache)"
                    )
                    return cached_name

                # Fetch from Slack API
                span.set_attribute("cache_hit", False)
                user_info = await self._fetch_user_from_slack(user_id)

                if user_info:
                    display_name = user_info.get(
                        "display_name", user_info.get("real_name", user_id)
                    )

                    # Cache the result
                    await self._cache_user_info(user_id, user_info)

                    span.set_attribute("display_name", display_name)
                    span.set_attribute("fallback_to_user_id", display_name == user_id)

                    logger.debug(
                        f"👤 Resolved user {user_id} to display name: '{display_name}' (fallback: {display_name == user_id})"
                    )
                    return display_name
                else:
                    span.set_attribute("display_name", user_id)
                    logger.debug(
                        f"👤 No user info found for {user_id}, using user_id as display name"
                    )
                    return user_id  # Fallback to user ID

        except Exception as e:
            logger.error(f"❌ Error getting user display name for {user_id}: {e}")
            logger.debug(f"👤 Falling back to user_id '{user_id}' as display name")
            return user_id  # Fallback to user ID

    async def get_user_info(self, user_id: str) -> Dict:
        """Get comprehensive user information"""
        try:
            # Check cache first
            cached_info = await self._get_cached_user_info(user_id, full_info=True)
            if cached_info:
                return cached_info

            # Fetch from Slack API
            user_info = await self._fetch_user_from_slack(user_id)

            if user_info:
                # Cache the result
                await self._cache_user_info(user_id, user_info)
                return user_info
            else:
                return {
                    "user_id": user_id,
                    "display_name": user_id,
                    "real_name": user_id,
                }

        except Exception as e:
            logger.error(f"Error getting user info for {user_id}: {e}")
            return {"user_id": user_id, "display_name": user_id, "real_name": user_id}

    async def get_multiple_users(self, user_ids: list) -> Dict[str, str]:
        """Get display names for multiple users efficiently"""
        tracer = get_tracer()

        try:
            with tracer.start_as_current_span("get_multiple_users") as span:
                span.set_attribute("user_count", len(user_ids))

                result = {}
                uncached_users = []

                # Check cache for all users
                for user_id in user_ids:
                    cached_name = await self._get_cached_user_info(user_id)
                    if cached_name:
                        result[user_id] = cached_name
                    else:
                        uncached_users.append(user_id)

                span.set_attribute("cache_hits", len(result))
                span.set_attribute("uncached_users", len(uncached_users))

                # Fetch uncached users in batches
                if uncached_users:
                    for i in range(0, len(uncached_users), self.batch_size):
                        batch = uncached_users[i : i + self.batch_size]
                        batch_result = await self._fetch_users_batch(batch)
                        result.update(batch_result)

                return result

        except Exception as e:
            logger.error(f"Error getting multiple users: {e}")
            # Return fallback mapping
            return {user_id: user_id for user_id in user_ids}

    async def preload_channel_users(self, channel_id: str) -> Dict[str, str]:
        """Preload user information for all members of a channel"""
        try:
            if not self.slack_client:
                logger.warning("No Slack client available for preloading users")
                return {}

            # Get channel members
            members_result = await self.slack_client.conversations_members(
                channel=channel_id
            )
            user_ids = members_result.get("members", [])

            if not user_ids:
                return {}

            logger.info(f"Preloading {len(user_ids)} users from channel {channel_id}")

            # Get display names for all users
            user_mapping = await self.get_multiple_users(user_ids)

            logger.info(f"Successfully preloaded {len(user_mapping)} user mappings")
            return user_mapping

        except Exception as e:
            logger.error(f"Error preloading channel users for {channel_id}: {e}")
            return {}

    async def _get_cached_user_info(
        self, user_id: str, full_info: bool = False
    ) -> Optional[str]:
        """Get user info from cache"""
        try:
            cache_key = f"user_info:{user_id}"
            cached_data = await self.redis_client.hgetall(cache_key)

            if not cached_data:
                return None

            # Check if cache is still valid
            cached_time = float(cached_data.get("cached_at", 0))
            if time.time() - cached_time > self.cache_ttl:
                # Cache expired, remove it
                await self.redis_client.delete(cache_key)
                return None

            if full_info:
                # Return full user info
                return {
                    "user_id": user_id,
                    "display_name": cached_data.get("display_name", user_id),
                    "real_name": cached_data.get("real_name", user_id),
                    "email": cached_data.get("email", ""),
                    "title": cached_data.get("title", ""),
                    "status_text": cached_data.get("status_text", ""),
                    "is_bot": cached_data.get("is_bot", "false") == "true",
                    "cached_at": cached_time,
                }
            else:
                # Return just display name
                return cached_data.get("display_name", user_id)

        except Exception as e:
            logger.error(f"Error getting cached user info for {user_id}: {e}")
            return None

    async def _cache_user_info(self, user_id: str, user_info: Dict):
        """Cache user information"""
        try:
            cache_key = f"user_info:{user_id}"
            cache_data = {
                "display_name": user_info.get(
                    "display_name", user_info.get("real_name", user_id)
                ),
                "real_name": user_info.get("real_name", user_id),
                "email": user_info.get("email", ""),
                "title": user_info.get("title", ""),
                "status_text": user_info.get("status_text", ""),
                "is_bot": str(user_info.get("is_bot", False)),
                "cached_at": str(time.time()),
            }

            # Set cache with TTL
            await self.redis_client.hset(cache_key, mapping=cache_data)
            await self.redis_client.expire(cache_key, self.cache_ttl)

        except Exception as e:
            logger.error(f"Error caching user info for {user_id}: {e}")

    async def _fetch_user_from_slack(self, user_id: str) -> Optional[Dict]:
        """Fetch user information from Slack API"""
        try:
            if not self.slack_client:
                logger.warning(f"🚨 No Slack client available to fetch user {user_id}")
                return None

            logger.debug(f"🔍 Fetching user info from Slack API for {user_id}")

            response = await self.slack_client.users_info(user=user_id)

            if not response.get("ok"):
                logger.warning(
                    f"🚨 Failed to fetch user {user_id}: {response.get('error')}"
                )
                return None

            user_data = response.get("user", {})
            profile = user_data.get("profile", {})
            logger.debug(f"✅ Successfully fetched user data from Slack for {user_id}")

            # Extract relevant information
            user_info = {
                "user_id": user_id,
                "real_name": user_data.get("real_name", ""),
                "display_name": profile.get("display_name")
                or profile.get("real_name")
                or user_data.get("name", user_id),
                "email": profile.get("email", ""),
                "title": profile.get("title", ""),
                "status_text": profile.get("status_text", ""),
                "is_bot": user_data.get("is_bot", False),
                "is_admin": user_data.get("is_admin", False),
                "timezone": user_data.get("tz", ""),
            }

            return user_info

        except Exception as e:
            logger.error(f"Error fetching user {user_id} from Slack: {e}")
            return None

    async def _fetch_users_batch(self, user_ids: list) -> Dict[str, str]:
        """Fetch multiple users efficiently"""
        result = {}

        # Note: Slack doesn't have a batch users.info API, so we need to make individual calls
        # But we can do them concurrently for better performance
        import asyncio

        async def fetch_single_user(user_id):
            user_info = await self._fetch_user_from_slack(user_id)
            if user_info:
                display_name = user_info.get(
                    "display_name", user_info.get("real_name", user_id)
                )
                await self._cache_user_info(user_id, user_info)
                return user_id, display_name
            else:
                return user_id, user_id

        # Process batch concurrently with rate limiting
        tasks = [fetch_single_user(user_id) for user_id in user_ids]

        try:
            # Add small delays to avoid rate limiting
            results = []
            for i, task in enumerate(tasks):
                if i > 0 and i % 10 == 0:  # Rate limit: max 10 concurrent requests
                    await asyncio.sleep(1)
                results.append(await task)

            # Convert to dict
            for user_id, display_name in results:
                result[user_id] = display_name

        except Exception as e:
            logger.error(f"Error in batch user fetch: {e}")
            # Fallback: return user IDs as display names
            for user_id in user_ids:
                result[user_id] = user_id

        return result

    async def invalidate_user_cache(self, user_id: str):
        """Invalidate cached user information"""
        try:
            cache_key = f"user_info:{user_id}"
            await self.redis_client.delete(cache_key)
            logger.debug(f"Invalidated cache for user {user_id}")
        except Exception as e:
            logger.error(f"Error invalidating cache for user {user_id}: {e}")

    async def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            # Count cached users
            cache_keys = await self.redis_client.keys("user_info:*")
            total_cached = len(cache_keys)

            # Get a sample to check freshness
            fresh_count = 0
            stale_count = 0
            current_time = time.time()

            for key in cache_keys[:100]:  # Sample first 100
                cached_data = await self.redis_client.hgetall(key)
                if cached_data:
                    cached_time = float(cached_data.get("cached_at", 0))
                    if current_time - cached_time <= self.cache_ttl:
                        fresh_count += 1
                    else:
                        stale_count += 1

            return {
                "total_cached_users": total_cached,
                "cache_ttl_hours": self.cache_ttl / 3600,
                "sample_fresh": fresh_count,
                "sample_stale": stale_count,
                "sample_size": min(100, total_cached),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close Redis connection"""
        await self.redis_client.close()
