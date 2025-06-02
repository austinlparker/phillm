import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Ensure .env is loaded before accessing environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# OpenTelemetry imports (after .env loading)
# ruff: noqa: E402
from opentelemetry import trace, metrics, _logs, _events
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.resources import get_aggregated_resources
from opentelemetry.sdk.extension.aws.resource import (
    AwsEc2ResourceDetector,
    AwsEcsResourceDetector,
    AwsEksResourceDetector,
    AwsLambdaResourceDetector,
    AwsBeanstalkResourceDetector,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor


class TelemetryConfig:
    """OpenTelemetry configuration for PhiLLM"""

    def __init__(self) -> None:
        self.service_name = "phillm"
        self.service_version = "0.1.0"
        self.honeycomb_api_key = os.getenv("HONEYCOMB_API_KEY")
        self.honeycomb_endpoint = "https://api.honeycomb.io:443"
        self.environment = os.getenv("ENVIRONMENT", "development")

        # Custom tracer, meter, and logger
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None
        self.logger: Optional[_logs.Logger] = None

        # Custom metrics
        self.embedding_counter: Optional[metrics.Counter] = None
        self.completion_counter: Optional[metrics.Counter] = None
        self.similarity_search_counter: Optional[metrics.Counter] = None
        self.dm_counter: Optional[metrics.Counter] = None
        self.mention_counter: Optional[metrics.Counter] = None
        self.scraping_counter: Optional[metrics.Counter] = None
        self.response_time_histogram: Optional[metrics.Histogram] = None
        self.embedding_time_histogram: Optional[metrics.Histogram] = None
        self.similarity_score_histogram: Optional[metrics.Histogram] = None

    def setup_telemetry(self) -> bool:
        """Initialize OpenTelemetry with Honeycomb export (if API key available)"""
        # Skip if already setup (prevents issues with reloaders)
        if self.tracer is not None:
            logger.info("OpenTelemetry already configured")
            return True

        # Also check if global providers are already set up
        try:
            existing_tracer = trace.get_tracer_provider()
            existing_meter = metrics.get_meter_provider()
            existing_logger = _logs.get_logger_provider()

            # If any provider has a resource, assume they're properly configured
            if (
                hasattr(existing_tracer, "resource")
                and hasattr(existing_meter, "resource")
                and hasattr(existing_logger, "resource")
            ):
                logger.info("OpenTelemetry providers already configured globally")
                self.tracer = trace.get_tracer(self.service_name, self.service_version)
                self.meter = metrics.get_meter(self.service_name, self.service_version)
                self.logger = _logs.get_logger(self.service_name, self.service_version)
                self._setup_custom_metrics()
                return True
        except Exception:
            pass

        try:
            # Create base resource
            base_resource = Resource.create(
                {
                    SERVICE_NAME: self.service_name,
                    SERVICE_VERSION: self.service_version,
                    "environment": self.environment,
                    "service.instance.id": os.getenv("HOSTNAME", "unknown"),
                }
            )

            # Create AWS resource detectors
            aws_detectors = [
                AwsEcsResourceDetector(),  # For ECS Fargate
                AwsEc2ResourceDetector(),  # For EC2 instances
                AwsEksResourceDetector(),  # For EKS
                AwsBeanstalkResourceDetector(),  # For Elastic Beanstalk
                AwsLambdaResourceDetector(),  # For Lambda
            ]

            # Aggregate resources with AWS detection
            try:
                resource = get_aggregated_resources(
                    detectors=aws_detectors,
                    initial_resource=base_resource,
                    timeout=5,  # 5 second timeout for resource detection
                )
                logger.info(
                    f"🔧 Detected AWS resources: {[str(attr) for attr in resource.attributes.keys() if attr.startswith('aws.') or attr.startswith('cloud.')]}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to detect AWS resources: {e}, using base resource"
                )
                resource = base_resource

            # Setup tracing
            self._setup_tracing(resource)

            # Setup metrics
            self._setup_metrics(resource)

            # Setup logging
            self._setup_logging(resource)

            # Setup event logging
            self._setup_event_logging(resource)

            # Setup auto-instrumentation
            self._setup_auto_instrumentation()

            # Setup custom metrics
            self._setup_custom_metrics()

            export_target = (
                self.honeycomb_endpoint
                if self.honeycomb_api_key
                else "no export (local only)"
            )
            logger.info(
                f"🔭 OpenTelemetry configured for {self.service_name} -> {export_target}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
            # Even if setup fails, we should still initialize basic tracers
            self._setup_fallback_telemetry()
            return False

    def _setup_fallback_telemetry(self) -> None:
        """Setup basic telemetry if main setup fails"""
        try:
            # Minimal tracer setup
            resource = Resource.create({SERVICE_NAME: self.service_name})
            trace_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(trace_provider)
            self.tracer = trace.get_tracer(self.service_name, self.service_version)

            # Minimal meter setup
            metric_provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(metric_provider)
            self.meter = metrics.get_meter(self.service_name, self.service_version)

            # Minimal logger setup
            logger_provider = LoggerProvider(resource=resource)
            _logs.set_logger_provider(logger_provider)
            self.logger = _logs.get_logger(self.service_name, self.service_version)

            logger.info("🔭 Fallback telemetry configured (no export)")
        except Exception as e:
            logger.error(f"Even fallback telemetry failed: {e}")

    def _setup_tracing(self, resource: Resource) -> None:
        """Configure tracing with OTLP export to Honeycomb (if API key available)"""
        # Check if we should skip setup
        try:
            existing_provider = trace.get_tracer_provider()
            if (
                hasattr(existing_provider, "resource")
                and existing_provider.__class__.__name__ == "TracerProvider"
            ):
                logger.debug("Tracer provider already exists, skipping setup")
                self.tracer = trace.get_tracer(self.service_name, self.service_version)
                return
        except Exception:
            pass

        trace_provider = TracerProvider(resource=resource)

        # Only add exporter if Honeycomb API key is available
        if self.honeycomb_api_key:
            exporter = OTLPSpanExporter(
                endpoint=self.honeycomb_endpoint,
                headers={
                    "x-honeycomb-team": self.honeycomb_api_key,
                },
                insecure=False,
            )
            span_processor = BatchSpanProcessor(exporter)
            trace_provider.add_span_processor(span_processor)
            logger.debug("🔭 Using Honeycomb exporter for traces")
        else:
            logger.debug("🔭 No Honeycomb API key - traces captured locally only")

        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(self.service_name, self.service_version)

    def _setup_metrics(self, resource: Resource) -> None:
        """Configure metrics with OTLP export to Honeycomb (if API key available)"""
        # Check if we should skip setup
        try:
            existing_provider = metrics.get_meter_provider()
            if (
                hasattr(existing_provider, "resource")
                and existing_provider.__class__.__name__ == "MeterProvider"
            ):
                logger.debug("Meter provider already exists, skipping setup")
                self.meter = metrics.get_meter(self.service_name, self.service_version)
                return
        except Exception:
            pass

        # Only add exporter if Honeycomb API key is available
        metric_readers = []
        if self.honeycomb_api_key:
            exporter = OTLPMetricExporter(
                endpoint=self.honeycomb_endpoint,
                headers={
                    "x-honeycomb-team": self.honeycomb_api_key,
                },
                insecure=False,
            )
            metric_reader = PeriodicExportingMetricReader(
                exporter=exporter,
                export_interval_millis=10000,  # Export every 10 seconds
            )
            metric_readers.append(metric_reader)
            logger.debug("🔭 Using Honeycomb exporter for metrics")
        else:
            logger.debug("🔭 No Honeycomb API key - metrics captured locally only")

        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers,
        )

        metrics.set_meter_provider(metric_provider)
        self.meter = metrics.get_meter(self.service_name, self.service_version)

    def _setup_logging(self, resource: Resource) -> None:
        """Configure logging with OTLP export to Honeycomb (if API key available)"""
        # Check if we should skip setup
        try:
            existing_provider = _logs.get_logger_provider()
            if (
                hasattr(existing_provider, "resource")
                and existing_provider.__class__.__name__ == "LoggerProvider"
            ):
                logger.debug("Logger provider already exists, skipping setup")
                self.logger = _logs.get_logger(self.service_name, self.service_version)
                return
        except Exception:
            pass

        logger_provider = LoggerProvider(resource=resource)

        # Only add exporter if Honeycomb API key is available
        if self.honeycomb_api_key:
            exporter = OTLPLogExporter(
                endpoint=self.honeycomb_endpoint,
                headers={
                    "x-honeycomb-team": self.honeycomb_api_key,
                },
                insecure=False,
            )
            log_processor = BatchLogRecordProcessor(exporter)
            logger_provider.add_log_record_processor(log_processor)
            logger.debug("🔭 Using Honeycomb exporter for logs")
        else:
            logger.debug("🔭 No Honeycomb API key - logs captured locally only")

        _logs.set_logger_provider(logger_provider)
        self.logger = _logs.get_logger(self.service_name, self.service_version)

        # Setup handler to bridge standard Python logging to OpenTelemetry
        if self.honeycomb_api_key:
            import logging

            handler = LoggingHandler(
                level=logging.INFO, logger_provider=logger_provider
            )
            # Add to root logger to capture app logs
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)

    def _setup_event_logging(self, resource: Resource) -> None:
        """Configure event logging for OpenTelemetry events (required for OpenAI span events)"""
        try:
            # Check if we should skip setup
            existing_provider = _events.get_event_logger_provider()
            if (
                hasattr(existing_provider, "resource")
                and existing_provider.__class__.__name__ == "EventLoggerProvider"
            ):
                logger.debug("Event logger provider already exists, skipping setup")
                return
        except Exception:
            pass

        # Set up EventLoggerProvider - this is required for OpenAI instrumentation events
        event_logger_provider = EventLoggerProvider()
        _events.set_event_logger_provider(event_logger_provider)

        logger.debug("🔭 Event logger provider configured for OpenAI span events")

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries"""
        try:
            # Check if already instrumented to avoid conflicts
            if not getattr(FastAPIInstrumentor, "_is_instrumented", False):
                FastAPIInstrumentor().instrument()

            # HTTP client instrumentation
            if not getattr(HTTPXClientInstrumentor, "_is_instrumented", False):
                HTTPXClientInstrumentor().instrument()

            if not getattr(RequestsInstrumentor, "_is_instrumented", False):
                RequestsInstrumentor().instrument()

            if not getattr(AioHttpClientInstrumentor, "_is_instrumented", False):
                AioHttpClientInstrumentor().instrument()

            # Redis instrumentation
            if not getattr(RedisInstrumentor, "_is_instrumented", False):
                RedisInstrumentor().instrument()

            # OpenAI instrumentation - automatic tracing of all OpenAI API calls
            # Ensure the environment variable is set for content capture
            import os

            os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

            if not getattr(OpenAIInstrumentor, "_is_instrumented", False):
                OpenAIInstrumentor().instrument()

            logger.info(
                "🔧 Auto-instrumentation enabled for FastAPI, HTTP clients, Redis, and OpenAI"
            )

        except Exception as e:
            logger.warning(
                f"Some auto-instrumentation failed (this is normal in dev mode): {e}"
            )

    def _setup_custom_metrics(self) -> None:
        """Setup custom business metrics"""
        if not self.meter:
            return

        # Counters
        self.embedding_counter = self.meter.create_counter(
            "phillm_embeddings_total",
            description="Total number of embeddings created",
            unit="1",
        )

        self.completion_counter = self.meter.create_counter(
            "phillm_completions_total",
            description="Total number of AI completions generated",
            unit="1",
        )

        self.similarity_search_counter = self.meter.create_counter(
            "phillm_similarity_searches_total",
            description="Total number of similarity searches performed",
            unit="1",
        )

        self.dm_counter = self.meter.create_counter(
            "phillm_dm_messages_total",
            description="Total number of DM messages processed",
            unit="1",
        )

        self.mention_counter = self.meter.create_counter(
            "phillm_mention_messages_total",
            description="Total number of @ mention messages processed",
            unit="1",
        )

        self.scraping_counter = self.meter.create_counter(
            "phillm_messages_scraped_total",
            description="Total number of messages scraped from Slack",
            unit="1",
        )

        # Histograms
        self.response_time_histogram = self.meter.create_histogram(
            "phillm_response_time_seconds",
            description="Time taken to generate AI responses",
            unit="s",
        )

        self.embedding_time_histogram = self.meter.create_histogram(
            "phillm_embedding_time_seconds",
            description="Time taken to create embeddings",
            unit="s",
        )

        self.similarity_score_histogram = self.meter.create_histogram(
            "phillm_similarity_scores",
            description="Similarity scores from vector search",
            unit="1",
        )

        logger.info("📊 Custom metrics initialized")

    def record_embedding_created(self, text_length: int, duration: float) -> None:
        """Record embedding creation metrics"""
        try:
            if self.embedding_counter:
                self.embedding_counter.add(
                    1, {"text_length_bucket": self._get_length_bucket(text_length)}
                )
            if self.embedding_time_histogram:
                self.embedding_time_histogram.record(duration)
        except Exception as e:
            logger.debug(f"Failed to record embedding metrics: {e}")

    def record_completion_generated(
        self,
        query_length: int,
        response_length: int,
        duration: float,
        temperature: float,
    ) -> None:
        """Record completion generation metrics"""
        try:
            if self.completion_counter:
                self.completion_counter.add(
                    1,
                    {
                        "query_length_bucket": self._get_length_bucket(query_length),
                        "temperature_bucket": self._get_temperature_bucket(temperature),
                    },
                )
            if self.response_time_histogram:
                self.response_time_histogram.record(duration)
        except Exception as e:
            logger.debug(f"Failed to record completion metrics: {e}")

    def record_similarity_search(
        self, query_length: int, results_count: int, threshold: float, max_score: float
    ) -> None:
        """Record similarity search metrics"""
        try:
            if self.similarity_search_counter:
                self.similarity_search_counter.add(
                    1,
                    {
                        "results_count_bucket": self._get_results_bucket(results_count),
                        "threshold": str(threshold),
                    },
                )
            if self.similarity_score_histogram and max_score > 0:
                self.similarity_score_histogram.record(max_score)
        except Exception as e:
            logger.debug(f"Failed to record similarity search metrics: {e}")

    def record_dm_processed(self, user_id: str, response_length: int) -> None:
        """Record DM processing metrics"""
        try:
            if self.dm_counter:
                self.dm_counter.add(
                    1,
                    {
                        "user_id": user_id,
                        "response_length_bucket": self._get_length_bucket(
                            response_length
                        ),
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to record DM metrics: {e}")

    def record_mention_processed(
        self, user_id: str, channel_id: str, response_length: int
    ) -> None:
        """Record @ mention processing metrics"""
        try:
            if self.mention_counter:
                self.mention_counter.add(
                    1,
                    {
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "response_length_bucket": self._get_length_bucket(
                            response_length
                        ),
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to record mention metrics: {e}")

    def record_message_scraped(self, channel_id: str, user_id: str) -> None:
        """Record message scraping metrics"""
        try:
            if self.scraping_counter:
                self.scraping_counter.add(
                    1,
                    {
                        "channel_id": channel_id,
                        "user_id": user_id,
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to record scraping metrics: {e}")

    def _get_length_bucket(self, length: int) -> str:
        """Get length bucket for text length"""
        if length < 50:
            return "short"
        elif length < 200:
            return "medium"
        elif length < 500:
            return "long"
        else:
            return "very_long"

    def _get_temperature_bucket(self, temperature: float) -> str:
        """Get temperature bucket"""
        if temperature < 0.5:
            return "low"
        elif temperature < 1.0:
            return "medium"
        else:
            return "high"

    def _get_results_bucket(self, count: int) -> str:
        """Get results count bucket"""
        if count == 0:
            return "none"
        elif count < 5:
            return "few"
        elif count < 15:
            return "many"
        else:
            return "lots"


# Global telemetry instance
telemetry = TelemetryConfig()


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance, or the default noop tracer if not initialized"""
    if telemetry.tracer:
        return telemetry.tracer
    # OpenTelemetry's default tracer is a noop when no provider is set
    return trace.get_tracer(__name__)
