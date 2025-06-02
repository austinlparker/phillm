import pytest
from unittest.mock import MagicMock, patch
import os

from phillm.telemetry import TelemetryConfig, get_tracer


@pytest.fixture
def telemetry_config():
    with patch.dict(os.environ, {"HONEYCOMB_API_KEY": "test-key"}):
        return TelemetryConfig()


def test_telemetry_config_init(telemetry_config):
    assert telemetry_config.service_name == "phillm"
    assert telemetry_config.service_version == "0.1.0"
    assert telemetry_config.honeycomb_api_key == "test-key"


def test_telemetry_config_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        config = TelemetryConfig()
        assert config.honeycomb_api_key is None


@patch("phillm.telemetry.trace")
@patch("phillm.telemetry.metrics")
@patch("phillm.telemetry._logs")
def test_setup_telemetry_already_configured(mock_logs, mock_metrics, mock_trace):
    config = TelemetryConfig()
    config.tracer = MagicMock()  # Already configured

    result = config.setup_telemetry()
    assert result is True


@patch("phillm.telemetry.TracerProvider")
@patch("phillm.telemetry.MeterProvider")
@patch("phillm.telemetry.LoggerProvider")
@patch("phillm.telemetry.EventLoggerProvider")
def test_setup_telemetry_success(
    mock_event_provider, mock_log_provider, mock_meter_provider, mock_trace_provider
):
    with patch.dict(os.environ, {"HONEYCOMB_API_KEY": "test-key"}):
        config = TelemetryConfig()

    with (
        patch.object(config, "_setup_tracing"),
        patch.object(config, "_setup_metrics"),
        patch.object(config, "_setup_logging"),
        patch.object(config, "_setup_event_logging"),
        patch.object(config, "_setup_auto_instrumentation"),
        patch.object(config, "_setup_custom_metrics"),
    ):
        result = config.setup_telemetry()

    assert result is True


def test_record_embedding_created(telemetry_config):
    telemetry_config.embedding_counter = MagicMock()
    telemetry_config.embedding_time_histogram = MagicMock()

    telemetry_config.record_embedding_created(100, 0.5)

    telemetry_config.embedding_counter.add.assert_called_once_with(
        1, {"text_length_bucket": "medium"}
    )
    telemetry_config.embedding_time_histogram.record.assert_called_once_with(0.5)


def test_record_completion_generated(telemetry_config):
    telemetry_config.completion_counter = MagicMock()
    telemetry_config.response_time_histogram = MagicMock()

    telemetry_config.record_completion_generated(50, 200, 1.0, 0.7)

    telemetry_config.completion_counter.add.assert_called_once_with(
        1, {"query_length_bucket": "medium", "temperature_bucket": "medium"}
    )
    telemetry_config.response_time_histogram.record.assert_called_once_with(1.0)


def test_record_similarity_search(telemetry_config):
    telemetry_config.similarity_search_counter = MagicMock()
    telemetry_config.similarity_score_histogram = MagicMock()

    telemetry_config.record_similarity_search(100, 5, 0.7, 0.95)

    telemetry_config.similarity_search_counter.add.assert_called_once_with(
        1, {"results_count_bucket": "many", "threshold": "0.7"}
    )
    telemetry_config.similarity_score_histogram.record.assert_called_once_with(0.95)


def test_record_dm_processed(telemetry_config):
    telemetry_config.dm_counter = MagicMock()

    telemetry_config.record_dm_processed("U123", 150)

    telemetry_config.dm_counter.add.assert_called_once_with(
        1, {"user_id": "U123", "response_length_bucket": "medium"}
    )


def test_record_message_scraped(telemetry_config):
    telemetry_config.scraping_counter = MagicMock()

    telemetry_config.record_message_scraped("C123", "U456")

    telemetry_config.scraping_counter.add.assert_called_once_with(
        1, {"channel_id": "C123", "user_id": "U456"}
    )


def test_get_length_bucket(telemetry_config):
    assert telemetry_config._get_length_bucket(30) == "short"
    assert telemetry_config._get_length_bucket(150) == "medium"
    assert telemetry_config._get_length_bucket(400) == "long"
    assert telemetry_config._get_length_bucket(600) == "very_long"


def test_get_temperature_bucket(telemetry_config):
    assert telemetry_config._get_temperature_bucket(0.3) == "low"
    assert telemetry_config._get_temperature_bucket(0.8) == "medium"
    assert telemetry_config._get_temperature_bucket(1.2) == "high"


def test_get_results_bucket(telemetry_config):
    assert telemetry_config._get_results_bucket(0) == "none"
    assert telemetry_config._get_results_bucket(3) == "few"
    assert telemetry_config._get_results_bucket(10) == "many"
    assert telemetry_config._get_results_bucket(20) == "lots"


def test_global_functions():
    # Test that global functions work without errors
    tracer = get_tracer()

    # These might be None if telemetry isn't set up, which is fine
    assert tracer is not None


@patch("phillm.telemetry.FastAPIInstrumentor")
@patch("phillm.telemetry.HTTPXClientInstrumentor")
@patch("phillm.telemetry.OpenAIInstrumentor")
def test_setup_auto_instrumentation(
    mock_openai, mock_httpx, mock_fastapi, telemetry_config
):
    # Mock instrumentors as not instrumented
    mock_fastapi._is_instrumented = False
    mock_httpx._is_instrumented = False
    mock_openai._is_instrumented = False

    with patch(
        "opentelemetry.instrumentation.openai_v2.utils.is_content_enabled",
        return_value=True,
    ):
        telemetry_config._setup_auto_instrumentation()

    mock_fastapi().instrument.assert_called_once()
    mock_httpx().instrument.assert_called_once()
    mock_openai().instrument.assert_called_once()


def test_setup_fallback_telemetry(telemetry_config):
    with (
        patch("phillm.telemetry.TracerProvider") as mock_trace_provider,
        patch("phillm.telemetry.MeterProvider") as mock_meter_provider,
        patch("phillm.telemetry.LoggerProvider") as mock_log_provider,
    ):
        telemetry_config._setup_fallback_telemetry()

        mock_trace_provider.assert_called_once()
        mock_meter_provider.assert_called_once()
        mock_log_provider.assert_called_once()


def test_record_metrics_with_no_meters(telemetry_config):
    # Test that recording metrics with None meters doesn't crash
    telemetry_config.embedding_counter = None
    telemetry_config.embedding_time_histogram = None

    # Should not raise any exceptions
    telemetry_config.record_embedding_created(100, 0.5)
    telemetry_config.record_completion_generated(50, 200, 1.0, 0.7)
    telemetry_config.record_similarity_search(100, 5, 0.7, 0.95)
    telemetry_config.record_dm_processed("U123", 150)
    telemetry_config.record_message_scraped("C123", "U456")
