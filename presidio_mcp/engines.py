"""
engines.py
==========
Singleton initialization of Presidio AnalyzerEngine and AnonymizerEngine.

Both engines are expensive to construct (spaCy model loading, registry
scanning) so they are created once at module import time and reused
across all MCP tool invocations.

Thread Safety
-------------
FastMCP runs tool handlers concurrently (asyncio). Both Presidio engines
are stateless after initialization and are safe to call from multiple
coroutines simultaneously.
"""

from __future__ import annotations

import logging
import threading

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine

from recognizers import build_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLP Engine configuration
# ---------------------------------------------------------------------------

_NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [
        # Medium model provides a good balance of accuracy vs startup time.
        # Use en_core_web_lg for higher accuracy (slower startup).
        {"lang_code": "en", "model_name": "en_core_web_lg"},
    ],
}

# ---------------------------------------------------------------------------
# Thread-safe lazy singletons
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None
_deanonymizer: DeanonymizeEngine | None = None


def get_analyzer() -> AnalyzerEngine:
    """
    Return the singleton AnalyzerEngine, creating it on first call.

    The engine is initialised with:
    - spaCy NLP backend (en_core_web_lg, falls back to en_core_web_sm)
    - All predefined Presidio recognizers
    - All HR-domain custom recognizers from recognizers.py

    Note: ``supported_languages`` is intentionally NOT passed here.
    Presidio derives it from the NLP engine's loaded models, keeping
    the registry and engine in sync and avoiding the
    "Misconfigured engine, supported languages have to be consistent"
    ValueError that occurs when the two sets diverge.
    """
    global _analyzer
    if _analyzer is None:
        with _lock:
            if _analyzer is None:  # double-checked locking
                logger.info("Initialising Presidio AnalyzerEngine …")
                try:
                    provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIGURATION)
                    nlp_engine = provider.create_engine()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to load en_core_web_lg (%s). "
                        "Falling back to en_core_web_sm. "
                        "Run: python -m spacy download en_core_web_lg",
                        exc,
                    )
                    _NLP_CONFIGURATION["models"] = [
                        {"lang_code": "en", "model_name": "en_core_web_sm"}
                    ]
                    provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIGURATION)
                    nlp_engine = provider.create_engine()

                registry = build_registry()
                # Do NOT pass supported_languages — let Presidio derive it
                # from the NLP engine so registry and engine stay in sync.
                _analyzer = AnalyzerEngine(
                    nlp_engine=nlp_engine,
                    registry=registry,
                )
                logger.info(
                    "AnalyzerEngine ready. Supported languages: %s",
                    _analyzer.supported_languages,
                )
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    """Return the singleton AnonymizerEngine."""
    global _anonymizer
    if _anonymizer is None:
        with _lock:
            if _anonymizer is None:
                logger.info("Initialising Presidio AnonymizerEngine …")
                _anonymizer = AnonymizerEngine()
                logger.info("AnonymizerEngine ready.")
    return _anonymizer


def get_deanonymizer() -> DeanonymizeEngine:
    """Return the singleton DeanonymizeEngine (for encrypted entity reversal)."""
    global _deanonymizer
    if _deanonymizer is None:
        with _lock:
            if _deanonymizer is None:
                logger.info("Initialising Presidio DeanonymizeEngine …")
                _deanonymizer = DeanonymizeEngine()
                logger.info("DeanonymizeEngine ready.")
    return _deanonymizer


# ---------------------------------------------------------------------------
# Warm-up helper (called during server lifespan)
# ---------------------------------------------------------------------------

def warm_up_engines() -> None:
    """
    Pre-initialise all engines so the first real request is not slow.
    Call this inside the FastMCP lifespan startup hook.
    """
    logger.info("Warming up Presidio engines …")
    analyzer = get_analyzer()
    get_anonymizer()
    get_deanonymizer()

    # Run a trivial analysis to force spaCy pipeline loading
    _ = analyzer.analyze(text="warm-up", language="en")
    logger.info("All Presidio engines warmed up and ready.")
