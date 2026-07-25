"""
server.py
=========
Presidio MCP Server — built with FastMCP 2.x.

Exposes Microsoft Presidio PII detection and anonymization capabilities
as a fully spec-compliant Model Context Protocol (MCP) server.

Three MCP primitives are implemented:
  - TOOLS    : Active operations (analyze, anonymize, batch-analyze, …)
  - RESOURCES: Static reference data (entity catalogue, config snapshot)
  - PROMPTS  : Reusable prompt templates for common PII-handling workflows

Transport
---------
Default: stdio (works with Claude Desktop, Cursor, Continue, etc.)
HTTP/SSE: run with  mcp.run(transport="sse", host="0.0.0.0", port=8001)

Usage
-----
    python -m presidio_mcp.server
    # or via uvx / fastmcp CLI:
    fastmcp run presidio_mcp/server.py
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastmcp import FastMCP
from fastmcp.server import Context

from engines import get_analyzer, get_anonymizer, get_deanonymizer, warm_up_engines
from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzerResult,
    AnonymizeRequest,
    AnonymizeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    CustomRecognizerRequest,
    DeanonymizeRequest,
    EntityOperatorConfig,
    RecognizerInfo,
    RecognizersResponse,
    SupportedEntitiesResponse,
)
from recognizers import CUSTOM_ENTITY_TYPES, build_registry

from presidio_analyzer import PatternRecognizer
from presidio_analyzer.pattern import Pattern
from presidio_anonymizer.entities import OperatorConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # MCP uses stdout for protocol; keep logs on stderr
)
logger = logging.getLogger("presidio_mcp")

# ---------------------------------------------------------------------------
# Entity groupings (used by the supported-entities resource)
# ---------------------------------------------------------------------------

ENTITY_GROUPS: dict[str, list[str]] = {
    "Personal Identity": [
        "PERSON", "DATE_TIME", "AGE", "NRP",
    ],
    "Contact Information": [
        "EMAIL_ADDRESS", "PHONE_NUMBER", "URL",
    ],
    "Location": [
        "LOCATION", "ADDRESS",
    ],
    "Financial": [
        "CREDIT_CARD", "IBAN_CODE", "BANK_ACCOUNT", "SALARY",
    ],
    "Government IDs": [
        "US_SSN", "US_DRIVER_LICENSE", "US_PASSPORT", "US_BANK_NUMBER",
        "UK_NHS", "UK_NINO", "PASSPORT",
    ],
    "Network & Tech": [
        "IP_ADDRESS", "MEDICAL_LICENSE",
    ],
    "HR Domain (Custom)": CUSTOM_ENTITY_TYPES,
}

# ---------------------------------------------------------------------------
# Lifespan — warm up engines before accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Warm up Presidio engines during server startup."""
    logger.info("Presidio MCP Server starting up …")
    warm_up_engines()
    logger.info("Presidio MCP Server is ready.")
    yield
    logger.info("Presidio MCP Server shutting down.")


# ---------------------------------------------------------------------------
# FastMCP Server instantiation
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="presidio-mcp",
    instructions=(
        "This server exposes Microsoft Presidio PII detection and anonymization "
        "capabilities. Use it to scan text for personally identifiable information "
        "(PII), anonymize sensitive content, or perform batch processing. "
        "All tools accept structured Pydantic-validated inputs."
    ),
    lifespan=lifespan,
)


# ===========================================================================
# TOOLS
# ===========================================================================

# ---------------------------------------------------------------------------
# Tool 1: analyze_text
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": True,          # Does not mutate external state
        "idempotentHint": True,        # Same input → same output
        "openWorldHint": False,        # No external network calls
    }
)
def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Scan text for PII entities using Microsoft Presidio's NLP engine.

    Detects a wide range of PII types including names, emails, phone numbers,
    credit cards, government IDs, and HR-domain entities (employee IDs, salaries).

    Returns a structured list of detected entities with confidence scores and
    character offsets so callers can pinpoint PII in the source text.

    Examples
    --------
    • Detect all PII in an HR email:
        {"text": "Hi John, your new salary is $95,000. NI: AB 12 34 56 C", "language": "en"}

    • Detect only emails and phone numbers:
        {"text": "Call me at 555-1234 or email@example.com", "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]}
    """
    analyzer = get_analyzer()

    results = analyzer.analyze(
        text=request.text,
        language=request.language.value,
        entities=request.entities,
        score_threshold=request.score_threshold,
        return_decision_process=request.return_decision_process,
    )

    entities = [
        AnalyzerResult(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=round(r.score, 4),
            text=request.text[r.start : r.end],
        )
        for r in results
    ]

    return AnalyzeResponse(
        text_length=len(request.text),
        entity_count=len(entities),
        entities=entities,
        language=request.language.value,
        score_threshold=request.score_threshold,
        has_pii=len(entities) > 0,
    )


# ---------------------------------------------------------------------------
# Tool 2: anonymize_text
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def anonymize_text(request: AnonymizeRequest) -> AnonymizeResponse:
    """
    Detect and anonymize PII in text using configurable per-entity strategies.

    Supported anonymization operators:
      • replace  — Replace with a placeholder tag, e.g. <PERSON>  (default)
      • redact   — Delete the entity text entirely
      • mask     — Overwrite characters with '*' or a custom masking character
      • hash     — Replace with SHA-256 (or MD5/SHA-512) hash of the original
      • encrypt  — AES-encrypt the entity (reversible with deanonymize_text)

    Examples
    --------
    • Anonymize everything with defaults (replace):
        {"text": "John Smith earns $120,000 and lives in London"}

    • Mask phone numbers, hash emails, replace everything else:
        {
          "text": "Call Jane on 07911123456 or jane@company.com",
          "operators": [
            {"entity_type": "PHONE_NUMBER", "operator": "mask", "chars_to_mask": 6},
            {"entity_type": "EMAIL_ADDRESS", "operator": "hash", "hash_type": "sha256"}
          ]
        }
    """
    analyzer = get_analyzer()
    anonymizer = get_anonymizer()

    # Step 1: Analyze
    analysis_results = analyzer.analyze(
        text=request.text,
        language=request.language.value,
        entities=request.entities,
        score_threshold=request.score_threshold,
    )

    # Step 2: Build operator map
    operators: dict[str, OperatorConfig] = {}
    operators_applied: dict[str, str] = {}

    if request.operators:
        for op_cfg in request.operators:
            entity = op_cfg.entity_type
            op_name = op_cfg.operator.value

            if op_name == "mask":
                params = {
                    "masking_char": op_cfg.masking_char,
                    "chars_to_mask": op_cfg.chars_to_mask,
                    "from_end": op_cfg.from_end,
                }
            elif op_name == "replace":
                params = {}
                if op_cfg.new_value:
                    params["new_value"] = op_cfg.new_value
            elif op_name == "hash":
                params = {"hash_type": op_cfg.hash_type.value}
            elif op_name == "encrypt":
                if not op_cfg.key:
                    raise ValueError(
                        f"Operator 'encrypt' for entity '{entity}' requires a 'key' parameter."
                    )
                params = {"key": op_cfg.key}
            else:  # redact
                params = {}

            operators[entity] = OperatorConfig(op_name, params)
            operators_applied[entity] = op_name

    # Default to replace for entities not explicitly configured
    for result in analysis_results:
        if result.entity_type not in operators:
            operators[result.entity_type] = OperatorConfig("replace", {})
            operators_applied[result.entity_type] = "replace"

    # Step 3: Anonymize
    anonymized = anonymizer.anonymize(
        text=request.text,
        analyzer_results=analysis_results,
        operators=operators if operators else None,
    )

    entities = [
        AnalyzerResult(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=round(r.score, 4),
            text=request.text[r.start : r.end],
        )
        for r in analysis_results
    ]

    return AnonymizeResponse(
        original_text=request.text,
        anonymized_text=anonymized.text,
        entity_count=len(entities),
        entities_found=entities,
        operators_applied=operators_applied,
    )


# ---------------------------------------------------------------------------
# Tool 3: deanonymize_text
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def deanonymize_text(request: DeanonymizeRequest) -> dict[str, str]:
    """
    Reverse AES-encrypted anonymization to recover original PII values.

    This tool only works when the original anonymization was performed
    with the 'encrypt' operator and the same AES key.

    Important: Store encryption keys securely. If the key is lost,
    recovery is impossible.

    Example
    -------
        {
          "anonymized_text": "Contact <PERSON> at <EMAIL_ADDRESS>",
          "entities": ["PERSON", "EMAIL_ADDRESS"],
          "key": "WmZq4t7w!z%C&F)J"
        }
    """
    from presidio_anonymizer.entities import RecognizerResult, OperatorResult

    deanonymizer = get_deanonymizer()

    # Build mock OperatorResult list for the deanonymizer
    # In practice, callers should pass the actual anonymization results;
    # here we use a simplified approach for demo purposes.
    result = deanonymizer.deanonymize(
        text=request.anonymized_text,
        entities=[
            OperatorResult(
                operator="decrypt",
                entity_type=entity,
                start=0,
                end=0,
                text="",
            )
            for entity in request.entities
        ],
        operators={
            entity: OperatorConfig("decrypt", {"key": request.key})
            for entity in request.entities
        },
    )

    return {
        "original_text": request.anonymized_text,
        "deanonymized_text": result.text,
    }


# ---------------------------------------------------------------------------
# Tool 4: batch_analyze
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def batch_analyze(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """
    Analyze multiple texts for PII in a single call (up to 50 texts).

    More efficient than calling analyze_text repeatedly when processing
    lists of records (e.g., CSV rows, resume excerpts, email bodies).

    Returns individual results per text plus a summary of entity type
    frequencies across the entire batch.

    Example
    -------
        {
          "texts": [
            "Alice Smith, alice@acme.com",
            "Employee Bob, EMP-001234, salary $80,000"
          ],
          "entities": ["PERSON", "EMAIL_ADDRESS", "EMPLOYEE_ID", "SALARY"]
        }
    """
    analyzer = get_analyzer()

    individual_results: list[AnalyzeResponse] = []
    entity_frequency: dict[str, int] = {}
    total_pii = 0

    for text in request.texts:
        if not text.strip():
            individual_results.append(
                AnalyzeResponse(
                    text_length=0, entity_count=0, entities=[],
                    language=request.language.value,
                    score_threshold=request.score_threshold,
                    has_pii=False,
                )
            )
            continue

        results = analyzer.analyze(
            text=text,
            language=request.language.value,
            entities=request.entities,
            score_threshold=request.score_threshold,
        )

        entities = [
            AnalyzerResult(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=round(r.score, 4),
                text=text[r.start : r.end],
            )
            for r in results
        ]

        for e in entities:
            entity_frequency[e.entity_type] = entity_frequency.get(e.entity_type, 0) + 1
            total_pii += 1

        individual_results.append(
            AnalyzeResponse(
                text_length=len(text),
                entity_count=len(entities),
                entities=entities,
                language=request.language.value,
                score_threshold=request.score_threshold,
                has_pii=len(entities) > 0,
            )
        )

    summary = {
        "total_pii_instances": total_pii,
        "texts_with_pii": sum(1 for r in individual_results if r.has_pii),
        "texts_without_pii": sum(1 for r in individual_results if not r.has_pii),
        "entity_frequency": entity_frequency,
    }

    return BatchAnalyzeResponse(
        total_texts=len(request.texts),
        results=individual_results,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Tool 5: list_supported_entities
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def list_supported_entities(language: str = "en") -> SupportedEntitiesResponse:
    """
    Return all PII entity types that this server can detect.

    Includes both built-in Presidio entity types and HR-domain custom
    entities (EMPLOYEE_ID, SALARY, UK_NINO, etc.).

    Use the returned entity type strings as values for the 'entities'
    parameter in analyze_text and anonymize_text.

    Args:
        language: ISO 639-1 language code to filter by (default: 'en')
    """
    analyzer = get_analyzer()
    entities = analyzer.get_supported_entities(language=language)
    entities_sorted = sorted(entities)

    return SupportedEntitiesResponse(
        entities=entities_sorted,
        total=len(entities_sorted),
        grouped=ENTITY_GROUPS,
    )


# ---------------------------------------------------------------------------
# Tool 6: add_custom_recognizer
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "idempotentHint": False,  # Adding the same recognizer twice has side effects
        "destructiveHint": False,
        "openWorldHint": False,
    }
)
def add_custom_recognizer(request: CustomRecognizerRequest) -> dict[str, Any]:
    """
    Dynamically register a new regex-based PII recognizer at runtime.

    Useful for adding domain-specific patterns without restarting the server.
    Note: Custom recognizers added via this tool are not persisted across
    server restarts. To make them permanent, add them to recognizers.py.

    Example
    -------
    Add a recognizer for project codes like PRJ-2024-001:
        {
          "name": "PROJECT_CODE",
          "patterns": [{"name": "project_code", "regex": "PRJ-\\d{4}-\\d{3}", "score": 0.85}],
          "context_words": ["project", "code", "prj"]
        }
    """
    analyzer = get_analyzer()

    patterns = [
        Pattern(
            name=p.get("name", f"pattern_{i}"),
            regex=p["regex"],
            score=float(p.get("score", 0.7)),
        )
        for i, p in enumerate(request.patterns)
    ]

    recognizer = PatternRecognizer(
        supported_entity=request.name,
        name=f"{request.name}Recognizer",
        patterns=patterns,
        context=request.context_words or [],
        supported_language=request.supported_language.value,
    )

    analyzer.registry.add_recognizer(recognizer)
    logger.info("Dynamically added custom recognizer: %s", request.name)

    return {
        "success": True,
        "entity_type": request.name,
        "patterns_added": len(patterns),
        "message": f"Recognizer '{request.name}' successfully registered. It will be active for all subsequent analyze/anonymize calls.",
    }


# ---------------------------------------------------------------------------
# Tool 7: score_pii_risk
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def score_pii_risk(text: str, language: str = "en") -> dict[str, Any]:
    """
    Compute a PII risk score (0–100) for a given text.

    The score reflects the density and severity of PII detected:
      • 0–20   : Low risk — minimal or no PII detected
      • 21–50  : Medium risk — some PII present, review recommended
      • 51–80  : High risk — significant PII density, action required
      • 81–100 : Critical risk — extremely high PII density

    High-severity entities (SSN, credit card, bank account) contribute
    more to the score than lower-severity ones (URLs, dates).

    Args:
        text: The text to evaluate.
        language: ISO 639-1 language code (default: 'en').
    """
    HIGH_SEVERITY = {"US_SSN", "CREDIT_CARD", "IBAN_CODE", "BANK_ACCOUNT", "UK_NINO", "PASSPORT"}
    MEDIUM_SEVERITY = {"EMAIL_ADDRESS", "PHONE_NUMBER", "US_DRIVER_LICENSE", "EMPLOYEE_ID", "SALARY"}

    analyzer = get_analyzer()
    results = analyzer.analyze(text=text, language=language, score_threshold=0.4)

    if not results:
        return {
            "risk_score": 0,
            "risk_level": "Low",
            "entity_count": 0,
            "entities_detected": [],
            "recommendation": "No PII detected. Text appears safe to share.",
        }

    base_score = 0
    detected_types: set[str] = set()

    for r in results:
        detected_types.add(r.entity_type)
        if r.entity_type in HIGH_SEVERITY:
            base_score += 20 * r.score
        elif r.entity_type in MEDIUM_SEVERITY:
            base_score += 10 * r.score
        else:
            base_score += 5 * r.score

    # Density factor: more entities per 100 chars raises score
    density_factor = (len(results) / max(len(text), 1)) * 100
    risk_score = min(100, int(base_score + density_factor * 2))

    if risk_score <= 20:
        level = "Low"
        recommendation = "Minimal PII. Review before sharing externally."
    elif risk_score <= 50:
        level = "Medium"
        recommendation = "PII detected. Anonymize before sharing."
    elif risk_score <= 80:
        level = "High"
        recommendation = "High PII density. Anonymization mandatory before any sharing."
    else:
        level = "Critical"
        recommendation = "Critical PII density. Do NOT share. Immediately anonymize or restrict access."

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "entity_count": len(results),
        "entities_detected": sorted(detected_types),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Tool 8: list_recognizers
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def list_recognizers(language: Optional[str] = None) -> RecognizersResponse:
    """
    List all active PII recognizers with their metadata.

    Shows both built-in Presidio recognizers and any custom recognizers
    (including those added dynamically via add_custom_recognizer).

    Args:
        language: Optional ISO 639-1 filter. If provided, only recognizers
                  supporting that language are returned.
    """
    analyzer = get_analyzer()

    all_recognizers = list(analyzer.registry.recognizers)
    if language:
        all_recognizers = [
            r for r in all_recognizers
            if hasattr(r, "supported_language") and r.supported_language == language
        ]

    custom_names = {f"{et}Recognizer" for et in CUSTOM_ENTITY_TYPES}
    recognizer_infos = []

    for rec in all_recognizers:
        supported_entities = (
            [rec.supported_entity]
            if hasattr(rec, "supported_entity")
            else getattr(rec, "supported_entities", [])
        )
        recognizer_infos.append(
            RecognizerInfo(
                name=getattr(rec, "name", type(rec).__name__),
                supported_entities=supported_entities if isinstance(supported_entities, list) else [supported_entities],
                supported_language=getattr(rec, "supported_language", "en"),
                is_custom=getattr(rec, "name", "") in custom_names,
            )
        )

    return RecognizersResponse(
        total=len(recognizer_infos),
        recognizers=recognizer_infos,
    )


# ===========================================================================
# RESOURCES
# ===========================================================================

@mcp.resource(
    uri="presidio://entities/catalogue",
    name="PII Entity Catalogue",
    description=(
        "Complete catalogue of all supported PII entity types, organized by category. "
        "Reference this before calling analyze_text to understand what can be detected."
    ),
    mime_type="application/json",
)
def entity_catalogue() -> str:
    """
    Returns the full PII entity catalogue as JSON.

    Categories include:
    - Personal Identity (PERSON, DATE_TIME, AGE …)
    - Contact Information (EMAIL_ADDRESS, PHONE_NUMBER, URL …)
    - Location (LOCATION, ADDRESS …)
    - Financial (CREDIT_CARD, IBAN_CODE, SALARY …)
    - Government IDs (US_SSN, UK_NINO, PASSPORT …)
    - HR Domain custom entities (EMPLOYEE_ID, SALARY …)
    """
    return json.dumps(
        {
            "version": "1.0.0",
            "description": "Microsoft Presidio supported PII entity types",
            "groups": ENTITY_GROUPS,
            "custom_entities": CUSTOM_ENTITY_TYPES,
            "total_groups": len(ENTITY_GROUPS),
        },
        indent=2,
    )


@mcp.resource(
    uri="presidio://operators/guide",
    name="Anonymization Operators Guide",
    description=(
        "Reference guide for all supported anonymization operators with parameters, "
        "examples, and use-case recommendations."
    ),
    mime_type="application/json",
)
def operators_guide() -> str:
    """
    Returns a JSON reference guide for all anonymization operators.

    Covers replace, redact, mask, hash, and encrypt operators with
    their configuration parameters and practical examples.
    """
    guide = {
        "operators": [
            {
                "name": "replace",
                "description": "Replace entity text with a placeholder tag like <PERSON>",
                "parameters": {
                    "new_value": "(optional) Custom replacement string. If omitted, uses <ENTITY_TYPE>"
                },
                "use_case": "General-purpose anonymization for sharing de-identified data",
                "example_output": "Hi <PERSON>, your <EMAIL_ADDRESS> has been updated.",
            },
            {
                "name": "redact",
                "description": "Remove the entity text entirely, leaving a gap",
                "parameters": {},
                "use_case": "When the placeholder itself may leak information",
                "example_output": "Hi , your  has been updated.",
            },
            {
                "name": "mask",
                "description": "Overwrite N characters of the entity with a masking character",
                "parameters": {
                    "masking_char": "Character to use (default: '*')",
                    "chars_to_mask": "Number of characters to mask (default: 6)",
                    "from_end": "If true, mask from the end of the string (default: false)",
                },
                "use_case": "Partial masking of credit cards, phone numbers, or IDs",
                "example_output": "Card: ****1234",
            },
            {
                "name": "hash",
                "description": "Replace entity with a cryptographic hash (non-reversible)",
                "parameters": {
                    "hash_type": "md5 | sha256 | sha512 (default: sha256)"
                },
                "use_case": "When you need consistent pseudonymization (same input → same hash)",
                "example_output": "Email: a87f3...",
            },
            {
                "name": "encrypt",
                "description": "AES-encrypt the entity value (reversible with the key)",
                "parameters": {
                    "key": "AES key string (must be 16, 24, or 32 bytes)"
                },
                "use_case": "When original values must be recoverable by authorised parties",
                "example_output": "Name: <encrypted:AbC12...>",
            },
        ]
    }
    return json.dumps(guide, indent=2)


@mcp.resource(
    uri="presidio://config/server",
    name="Server Configuration",
    description="Current server configuration, NLP engine details, and loaded recognizer counts.",
    mime_type="application/json",
)
def server_config() -> str:
    """
    Returns a snapshot of the current server configuration.

    Includes the NLP engine type, supported languages, total recognizers
    loaded, and custom entity types. Useful for debugging and health checks.
    """
    try:
        analyzer = get_analyzer()
        recognizer_count = len(list(analyzer.registry.recognizers))
        supported_langs = getattr(analyzer, "supported_languages", ["en"])
        nlp_engine_type = type(analyzer.nlp_engine).__name__
    except Exception:  # noqa: BLE001
        recognizer_count = 0
        supported_langs = ["en"]
        nlp_engine_type = "unknown"

    config = {
        "server_name": "presidio-mcp",
        "version": "1.0.0",
        "nlp_engine": nlp_engine_type,
        "supported_languages": supported_langs,
        "total_recognizers": recognizer_count,
        "custom_entities": CUSTOM_ENTITY_TYPES,
        "default_score_threshold": 0.5,
        "max_batch_size": 50,
        "max_text_length": 100_000,
        "transport": "stdio",
    }
    return json.dumps(config, indent=2)


@mcp.resource(
    uri="presidio://examples/common",
    name="Common Usage Examples",
    description="Ready-to-use examples for the most common Presidio MCP use cases.",
    mime_type="application/json",
)
def usage_examples() -> str:
    """
    Returns a curated set of ready-to-use examples for common PII workflows.

    Examples cover: HR onboarding documents, payroll sheets, email logs,
    customer support tickets, and batch processing.
    """
    examples = {
        "examples": [
            {
                "title": "Analyze HR onboarding form",
                "tool": "analyze_text",
                "input": {
                    "text": "New hire: Sarah Connor, DOB 15-Mar-1985, SSN 123-45-6789, email s.connor@company.com",
                    "language": "en",
                    "entities": ["PERSON", "DATE_TIME", "US_SSN", "EMAIL_ADDRESS"],
                },
            },
            {
                "title": "Anonymize payroll record",
                "tool": "anonymize_text",
                "input": {
                    "text": "Employee John Smith (EMP-000456) receives a salary of £75,000/year. NI: QQ 12 34 56 C",
                    "operators": [
                        {"entity_type": "PERSON", "operator": "replace"},
                        {"entity_type": "EMPLOYEE_ID", "operator": "mask", "chars_to_mask": 4},
                        {"entity_type": "SALARY", "operator": "redact"},
                        {"entity_type": "UK_NINO", "operator": "hash", "hash_type": "sha256"},
                    ],
                },
            },
            {
                "title": "Risk-score a support ticket",
                "tool": "score_pii_risk",
                "input": {
                    "text": "Hi, my credit card 4111-1111-1111-1111 was charged incorrectly. My SSN is 123-45-6789.",
                },
            },
            {
                "title": "Batch analyze resumes",
                "tool": "batch_analyze",
                "input": {
                    "texts": [
                        "Alice Wanderland, alice@email.com, 07700 900123",
                        "Bob Builder, 14 Acacia Ave, London",
                    ],
                    "entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS"],
                },
            },
        ]
    }
    return json.dumps(examples, indent=2)


# ===========================================================================
# PROMPTS
# ===========================================================================

@mcp.prompt(
    name="pii_audit_report",
    description=(
        "Generate a structured PII audit report for a document or text snippet. "
        "Use this prompt when you need to produce a compliance-ready report of "
        "all PII found in a piece of text."
    ),
)
def pii_audit_report_prompt(
    document_text: str,
    document_type: str = "document",
    compliance_framework: str = "GDPR",
) -> list[dict[str, str]]:
    """
    Structured prompt for generating a PII audit report.

    Args:
        document_text: The text content to audit.
        document_type: Describe what kind of document this is (e.g. 'HR form', 'email', 'resume').
        compliance_framework: The compliance framework context (e.g. 'GDPR', 'HIPAA', 'CCPA').
    """
    return [
        {
            "role": "user",
            "content": (
                f"Please perform a comprehensive PII audit on the following {document_type} "
                f"in the context of {compliance_framework} compliance.\n\n"
                f"DOCUMENT:\n{document_text}\n\n"
                "Instructions:\n"
                "1. First, call analyze_text to detect all PII entities.\n"
                "2. Group the findings by entity category (identity, contact, financial, government ID).\n"
                "3. For each entity found, report: entity type, detected value, confidence score, "
                "   and the {compliance_framework} article or regulation it relates to.\n"
                "4. Call score_pii_risk to get an overall risk score.\n"
                "5. Produce a final audit report with:\n"
                "   - Executive Summary (risk score, total entities, key findings)\n"
                "   - Detailed Entity Table\n"
                f"  - {compliance_framework} Compliance Implications\n"
                "   - Recommended Actions (redaction, encryption, access control)\n"
            ),
        }
    ]


@mcp.prompt(
    name="anonymize_for_sharing",
    description=(
        "Prompt to guide anonymization of a document before sharing externally. "
        "Walks through entity detection, operator selection, and produces anonymized output."
    ),
)
def anonymize_for_sharing_prompt(
    text: str,
    sharing_context: str = "external partner",
    keep_structure: bool = True,
) -> list[dict[str, str]]:
    """
    Prompt for anonymizing text for a specific sharing context.

    Args:
        text: The text to be anonymized.
        sharing_context: Who will receive the anonymized text (e.g. 'external partner', 'public').
        keep_structure: If True, prefer replace/mask over redact to preserve document readability.
    """
    operator_preference = (
        "Prefer 'replace' and 'mask' operators to preserve document readability."
        if keep_structure
        else "You may use 'redact' to remove PII entirely."
    )

    return [
        {
            "role": "user",
            "content": (
                f"I need to share the following text with {sharing_context}. "
                f"Please anonymize all PII before sharing.\n\n"
                f"TEXT:\n{text}\n\n"
                f"Anonymization guidelines:\n"
                f"- {operator_preference}\n"
                "- Always mask or encrypt financial entities (credit cards, bank accounts, salaries).\n"
                "- Replace personal identifiers (names, SSNs, NI numbers) with placeholders.\n"
                "- Preserve the overall meaning and structure of the text.\n\n"
                "Steps:\n"
                "1. Call analyze_text to identify all PII.\n"
                "2. Determine the appropriate operator for each entity type.\n"
                "3. Call anonymize_text with the chosen operators.\n"
                "4. Return the anonymized text and a summary of what was changed.\n"
            ),
        }
    ]


@mcp.prompt(
    name="hr_data_privacy_review",
    description=(
        "Comprehensive HR data privacy review prompt. Use when reviewing employee records, "
        "contracts, performance reviews, or any HR-related document for PII compliance."
    ),
)
def hr_data_privacy_review_prompt(
    hr_document: str,
    document_category: str = "employee record",
    action: str = "review",
) -> list[dict[str, str]]:
    """
    Prompt for HR-specific PII review workflows.

    Args:
        hr_document: The HR document content.
        document_category: Type of HR document (e.g. 'employee record', 'payroll', 'contract').
        action: Desired action — 'review' (detect only) or 'anonymize' (detect and anonymize).
    """
    action_instructions = (
        "Detect and catalogue all PII without modifying the text."
        if action == "review"
        else "Detect all PII and produce a fully anonymized version suitable for archival."
    )

    return [
        {
            "role": "user",
            "content": (
                f"Please perform an HR data privacy {action} on the following {document_category}.\n\n"
                f"DOCUMENT:\n{hr_document}\n\n"
                f"Task: {action_instructions}\n\n"
                "Pay special attention to HR-specific PII:\n"
                "  • Employee IDs (format: EMP-XXXXXX)\n"
                "  • Salary and compensation figures\n"
                "  • National Insurance / Social Security numbers\n"
                "  • Bank account and payment details\n"
                "  • Passport and driver's licence numbers\n"
                "  • Personal contact information\n\n"
                "Use the available Presidio tools in this order:\n"
                "1. list_supported_entities — confirm what HR entities are detectable\n"
                "2. analyze_text — with entities focused on HR-domain types\n"
                "3. score_pii_risk — assess overall risk\n"
                + (
                    "4. anonymize_text — apply appropriate operators per entity type\n"
                    if action == "anonymize"
                    else ""
                )
                + "Finally, produce a structured report with findings and recommendations.\n"
            ),
        }
    ]


@mcp.prompt(
    name="batch_pii_policy_check",
    description=(
        "Prompt for running a PII policy compliance check across multiple text records. "
        "Ideal for processing lists of records from CSV exports, databases, or APIs."
    ),
)
def batch_pii_policy_check_prompt(
    records_description: str,
    policy_name: str = "Data Minimization Policy",
    max_allowed_risk: str = "Medium",
) -> list[dict[str, str]]:
    """
    Prompt for batch PII policy compliance checking.

    Args:
        records_description: Description of the records to check (e.g. '50 employee email records').
        policy_name: The internal policy being enforced.
        max_allowed_risk: Maximum risk level allowed ('Low', 'Medium', 'High').
    """
    return [
        {
            "role": "user",
            "content": (
                f"I need to run a {policy_name} compliance check on {records_description}.\n\n"
                f"Policy: No record may have a PII risk level above '{max_allowed_risk}'.\n\n"
                "Workflow:\n"
                "1. Call batch_analyze to scan all records simultaneously.\n"
                "2. Call score_pii_risk on any record that has more than 3 PII entities.\n"
                f"3. Flag any record with risk level above '{max_allowed_risk}'.\n"
                "4. For flagged records, suggest which anonymize_text operators to apply.\n"
                "5. Produce a compliance report with:\n"
                "   - Total records processed\n"
                "   - Records passing policy / failing policy\n"
                "   - Per-entity-type frequency table\n"
                "   - Recommended remediation steps for flagged records\n"
            ),
        }
    ]


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Presidio MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE transport")
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE transport")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
