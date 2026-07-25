"""
models.py
=========
Pydantic schemas for all MCP tool inputs and outputs.

Using Pydantic models as tool parameters enforces strict validation
before any Presidio processing occurs, providing clear error messages
to the calling LLM when inputs are malformed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnonymizerOperator(str, Enum):
    """Supported anonymization strategies in Presidio."""
    REPLACE = "replace"   # Replace with a placeholder tag, e.g. <PERSON>
    REDACT = "redact"     # Completely remove the entity text
    MASK = "mask"         # Overwrite characters with a masking character
    HASH = "hash"         # Replace with a SHA-256 hash of the entity
    ENCRYPT = "encrypt"   # AES-encrypt the entity (reversible with key)


class Language(str, Enum):
    """
    Languages supported by the currently loaded spaCy NLP model(s).

    Only ``ENGLISH`` is enabled by default (en_core_web_lg / en_core_web_sm).

    To add another language, add its spaCy model to ``_NLP_CONFIGURATION``
    in ``engines.py``, download the model, and add the enum value here.

    Example for Spanish::

        # engines.py — add to _NLP_CONFIGURATION["models"]:
        {"lang_code": "es", "model_name": "es_core_news_md"}

        # Then add here:
        SPANISH = "es"
    """
    ENGLISH = "en"



class HashType(str, Enum):
    MD5 = "md5"
    SHA256 = "sha256"
    SHA512 = "sha512"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class AnalyzerResult(BaseModel):
    """A single PII entity detected by the Analyzer."""
    entity_type: str = Field(..., description="Category of PII, e.g. PERSON, EMAIL_ADDRESS")
    start: int = Field(..., description="Start character offset in the source text")
    end: int = Field(..., description="End character offset in the source text")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0–1)")
    text: str = Field(..., description="The actual PII text that was detected")


class EntityOperatorConfig(BaseModel):
    """
    Per-entity anonymization configuration.

    Examples
    --------
    Replace PERSON with <PERSON>:
        {"entity_type": "PERSON", "operator": "replace"}

    Mask PHONE_NUMBER with *:
        {"entity_type": "PHONE_NUMBER", "operator": "mask",
         "masking_char": "*", "chars_to_mask": 4, "from_end": True}

    Hash EMAIL_ADDRESS:
        {"entity_type": "EMAIL_ADDRESS", "operator": "hash", "hash_type": "sha256"}

    Encrypt CREDIT_CARD:
        {"entity_type": "CREDIT_CARD", "operator": "encrypt", "key": "WmZq4t7w!z%C&F)J"}
    """
    entity_type: str = Field(..., description="Entity type this config applies to")
    operator: AnonymizerOperator = Field(
        AnonymizerOperator.REPLACE,
        description="Anonymization strategy to apply"
    )
    # mask options
    masking_char: str = Field("*", description="Character used when operator=mask")
    chars_to_mask: int = Field(6, ge=1, description="Number of characters to mask")
    from_end: bool = Field(False, description="If True, mask from the end of the string")
    # replace option
    new_value: Optional[str] = Field(None, description="Replacement text when operator=replace")
    # hash option
    hash_type: HashType = Field(HashType.SHA256, description="Hash algorithm when operator=hash")
    # encrypt option
    key: Optional[str] = Field(None, description="AES encryption key (16/24/32 bytes) when operator=encrypt")


# ---------------------------------------------------------------------------
# Tool Input Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Input for the analyze_text tool."""
    text: str = Field(..., min_length=1, max_length=100_000, description="Text to scan for PII")
    language: Language = Field(Language.ENGLISH, description="Language of the input text")
    entities: Optional[list[str]] = Field(
        None,
        description=(
            "Specific entity types to detect. If None, all supported entities are scanned. "
            "Examples: ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD']"
        ),
    )
    score_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Minimum confidence score (0–1) to report an entity"
    )
    return_decision_process: bool = Field(
        False,
        description="Include the internal decision process details in the response"
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v


class AnonymizeRequest(BaseModel):
    """Input for the anonymize_text tool."""
    text: str = Field(..., min_length=1, max_length=100_000, description="Text to anonymize")
    language: Language = Field(Language.ENGLISH, description="Language of the input text")
    entities: Optional[list[str]] = Field(
        None,
        description="Limit detection to these entity types. None means detect all."
    )
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence score")
    operators: Optional[list[EntityOperatorConfig]] = Field(
        None,
        description=(
            "Per-entity anonymization configuration. "
            "If not provided, defaults to REPLACE for all detected entities."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v


class DeanonymizeRequest(BaseModel):
    """Input for the deanonymize_text tool (for encrypted entities only)."""
    anonymized_text: str = Field(..., description="Text that was previously anonymized with encryption")
    entities: list[str] = Field(..., description="Entity types that were encrypted")
    key: str = Field(..., description="The AES key used during encryption")


class BatchAnalyzeRequest(BaseModel):
    """Input for the batch_analyze tool."""
    texts: list[str] = Field(
        ..., min_length=1, max_length=50,
        description="List of texts to analyze (max 50 items)"
    )
    language: Language = Field(Language.ENGLISH)
    entities: Optional[list[str]] = Field(None)
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)


class CustomRecognizerRequest(BaseModel):
    """Input for adding a regex-based custom recognizer."""
    name: str = Field(..., description="Unique name for this recognizer, e.g. EMPLOYEE_ID")
    patterns: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of regex pattern dicts, each with 'regex' (str), 'score' (float 0-1), "
            "and optionally 'name' (str). Example: [{'name': 'id', 'regex': r'EMP-\\d{6}', 'score': 0.85}]"
        ),
    )
    context_words: Optional[list[str]] = Field(
        None,
        description="Context words that raise confidence when found near the pattern (e.g. ['employee', 'id'])"
    )
    supported_language: Language = Field(Language.ENGLISH)


# ---------------------------------------------------------------------------
# Tool Output Models
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Output from analyze_text."""
    text_length: int
    entity_count: int
    entities: list[AnalyzerResult]
    language: str
    score_threshold: float
    has_pii: bool


class AnonymizeResponse(BaseModel):
    """Output from anonymize_text."""
    original_text: str
    anonymized_text: str
    entity_count: int
    entities_found: list[AnalyzerResult]
    operators_applied: dict[str, str]


class BatchAnalyzeResponse(BaseModel):
    """Output from batch_analyze."""
    total_texts: int
    results: list[AnalyzeResponse]
    summary: dict[str, Any]


class SupportedEntitiesResponse(BaseModel):
    """Output from list_supported_entities."""
    entities: list[str]
    total: int
    grouped: dict[str, list[str]]


class RecognizerInfo(BaseModel):
    """Metadata about a single recognizer."""
    name: str
    supported_entities: list[str]
    supported_language: str
    is_custom: bool


class RecognizersResponse(BaseModel):
    """Output from list_recognizers."""
    total: int
    recognizers: list[RecognizerInfo]
