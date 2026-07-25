"""
recognizers.py
==============
Domain-specific custom recognizers for the HR domain.

These extend Presidio's built-in capabilities with patterns that are
relevant to HR / employment contexts, such as employee IDs, salary
figures, national insurance numbers, and internal department codes.

Each recognizer is registered into a shared RecognizerRegistry that
is passed to the AnalyzerEngine at startup.
"""

from __future__ import annotations

import logging

from presidio_analyzer import PatternRecognizer, RecognizerRegistry
from presidio_analyzer.pattern import Pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual custom recognizers
# ---------------------------------------------------------------------------

def _build_employee_id_recognizer() -> PatternRecognizer:
    """
    Detects internal employee identifiers in the format EMP-XXXXXX
    (e.g. EMP-123456).
    """
    return PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        name="EmployeeIdRecognizer",
        patterns=[
            Pattern(
                name="employee_id_pattern",
                regex=r"\bEMP[-_]?\d{4,8}\b",
                score=0.85,
            )
        ],
        context=["employee", "staff", "id", "identifier", "emp"],
        supported_language="en",
    )


def _build_salary_recognizer() -> PatternRecognizer:
    """
    Detects salary / compensation figures (USD, GBP, EUR).

    Patterns match values like:
      - $85,000
      - £45,000/year
      - EUR 120000
      - 95k salary
    """
    return PatternRecognizer(
        supported_entity="SALARY",
        name="SalaryRecognizer",
        patterns=[
            Pattern(
                name="currency_amount",
                regex=r"(?:USD|GBP|EUR|£|\$|€)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s?(?:k|K|thousand|per\s+year|/yr|/year|pa))?",
                score=0.75,
            ),
            Pattern(
                name="salary_k_notation",
                regex=r"\b\d{2,3}[kK]\s*(?:salary|compensation|package|ctc|base|annual)\b",
                score=0.70,
            ),
        ],
        context=["salary", "compensation", "pay", "wage", "remuneration", "annual", "ctc", "package"],
        supported_language="en",
    )


def _build_national_insurance_recognizer() -> PatternRecognizer:
    """
    Detects UK National Insurance Numbers (NINO).
    Format: XX 99 99 99 X (e.g. AB 12 34 56 C).
    """
    return PatternRecognizer(
        supported_entity="UK_NINO",
        name="UKNationalInsuranceRecognizer",
        patterns=[
            Pattern(
                name="uk_nino_spaced",
                regex=r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
                score=0.90,
            )
        ],
        context=["national insurance", "ni number", "nino", "insurance"],
        supported_language="en",
    )


def _build_us_ssn_recognizer() -> PatternRecognizer:
    """
    High-confidence US Social Security Number recognizer with both
    hyphenated and non-hyphenated formats.
    """
    return PatternRecognizer(
        supported_entity="US_SSN",
        name="USSocialSecurityRecognizer",
        patterns=[
            Pattern(
                name="ssn_hyphenated",
                regex=r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
                score=0.85,
            ),
            Pattern(
                name="ssn_no_dashes",
                regex=r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b",
                score=0.50,  # lower because 9 digits match many non-SSN numbers
            ),
        ],
        context=["ssn", "social security", "tax id", "tin"],
        supported_language="en",
    )


def _build_bank_account_recognizer() -> PatternRecognizer:
    """
    Detects UK sort code + account number combos and generic IBAN prefixes.
    """
    return PatternRecognizer(
        supported_entity="BANK_ACCOUNT",
        name="BankAccountRecognizer",
        patterns=[
            Pattern(
                name="uk_sort_code_account",
                regex=r"\b\d{2}-\d{2}-\d{2}\s+\d{8}\b",
                score=0.90,
            ),
            Pattern(
                name="iban_prefix",
                regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
                score=0.60,
            ),
        ],
        context=["account", "bank", "sort code", "iban", "bic", "swift"],
        supported_language="en",
    )


def _build_passport_recognizer() -> PatternRecognizer:
    """
    Detects passport numbers for US (9 chars) and UK (9 alphanumeric chars).
    """
    return PatternRecognizer(
        supported_entity="PASSPORT",
        name="PassportRecognizer",
        patterns=[
            Pattern(
                name="us_passport",
                regex=r"\b[A-Z]\d{8}\b",
                score=0.65,
            ),
            Pattern(
                name="uk_passport",
                regex=r"\b\d{9}\b",
                score=0.40,  # many 9-digit numbers; rely on context for boost
            ),
        ],
        context=["passport", "travel document", "pass no", "passport number"],
        supported_language="en",
    )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_registry() -> RecognizerRegistry:
    """
    Build and return a RecognizerRegistry pre-loaded with all standard
    Presidio recognizers plus the HR-domain custom recognizers.
    """
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    custom_recognizers = [
        _build_employee_id_recognizer(),
        _build_salary_recognizer(),
        _build_national_insurance_recognizer(),
        _build_us_ssn_recognizer(),
        _build_bank_account_recognizer(),
        _build_passport_recognizer(),
    ]

    for recognizer in custom_recognizers:
        registry.add_recognizer(recognizer)
        logger.info("Registered custom recognizer: %s", recognizer.name)

    logger.info(
        "RecognizerRegistry built with %d total recognizers (%d custom).",
        len(list(registry.recognizers)),
        len(custom_recognizers),
    )
    return registry


CUSTOM_ENTITY_TYPES: list[str] = [
    "EMPLOYEE_ID",
    "SALARY",
    "UK_NINO",
    "US_SSN",
    "BANK_ACCOUNT",
    "PASSPORT",
]
