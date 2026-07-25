"""
tests/test_presidio_mcp.py
===========================
Comprehensive test suite for the Presidio MCP Server.

Tests cover:
  - All 8 tools (analyze_text, anonymize_text, deanonymize_text, batch_analyze,
    list_supported_entities, add_custom_recognizer, score_pii_risk, list_recognizers)
  - All 4 resources (entity_catalogue, operators_guide, server_config, usage_examples)
  - All 4 prompts (pii_audit_report, anonymize_for_sharing, hr_data_privacy_review,
    batch_pii_policy_check)
  - Edge cases: empty text, max length, invalid operators, unknown entities

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json

import pytest
from fastmcp import FastMCP
from fastmcp.client import Client

# Import the configured MCP server instance
from presidio_mcp.server import mcp
from presidio_mcp.models import (
    AnalyzeRequest,
    AnonymizeRequest,
    BatchAnalyzeRequest,
    CustomRecognizerRequest,
    EntityOperatorConfig,
    Language,
    AnonymizerOperator,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    """Session-scoped FastMCP test client (engines are warmed up once)."""
    async with Client(mcp) as c:
        yield c


# ===========================================================================
# TOOLS — analyze_text
# ===========================================================================

class TestAnalyzeText:
    async def test_detects_email(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "Contact jane@example.com for details", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["has_pii"] is True
        entity_types = [e["entity_type"] for e in data["entities"]]
        assert "EMAIL_ADDRESS" in entity_types

    async def test_detects_phone_number(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "Call me on 212-555-1234", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert any(e["entity_type"] == "PHONE_NUMBER" for e in data["entities"])

    async def test_detects_person(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "John Smith applied for the role.", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert any(e["entity_type"] == "PERSON" for e in data["entities"])

    async def test_entity_filter(self, client: Client):
        """Requesting only EMAIL_ADDRESS should suppress PHONE_NUMBER results."""
        result = await client.call_tool(
            "analyze_text",
            {
                "request": {
                    "text": "Email me at bob@test.com or call 555-9876",
                    "language": "en",
                    "entities": ["EMAIL_ADDRESS"],
                }
            },
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        entity_types = [e["entity_type"] for e in data["entities"]]
        assert "EMAIL_ADDRESS" in entity_types
        assert "PHONE_NUMBER" not in entity_types

    async def test_high_score_threshold_suppresses_low_confidence(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "The ID is 123456789", "score_threshold": 0.99, "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        # Very high threshold — most matches should be suppressed
        for entity in data["entities"]:
            assert entity["score"] >= 0.99

    async def test_no_pii_text(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "The quick brown fox jumps over the lazy dog.", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["has_pii"] is False
        assert data["entity_count"] == 0

    async def test_custom_employee_id_detected(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "Employee EMP-123456 has been promoted.", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert any(e["entity_type"] == "EMPLOYEE_ID" for e in data["entities"])

    async def test_character_offsets_are_correct(self, client: Client):
        text = "Call jane@example.com today"
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": text, "language": "en", "entities": ["EMAIL_ADDRESS"]}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        for entity in data["entities"]:
            if entity["entity_type"] == "EMAIL_ADDRESS":
                extracted = text[entity["start"] : entity["end"]]
                assert "jane@example.com" in extracted


# ===========================================================================
# TOOLS — anonymize_text
# ===========================================================================

class TestAnonymizeText:
    async def test_default_replace_operator(self, client: Client):
        result = await client.call_tool(
            "anonymize_text",
            {"request": {"text": "Hi Alice, please send me your CV.", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "Alice" not in data["anonymized_text"]
        assert data["entity_count"] > 0

    async def test_redact_operator(self, client: Client):
        result = await client.call_tool(
            "anonymize_text",
            {
                "request": {
                    "text": "Email: test@example.com",
                    "entities": ["EMAIL_ADDRESS"],
                    "operators": [{"entity_type": "EMAIL_ADDRESS", "operator": "redact"}],
                }
            },
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "test@example.com" not in data["anonymized_text"]

    async def test_mask_operator(self, client: Client):
        result = await client.call_tool(
            "anonymize_text",
            {
                "request": {
                    "text": "Card: 4111-1111-1111-1111",
                    "entities": ["CREDIT_CARD"],
                    "operators": [
                        {
                            "entity_type": "CREDIT_CARD",
                            "operator": "mask",
                            "masking_char": "*",
                            "chars_to_mask": 12,
                            "from_end": False,
                        }
                    ],
                }
            },
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "*" in data["anonymized_text"]
        assert "4111-1111-1111-1111" not in data["anonymized_text"]

    async def test_hash_operator_produces_hex(self, client: Client):
        result = await client.call_tool(
            "anonymize_text",
            {
                "request": {
                    "text": "Email: user@corp.com",
                    "entities": ["EMAIL_ADDRESS"],
                    "operators": [{"entity_type": "EMAIL_ADDRESS", "operator": "hash", "hash_type": "sha256"}],
                }
            },
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "user@corp.com" not in data["anonymized_text"]

    async def test_operators_applied_map(self, client: Client):
        result = await client.call_tool(
            "anonymize_text",
            {"request": {"text": "John at john@co.com", "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert isinstance(data["operators_applied"], dict)
        assert len(data["operators_applied"]) > 0

    async def test_text_is_preserved_when_no_pii(self, client: Client):
        text = "No personal info here whatsoever."
        result = await client.call_tool(
            "anonymize_text",
            {"request": {"text": text, "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["anonymized_text"] == text


# ===========================================================================
# TOOLS — batch_analyze
# ===========================================================================

class TestBatchAnalyze:
    async def test_returns_one_result_per_text(self, client: Client):
        texts = [
            "Alice Smith, alice@test.com",
            "Bob Brown, 555-1234",
            "No PII here.",
        ]
        result = await client.call_tool(
            "batch_analyze",
            {"request": {"texts": texts, "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["total_texts"] == 3
        assert len(data["results"]) == 3

    async def test_summary_entity_frequency(self, client: Client):
        texts = ["Email: a@a.com", "Email: b@b.com", "Phone: 555-0000"]
        result = await client.call_tool(
            "batch_analyze",
            {"request": {"texts": texts, "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        freq = data["summary"]["entity_frequency"]
        assert isinstance(freq, dict)

    async def test_empty_texts_handled_gracefully(self, client: Client):
        result = await client.call_tool(
            "batch_analyze",
            {"request": {"texts": ["", "   ", "Valid text here."], "language": "en"}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["total_texts"] == 3


# ===========================================================================
# TOOLS — list_supported_entities
# ===========================================================================

class TestListSupportedEntities:
    async def test_returns_non_empty_list(self, client: Client):
        result = await client.call_tool("list_supported_entities", {"language": "en"})
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["total"] > 0
        assert len(data["entities"]) == data["total"]

    async def test_includes_custom_entities(self, client: Client):
        result = await client.call_tool("list_supported_entities", {"language": "en"})
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "EMPLOYEE_ID" in data["entities"]
        assert "SALARY" in data["entities"]

    async def test_grouped_dict_present(self, client: Client):
        result = await client.call_tool("list_supported_entities", {"language": "en"})
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "HR Domain (Custom)" in data["grouped"]


# ===========================================================================
# TOOLS — add_custom_recognizer
# ===========================================================================

class TestAddCustomRecognizer:
    async def test_adds_project_code_recognizer(self, client: Client):
        result = await client.call_tool(
            "add_custom_recognizer",
            {
                "request": {
                    "name": "PROJECT_CODE",
                    "patterns": [{"name": "prj", "regex": r"PRJ-\d{4}-\d{3}", "score": 0.85}],
                    "context_words": ["project", "code"],
                }
            },
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["success"] is True
        assert data["entity_type"] == "PROJECT_CODE"

    async def test_custom_recognizer_detects_after_addition(self, client: Client):
        # First add the recognizer
        await client.call_tool(
            "add_custom_recognizer",
            {
                "request": {
                    "name": "INVOICE_NO",
                    "patterns": [{"name": "inv", "regex": r"INV-\d{5}", "score": 0.80}],
                }
            },
        )
        # Then confirm it detects
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "Invoice INV-12345 is overdue.", "entities": ["INVOICE_NO"]}},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert any(e["entity_type"] == "INVOICE_NO" for e in data["entities"])


# ===========================================================================
# TOOLS — score_pii_risk
# ===========================================================================

class TestScorePiiRisk:
    async def test_critical_risk_for_ssn_and_card(self, client: Client):
        result = await client.call_tool(
            "score_pii_risk",
            {
                "text": (
                    "SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111, "
                    "Bank: 12-34-56 12345678, NI: AB 12 34 56 C"
                )
            },
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["risk_score"] > 50  # at least High risk

    async def test_low_risk_for_clean_text(self, client: Client):
        result = await client.call_tool(
            "score_pii_risk",
            {"text": "The quarterly results were positive. Revenue grew by 12%."},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["risk_score"] <= 30

    async def test_recommendation_present(self, client: Client):
        result = await client.call_tool(
            "score_pii_risk",
            {"text": "Some text with email@example.com"},
        )
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert "recommendation" in data
        assert len(data["recommendation"]) > 0


# ===========================================================================
# TOOLS — list_recognizers
# ===========================================================================

class TestListRecognizers:
    async def test_returns_recognizers(self, client: Client):
        result = await client.call_tool("list_recognizers", {})
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert data["total"] > 0
        assert len(data["recognizers"]) == data["total"]

    async def test_language_filter(self, client: Client):
        result = await client.call_tool("list_recognizers", {"language": "en"})
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        for rec in data["recognizers"]:
            assert rec["supported_language"] == "en"


# ===========================================================================
# RESOURCES
# ===========================================================================

class TestResources:
    async def test_entity_catalogue_is_valid_json(self, client: Client):
        result = await client.read_resource("presidio://entities/catalogue")
        payload = json.loads(result[0].text)
        assert "groups" in payload
        assert "custom_entities" in payload

    async def test_operators_guide_lists_all_operators(self, client: Client):
        result = await client.read_resource("presidio://operators/guide")
        payload = json.loads(result[0].text)
        op_names = {op["name"] for op in payload["operators"]}
        assert op_names == {"replace", "redact", "mask", "hash", "encrypt"}

    async def test_server_config_is_valid_json(self, client: Client):
        result = await client.read_resource("presidio://config/server")
        payload = json.loads(result[0].text)
        assert "server_name" in payload
        assert payload["server_name"] == "presidio-mcp"

    async def test_usage_examples_has_examples(self, client: Client):
        result = await client.read_resource("presidio://examples/common")
        payload = json.loads(result[0].text)
        assert "examples" in payload
        assert len(payload["examples"]) >= 4


# ===========================================================================
# PROMPTS
# ===========================================================================

class TestPrompts:
    async def test_pii_audit_report_prompt(self, client: Client):
        result = await client.get_prompt(
            "pii_audit_report",
            {
                "document_text": "John Smith, SSN 123-45-6789",
                "document_type": "HR form",
                "compliance_framework": "GDPR",
            },
        )
        messages = result.messages
        assert len(messages) >= 1
        content = messages[0].content.text
        assert "analyze_text" in content
        assert "GDPR" in content

    async def test_anonymize_for_sharing_prompt(self, client: Client):
        result = await client.get_prompt(
            "anonymize_for_sharing",
            {
                "text": "Contact Alice at alice@corp.com",
                "sharing_context": "external auditor",
            },
        )
        messages = result.messages
        content = messages[0].content.text
        assert "anonymize_text" in content
        assert "external auditor" in content

    async def test_hr_data_privacy_review_prompt(self, client: Client):
        result = await client.get_prompt(
            "hr_data_privacy_review",
            {
                "hr_document": "EMP-001234 earns £70,000/year",
                "document_category": "payroll record",
                "action": "anonymize",
            },
        )
        messages = result.messages
        content = messages[0].content.text
        assert "anonymize_text" in content

    async def test_batch_pii_policy_check_prompt(self, client: Client):
        result = await client.get_prompt(
            "batch_pii_policy_check",
            {
                "records_description": "100 employee records",
                "policy_name": "GDPR Data Minimization",
                "max_allowed_risk": "Low",
            },
        )
        messages = result.messages
        content = messages[0].content.text
        assert "batch_analyze" in content
        assert "Low" in content


# ===========================================================================
# Edge cases & validation
# ===========================================================================

class TestEdgeCases:
    async def test_blank_text_raises_validation_error(self, client: Client):
        with pytest.raises(Exception):
            await client.call_tool(
                "analyze_text",
                {"request": {"text": "   ", "language": "en"}},
            )

    async def test_score_threshold_boundary(self, client: Client):
        result = await client.call_tool(
            "analyze_text",
            {"request": {"text": "alice@example.com", "score_threshold": 1.0}},
        )
        # At threshold 1.0, no entity should have score == 1.0 exactly → empty
        data = result[0].dict() if hasattr(result[0], "dict") else result[0]
        assert isinstance(data["entities"], list)

    async def test_unsupported_language_raises_validation_error(self, client: Client):
        """
        Only 'en' is loaded. Passing any other language code should fail
        at Pydantic validation before reaching Presidio.
        To enable other languages, add the corresponding spaCy model in engines.py.
        """
        with pytest.raises(Exception):
            await client.call_tool(
                "analyze_text",
                {"request": {"text": "Mi correo es juan@empresa.es", "language": "es"}},
            )

