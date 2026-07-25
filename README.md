# Presidio MCP Server

A production-ready **Model Context Protocol (MCP) server** that exposes [Microsoft Presidio](https://microsoft.github.io/presidio/) PII detection and anonymization capabilities using [FastMCP 2.x](https://gofastmcp.com).

## Features

| Primitive | Count | Description |
|-----------|-------|-------------|
| 🔧 Tools | 8 | PII analysis, anonymization, batch processing, risk scoring |
| 📦 Resources | 4 | Entity catalogue, operator guide, server config, examples |
| 💬 Prompts | 4 | Audit reports, sharing workflows, HR reviews, batch checks |

### Tools

| Tool | Description |
|------|-------------|
| `analyze_text` | Detect PII entities with confidence scores & character offsets |
| `anonymize_text` | Anonymize PII with configurable per-entity operators |
| `deanonymize_text` | Reverse AES-encrypted anonymization |
| `batch_analyze` | Analyze up to 50 texts in one call |
| `list_supported_entities` | Return all detectable entity types |
| `add_custom_recognizer` | Dynamically register regex-based recognizers at runtime |
| `score_pii_risk` | Compute a 0–100 PII risk score with recommendations |
| `list_recognizers` | List all active recognizers with metadata |

### Resources (read-only reference data)

| URI | Description |
|-----|-------------|
| `presidio://entities/catalogue` | All supported PII types grouped by category |
| `presidio://operators/guide` | Anonymization operator reference with examples |
| `presidio://config/server` | Live server configuration snapshot |
| `presidio://examples/common` | Ready-to-use usage examples |

### Prompts (reusable workflow templates)

| Prompt | Use case |
|--------|----------|
| `pii_audit_report` | Generate a compliance-ready PII audit report |
| `anonymize_for_sharing` | Anonymize a document before external sharing |
| `hr_data_privacy_review` | HR-specific PII review and anonymization |
| `batch_pii_policy_check` | Run policy compliance checks across record batches |

### HR-Domain Custom Recognizers

Beyond Presidio's built-in NLP, this server includes custom recognizers for:

- **EMPLOYEE_ID** — `EMP-XXXXXX` patterns
- **SALARY** — Currency amounts in USD/GBP/EUR + k-notation
- **UK_NINO** — UK National Insurance Numbers
- **US_SSN** — US Social Security Numbers (improved patterns)
- **BANK_ACCOUNT** — UK sort codes + IBAN prefixes
- **PASSPORT** — US and UK passport number formats

---

## Project Structure

```
mcp-server/
├── presidio_mcp/
│   ├── __init__.py          # Package metadata
│   ├── __main__.py          # Module entry point (python -m presidio_mcp)
│   ├── server.py            # FastMCP server — tools, resources, prompts
│   ├── engines.py           # Singleton Presidio engine initialization
│   ├── models.py            # Pydantic input/output models
│   └── recognizers.py      # HR-domain custom recognizers
├── tests/
│   └── test_presidio_mcp.py # Comprehensive test suite
├── pyproject.toml           # Project config & dependencies
├── requirements.txt         # pip-installable dependencies
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites

- Python 3.10+
- pip or [uv](https://docs.astral.sh/uv/) (recommended)

### 2. Create virtual environment

```bash
cd mcp-server
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Production
pip install -r requirements.txt

# Or with dev tools (testing, linting):
pip install -e ".[dev]"
```

### 4. Download the spaCy NLP model

```bash
# Recommended (higher accuracy, ~741MB):
python -m spacy download en_core_web_lg

# Lightweight fallback (~12MB, auto-used if lg is missing):
python -m spacy download en_core_web_sm
```

---

## Running the Server

### stdio transport (Claude Desktop, Cursor, Continue)

```bash
python -m presidio_mcp
# or
presidio-mcp
```

### HTTP/SSE transport (remote clients, testing)

```bash
python -m presidio_mcp --transport sse --host 0.0.0.0 --port 8001
```

---

## Claude Desktop Integration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "presidio": {
      "command": "/path/to/mcp-server/.venv/bin/python",
      "args": ["-m", "presidio_mcp"],
      "cwd": "/path/to/mcp-server"
    }
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- All 8 tools (happy paths + edge cases)
- All 4 resources (JSON validity, content checks)
- All 4 prompts (message structure, keyword presence)
- Validation errors (blank text, boundary scores)
- Multi-language support

---

## Anonymization Operators Quick Reference

| Operator | Reversible | Best For |
|----------|-----------|----------|
| `replace` | ✗ | General de-identification |
| `redact` | ✗ | Complete removal |
| `mask` | ✗ | Partial masking (credit cards) |
| `hash` | ✗ | Consistent pseudonymization |
| `encrypt` | ✅ | Reversible by authorised parties |

---

## Example Usage (via LLM)

**Analyze HR document:**
```
Use analyze_text with:
  text: "New hire John Smith (EMP-001234), SSN 123-45-6789, salary $95,000"
  entities: ["PERSON", "EMPLOYEE_ID", "US_SSN", "SALARY"]
```

**Anonymize for sharing:**
```
Use anonymize_text with:
  text: "Contact Alice at alice@corp.com, card: 4111-1111-1111-1111"
  operators:
    - entity_type: EMAIL_ADDRESS, operator: replace
    - entity_type: CREDIT_CARD, operator: mask, chars_to_mask: 12
```

**Risk-score a document:**
```
Use score_pii_risk with:
  text: "Full employee record with SSN, bank account, and salary details"
```

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Singleton engines** | Presidio engine init is expensive (~2-5s). One instance per process. |
| **Pydantic models as tool params** | FastMCP uses model schemas for LLM tool descriptions + strict validation |
| **Graceful spaCy fallback** | Auto-falls back to `en_core_web_sm` if `en_core_web_lg` is missing |
| **`readOnlyHint` annotations** | Read-only tools skip confirmation prompts in MCP clients |
| **Thread-safe double-checked locking** | Engines are safe for concurrent asyncio tool calls |
| **stderr for logs** | MCP stdio protocol uses stdout; logs go to stderr to avoid interference |

---

## License

MIT
```
