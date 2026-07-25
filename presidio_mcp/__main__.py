"""
Entry point for running the Presidio MCP Server as a module.

Usage:
    python -m presidio_mcp                          # stdio (default)
    python -m presidio_mcp --transport sse          # HTTP/SSE on port 8001
    python -m presidio_mcp --transport sse --port 9000
"""

from presidio_mcp.server import mcp
import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Presidio MCP Server — PII detection & anonymization via MCP"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind when using SSE transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port to bind when using SSE transport (default: 8001)"
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
