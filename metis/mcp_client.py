"""MCP client — connects to Hermes (finance tools MCP server) and converts
discovered tools into LangChain-compatible tools for the LangGraph agent.

The user's JWT is forwarded to Hermes via the Authorization header on every
MCP request. Hermes uses it to authenticate Pluto calls — the JWT is never
exposed as a tool argument to the LLM.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPTool

from metis.config import get_settings
from metis.request_id import get_request_id

logger = logging.getLogger(__name__)


def _json_schema_to_pydantic(schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON Schema (from MCP tool inputSchema) to a Pydantic model.

    LangChain's StructuredTool requires a Pydantic model or a typed function.
    We build a dynamic model from the JSON Schema properties.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for name, prop in properties.items():
        py_type = type_map.get(prop.get("type", "string"), str)
        if name in required:
            fields[name] = (py_type, ...)
        else:
            default = prop.get("default")
            fields[name] = (py_type, default)

    # Pydantic requires at least one field; add a dummy if empty
    if not fields:
        fields["_placeholder"] = (str, "")

    return create_model("ToolArgs", **fields)


def _mcp_tool_to_langchain_tool(
    mcp_tool: MCPTool,
    client: Client,
) -> BaseTool:
    """Convert a single MCP tool into a LangChain StructuredTool.

    The tool calls the MCP server via `client.call_tool(name, args)` and
    extracts the text content from the result.
    """
    tool_name = mcp_tool.name
    tool_desc = mcp_tool.description or tool_name
    args_model = _json_schema_to_pydantic(mcp_tool.input_schema)

    async def _run(**kwargs: Any) -> str:
        # Remove placeholder field if present
        kwargs.pop("_placeholder", None)
        # Remove None values so optional params aren't sent
        args = {k: v for k, v in kwargs.items() if v is not None}
        try:
            result = await client.call_tool(tool_name, args)
            # Extract text from content blocks
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                else:
                    texts.append(str(block))
            return "\n".join(texts) if texts else json.dumps(
                {"result": result.structured_content}, ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"[MCP] Tool '{tool_name}' failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool_name,
        description=tool_desc,
        args_schema=args_model,
    )


async def discover_hermes_tools(auth_token: str) -> tuple[list[BaseTool], Client | None]:
    """Connect to Hermes, discover all tools, and return them as LangChain tools.

    Returns (tools, client) — the client must be kept alive while tools are
    in use. Returns ([], None) if Hermes is unreachable (best-effort: don't
    break the agent if MCP server is down).

    Args:
        auth_token: The user's JWT, forwarded to Hermes as Bearer token.
    """
    settings = get_settings()
    hermes_url = settings.hermes_base_url

    try:
        headers = {"Authorization": f"Bearer {auth_token}"}
        rid = get_request_id()
        if rid:
            headers["X-Request-ID"] = rid

        http_client = httpx.AsyncClient(
            headers=headers,
            timeout=settings.hermes_request_timeout_seconds,
        )

        transport = streamable_http_client(
            hermes_url,
            http_client=http_client,
        )

        client = Client(transport)
        await client.__aenter__()

        result = await client.list_tools()
        tools = [
            _mcp_tool_to_langchain_tool(t, client)
            for t in result.tools
        ]
        logger.info(f"[MCP] Discovered {len(tools)} tools from Hermes")
        return tools, client

    except BaseException as e:
        # Unwrap ExceptionGroup (Python 3.11+ — MCP SDK wraps errors)
        detail = str(e)
        if hasattr(e, 'exceptions'):
            sub_details = []
            for sub in e.exceptions:
                if hasattr(sub, 'exceptions'):
                    for sub2 in sub.exceptions:
                        sub_details.append(f"{type(sub2).__name__}: {sub2}")
                else:
                    sub_details.append(f"{type(sub).__name__}: {sub}")
            detail = "; ".join(sub_details) if sub_details else str(e)
        logger.warning(f"[MCP] Failed to connect to Hermes at {hermes_url}: {detail}")
        return [], None


async def close_hermes_client(client: Client | None) -> None:
    """Close the MCP client connection."""
    if client is not None:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"[MCP] Error closing Hermes client: {e}")
