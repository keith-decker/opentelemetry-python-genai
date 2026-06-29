# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, cast

from langchain_core.messages import BaseMessage

from opentelemetry.util.genai.types import (
    File,
    FunctionToolDefinition,
    GenericPart,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    Uri,
)

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
}


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def prepare_tool_definitions(tools: list[Any]) -> list[ToolDefinition] | None:
    if not tools:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = _get_property_value(tool, "type")
        if tool_type == "function":
            func = _get_property_value(tool, "function")
            if func:
                func_name = _get_property_value(func, "name")
                func_description = _get_property_value(func, "description")
                definitions.append(
                    FunctionToolDefinition(
                        name=str(func_name) if func_name is not None else "",
                        description=str(func_description)
                        if func_description is not None
                        else None,
                        parameters=_get_property_value(func, "parameters"),
                    )
                )
    return definitions or None


def make_input_message(data: Any) -> list[InputMessage]:
    """Create structured input messages from LangChain chain input data."""
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    messages: Any = data_dict.get("messages")
    if messages is not None:
        return make_input_messages_from_messages(
            messages, normalize_roles=True
        )

    # Fallback: serialize non-message state fields as input.
    # Common in LangGraph where nodes use structured state fields
    # (e.g., user_query) rather than a message list.
    exclude_keys = {"messages", "intermediate_steps"}
    input_data: dict[str, Any] = {
        k: v
        for k, v in data_dict.items()
        if k not in exclude_keys and v is not None
    }
    if input_data:
        serialized = serialize(input_data)
        if serialized:
            return [InputMessage(role="user", parts=[Text(serialized)])]
    return []


def make_input_messages_from_messages(
    messages: Any, *, normalize_roles: bool = False
) -> list[InputMessage]:
    """Convert LangChain message objects and raw dict messages to GenAI input messages."""
    input_messages: list[InputMessage] = []
    for message in _iter_messages(messages):
        genai_message = _message_to_input_message(
            message, normalize_roles=normalize_roles
        )
        if genai_message is not None:
            input_messages.append(genai_message)
    return input_messages


def make_output_message(data: Any) -> list[OutputMessage]:
    """Create structured output messages from LangChain chain output data."""
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    messages: list[Any] | None = data_dict.get("messages")
    if messages is None:
        return []
    return [
        message
        for raw_message in messages
        if (
            message := _message_to_output_message(
                raw_message,
                finish_reason="stop",
                normalize_roles=True,
                assistant_only=True,
            )
        )
        is not None
    ]


def make_output_messages_from_generations(
    generations: Any,
) -> list[OutputMessage]:
    """Convert LangChain LLMResult generations to GenAI output messages."""
    output_messages: list[OutputMessage] = []
    for generation in _iter_generation_items(generations):
        message = _get_value(generation, "message")
        if message is None:
            text = _get_value(generation, "text")
            if isinstance(text, str) and text:
                output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=[Text(text)],
                        finish_reason=_get_finish_reason(generation),
                    )
                )
            continue

        output_message = _message_to_output_message(
            message,
            finish_reason=_get_finish_reason(generation),
            normalize_roles=False,
            assistant_only=False,
        )
        if output_message is not None:
            output_messages.append(output_message)
    return output_messages


def make_last_output_message(data: Any) -> list[OutputMessage]:
    """Extract only the last AI message as the output.

    For Workflow and AgentInvocation spans, the final AI message best represents
    the actual output. Intermediate AI messages (e.g., tool-call decisions) are
    already captured in child LLM invocation spans.
    """
    all_messages = make_output_message(data)
    if all_messages:
        return [all_messages[-1]]
    return []


def make_retrieval_document(document: Any) -> dict[str, Any]:
    """Convert a LangChain document into a semconv retrieval document object."""
    result: dict[str, Any] = {}
    metadata = _get_value(document, "metadata")
    page_content = _get_value(document, "page_content")
    document_id = _get_retrieval_document_id(document, metadata, page_content)
    if document_id is not None:
        result["id"] = document_id
    result["score"] = _get_retrieval_document_score(document, metadata)
    if page_content is not None:
        result["content"] = str(page_content)
    if metadata is not None:
        result["metadata"] = metadata
    return result


def _get_retrieval_document_id(
    document: Any, metadata: Any, page_content: Any
) -> str | None:
    if (document_id := _get_value(document, "id")) is not None:
        return str(document_id)

    if isinstance(metadata, Mapping):
        for metadata_key in ("id", "document_id", "source"):
            if (document_id := metadata.get(metadata_key)) is not None:
                return str(document_id)

    serialized = serialize({"content": page_content, "metadata": metadata})
    if serialized:
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return None


def _get_retrieval_document_score(document: Any, metadata: Any) -> float:
    for score_source in (document, metadata):
        if score_source is None:
            continue
        for score_key in ("score", "relevance_score", "similarity_score"):
            score = _get_value(score_source, score_key)
            if score is None:
                continue
            try:
                return float(score)
            except (TypeError, ValueError):
                continue
    return 1.0


def serialize(obj: Any) -> Optional[str]:
    """Serialize object to JSON string.

    Uses default=str to handle non-JSON-serializable objects (like LangChain
    message objects) by converting them to their string representation while
    keeping the overall structure as valid JSON.
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _message_to_input_message(
    message: Any, *, normalize_roles: bool
) -> InputMessage | None:
    role = _message_role(message, normalize=normalize_roles)
    parts = _message_parts(message)
    if not parts:
        return None
    return InputMessage(role=role, parts=parts)


def _message_to_output_message(
    message: Any,
    *,
    finish_reason: str,
    normalize_roles: bool,
    assistant_only: bool,
) -> OutputMessage | None:
    role = _message_role(message, normalize=normalize_roles)
    if assistant_only and role not in ("assistant", "ai"):
        return None
    parts = _message_parts(message)
    if not parts:
        return None
    return OutputMessage(
        role=role,
        parts=parts,
        finish_reason=finish_reason,
    )


def _message_parts(message: Any) -> list[MessagePart]:
    role = _message_role(message, normalize=True)
    content = _message_content(message)
    if role in ("tool", "function"):
        return [
            ToolCallResponse(
                id=_string_or_none(_get_value(message, "tool_call_id")),
                response=content,
            )
        ]

    parts = _content_to_parts(content)
    if role == "assistant":
        parts.extend(_tool_call_parts(message))
    return parts


def _content_to_parts(content: Any) -> list[MessagePart]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [Text(content)]
    if isinstance(content, Mapping):
        return [_content_mapping_to_part(content)]
    if isinstance(content, Iterable) and not isinstance(content, (bytes, str)):
        parts: list[MessagePart] = []
        for item in content:
            if item is None or item == "":
                continue
            if isinstance(item, str):
                parts.append(Text(item))
            elif isinstance(item, Mapping):
                parts.append(_content_mapping_to_part(item))
            else:
                parts.append(GenericPart(value=item))
        return parts
    return [GenericPart(value=content)]


def _content_mapping_to_part(content: Mapping[str, Any]) -> MessagePart:
    content_type = content.get("type")
    if content_type in ("text", "input_text", "output_text") or (
        content_type is None and isinstance(content.get("text"), str)
    ):
        return Text(str(content.get("text") or content.get("content") or ""))

    if content_type == "image_url":
        image_url = content.get("image_url")
        uri = image_url.get("url") if isinstance(image_url, Mapping) else image_url
        if isinstance(uri, str):
            return Uri(
                mime_type=_mime_type_from_uri(uri),
                modality="image",
                uri=uri,
            )

    if content_type in ("image", "audio", "video", "file"):
        file_id = content.get("file_id")
        if isinstance(file_id, str):
            return File(
                mime_type=_string_or_none(content.get("mime_type")),
                modality=str(content_type),
                file_id=file_id,
            )

    return GenericPart(value=dict(content))


def _tool_call_parts(message: Any) -> list[ToolCallRequest]:
    parts: list[ToolCallRequest] = []
    for tool_call in _message_tool_calls(message):
        if part := _tool_call_part(tool_call):
            parts.append(part)

    additional_kwargs = _get_value(message, "additional_kwargs")
    if isinstance(additional_kwargs, Mapping):
        function_call = additional_kwargs.get("function_call")
        if isinstance(function_call, Mapping):
            parts.append(
                ToolCallRequest(
                    id=None,
                    name=str(function_call.get("name") or ""),
                    arguments=_tool_arguments(
                        function_call.get("arguments")
                    ),
                )
            )
    return parts


def _message_tool_calls(message: Any) -> list[Any]:
    tool_calls = _get_value(message, "tool_calls")
    if tool_calls:
        return list(tool_calls)
    additional_kwargs = _get_value(message, "additional_kwargs")
    if isinstance(additional_kwargs, Mapping) and additional_kwargs.get(
        "tool_calls"
    ):
        return list(additional_kwargs["tool_calls"])
    return []


def _tool_call_part(tool_call: Any) -> ToolCallRequest | None:
    function = _get_value(tool_call, "function")
    if function is not None:
        name = _get_value(function, "name") or ""
        arguments = _get_value(function, "arguments")
    else:
        name = _get_value(tool_call, "name") or ""
        arguments = (
            _get_value(tool_call, "args")
            if _get_value(tool_call, "args") is not None
            else _get_value(tool_call, "arguments")
        )
    return ToolCallRequest(
        id=_string_or_none(_get_value(tool_call, "id")),
        name=str(name),
        arguments=_tool_arguments(arguments),
    )


def _tool_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _iter_messages(messages: Any) -> Iterable[Any]:
    if isinstance(messages, (str, bytes)) or isinstance(messages, Mapping):
        yield messages
        return
    if isinstance(messages, Sequence):
        if len(messages) == 2 and isinstance(messages[0], str):
            yield {"role": messages[0], "content": messages[1]}
            return
        for item in messages:
            yield from _iter_messages(item)
        return
    yield messages


def _iter_generation_items(generations: Any) -> Iterable[Any]:
    if not generations:
        return
    for generation in generations:
        if isinstance(generation, Iterable) and not isinstance(
            generation, (Mapping, str, bytes)
        ):
            yield from generation
        else:
            yield generation


def _message_role(message: Any, *, normalize: bool) -> str:
    role = _get_value(message, "role") or _get_value(message, "type")
    if role is None and isinstance(message, BaseMessage):
        role = message.type
    if role is None:
        role = "user"
    role_str = str(role)
    if normalize:
        return _ROLE_MAP.get(role_str, role_str)
    return role_str


def _message_content(message: Any) -> Any:
    if isinstance(message, Mapping):
        if "content" in message:
            return message.get("content")
        kwargs = message.get("kwargs")
        if isinstance(kwargs, Mapping):
            return kwargs.get("content")
    return getattr(message, "content", None)


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        if key in obj:
            return obj.get(key)
        kwargs = obj.get("kwargs")
        if isinstance(kwargs, Mapping) and key in kwargs:
            return kwargs.get(key)
    return getattr(obj, key, None)


def _get_finish_reason(generation: Any) -> str:
    generation_info = _get_value(generation, "generation_info")
    if isinstance(generation_info, Mapping):
        finish_reason = generation_info.get("finish_reason")
        if finish_reason is not None:
            return str(finish_reason)

    message = _get_value(generation, "message")
    response_metadata = _get_value(message, "response_metadata")
    if isinstance(response_metadata, Mapping):
        for key in ("finish_reason", "stopReason"):
            finish_reason = response_metadata.get(key)
            if finish_reason is not None:
                return str(finish_reason)
    return "unknown"


def _mime_type_from_uri(uri: str) -> str | None:
    if uri.startswith("data:") and "," in uri:
        header = uri.split(",", 1)[0]
        return header[5:].split(";", 1)[0] or None
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
