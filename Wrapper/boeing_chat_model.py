"""
Enhanced Boeing Chat Model with Tool Calling Support
LangChain-compatible wrapper for Boeing's proprietary BCAI API
Supports: Tool calling, Structured output, Async operations, Streaming
"""

import json
import time
import requests
import httpx
import asyncio
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Iterator,
    Type,
    Union,
)

from uuid import uuid4
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

import logging
logger = logging.getLogger("BoeingChatModel")


class BoeingChatModel(BaseChatModel, BaseModel):
    """
    Enhanced LangChain chat model wrapper for Boeing BCAI API.

    Features:
    - Full tool calling support with bind_tools()
    - Structured output via with_structured_output()
    - Async operations (_agenerate)
    - Streaming support (_stream)
    - Token counting integration
    - Compatible with LangChain agents and chains
    """

    # ==================== Configuration Fields ====================

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    api_url: str = Field(
        default="https://bcai-test.web.boeing.com/bcai-public-api/conversation",
        description="The API endpoint for chat completions."
    )

    token_counter_url: str = Field(
        default="https://bcai-test.web.boeing.com/bcai-public-api/countTokens",
        description="The API endpoint for counting tokens."
    )

    udal_pat: str = Field(
        ...,
        description="The UDAL_PAT token for authorization."
    )

    model: str = Field(
        default="gpt-4.1-mini",
        description="The model to use (e.g., gpt-4o-mini, gpt-4o)."
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature between 0 and 2."
    )

    top_p: Optional[float] = Field(
    default=None,
    description=(
        "Nucleus sampling parameter (0–1). Only the tokens whose cumulative "
        "probability mass reaches top_p are considered at each step. "
        "Do not set alongside temperature — use one or the other. "
        "Primary sampling control for reasoning models where temperature must be None."
        )
    )

    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Controls how many hidden reasoning tokens a reasoning model (gpt-5.x, o4-mini) "
            "spends before producing a visible response. "
            "Accepted values: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'. "
            "Has no effect on standard models (gpt-4.1 family). "
            "Omitted from the payload entirely when None."
        )
    )

    frequency_penalty: Optional[float] = Field(
        default=None,
        description=(
            "Penalises tokens proportionally to how many times they have already "
            "appeared in the output. Higher values reduce word-level repetition. "
            "Range: -2.0 to 2.0. Use small positive values (0.1–0.2) on the Drafter "
            "to discourage padding; keep at 0.0 for Dispatcher and Evaluator."
        )
    )

    presence_penalty: Optional[float] = Field(
        default=None,
        description=(
            "Applies a flat penalty to any token that has appeared at least once in "
            "the output, encouraging topic diversity rather than word-frequency reduction. "
            "Range: -2.0 to 2.0. Use cautiously on avionics content — can cause the model "
            "to avoid precise technical terms it has already used."
        )
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate."
    )

    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds."
    )

    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests."
    )

    # Private attributes for tool binding
    _bound_tools: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)
    _tool_choice: Optional[Union[str, Dict[str, Any]]] = PrivateAttr(default=None)

    # ==================== Required LangChain Properties ====================

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "boeing_chat_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Returns the parameters that uniquely identify this model instance for
        LangChain tracing and logging. Only includes parameters that are set
        (non-None) to keep trace payloads clean.
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "api_url": self.api_url,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        return params

    # ==================== Helper Methods ====================

    def _get_headers(self) -> Dict[str, str]:
        """Construct HTTP headers for API requests."""
        return {
            "accept": "application/json",
            "Authorization": f"basic {self.udal_pat}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _convert_messages_to_api_format(
        messages: Sequence[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """
        Convert LangChain messages to Boeing API format.
        Supports: HumanMessage, AIMessage, SystemMessage, ToolMessage
        """
        api_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                api_messages.append({
                    "role": "user",
                    "content": str(message.content)
                })

            elif isinstance(message, AIMessage):
                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": str(message.content) if message.content else ""
                }

                # Add tool_calls if present
                if hasattr(message, "tool_calls") and message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"])
                            }
                        }
                        for tc in message.tool_calls
                    ]

                api_messages.append(msg_dict)

            elif isinstance(message, SystemMessage):
                api_messages.append({
                    "role": "system",
                    "content": str(message.content)
                })

            elif isinstance(message, ToolMessage):
                api_messages.append({
                    "role": "tool",
                    "content": str(message.content),
                    "tool_call_id": message.tool_call_id
                })

            else:
                # Fallback for unknown message types
                api_messages.append({
                    "role": "user",
                    "content": str(message.content)
                })

        return api_messages

    def _build_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Constructs the JSON request body sent to the BCAI conversation endpoint.

        All optional sampling and generation parameters are included only when
        explicitly set (non-None). This matches BCAI's Azure OpenAI proxy
        behaviour where omitting a field lets the backend apply its own default,
        which is safer than sending null values for unsupported model families.

        Temperature and top_p are mutually exclusive sampling controls:
            - Standard models (gpt-4.1 family): set temperature, omit top_p.
            - Reasoning models (gpt-5.x, o4-mini): omit temperature entirely,
              optionally set top_p as the sampling control.

        reasoning_effort is silently ignored by BCAI for non-reasoning models,
        so it is always safe to include when set — it only takes effect on
        models that support it (gpt-5.x, o4-mini).
        """
        api_messages = self._convert_messages_to_api_format(messages)

        payload: Dict[str, Any] = {
            "messages": api_messages,
            "model": self.model,
            "stream": False,
            "conversation_guid": str(uuid4()),
        }

        # Add tools if bound
        if self._bound_tools:
            payload["tools"] = self._bound_tools
            if self._tool_choice:
                payload["tool_choice"] = self._tool_choice

        # Add optional parameters
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            payload["response_max_tokens"] = self.max_tokens
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if stop:
            payload["stop"] = stop

        # Override with any kwargs
        payload.update(kwargs)

        return payload

    def _process_tool_calls(
        self,
        tool_calls_data: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """
        Process raw tool call data from API response.

        Args:
            tool_calls_data: List of tool calls from API response

        Returns:
            List of ToolCall objects for LangChain
        """
        tool_calls = []

        for tool_call in tool_calls_data:
            try:
                function_data = tool_call.get("function", {})
                tool_calls.append(
                    ToolCall(
                        name=function_data.get("name", ""),
                        args=json.loads(function_data.get("arguments", "{}")),
                        id=tool_call.get("id", f"call_{uuid4().hex[:24]}"),
                    )
                )
            except (KeyError, json.JSONDecodeError) as e:
                # Log warning but continue processing other tool calls
                print(f"Warning: Failed to parse tool call: {e}")
                continue

        return tool_calls

    def _process_response(self, data: Dict[str, Any]) -> ChatResult:
        """
        Process API response and return ChatResult.
        Handles both regular text responses and tool calling responses.
        """
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in API response")

        choice = choices[0]
        finish_reason = choice.get("finish_reason", "stop")
        msg_data = choice.get("message", {})

        # Check if this is a tool call response
        if "tool_calls" in msg_data and msg_data["tool_calls"]:
            tool_calls = self._process_tool_calls(msg_data["tool_calls"])
            message = AIMessage(
                content=msg_data.get("content", "") or "",
                tool_calls=tool_calls
            )
        else:
            # Regular text response
            message = AIMessage(content=msg_data.get("content", ""))

        # Extract token usage
        usage_data = data.get("usage", {})
        token_usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

        llm_output: Dict[str, Any] | None = {
            "token_usage": token_usage,
            "model_name": data.get("model", self.model),
            "finish_reason": finish_reason,
        }

        # Attach usage to response_metadata on the message itself.
        # This makes token counts accessible directly from the AIMessage
        # returned by ainvoke() — without needing to inspect ChatResult.llm_output.
        # Nodes can read msg.response_metadata["usage"] after every ainvoke call.
        if isinstance(message, AIMessage):
            message.response_metadata["usage"] = token_usage

        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output=llm_output
        )

    # ==================== Core Generation Methods ====================

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Synchronous chat generation (required by BaseChatModel).

        Args:
            messages: List of messages in the conversation
            stop: List of stop sequences
            run_manager: Callback manager for tracing
            **kwargs: Additional parameters

        Returns:
            ChatResult containing the model's response
        """
        payload = self._build_payload(messages, stop, **kwargs)

        # Make request with retries
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout,
                    verify=False
                )
                response.raise_for_status()
                data = response.json()
                return self._process_response(data)

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

        raise RuntimeError(
            f"Failed to call Boeing API after {self.max_retries} attempts: {last_exception}"
        ) from last_exception

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async chat generation for concurrent operations.

        Args:
            messages: List of messages in the conversation
            stop: List of stop sequences
            run_manager: Async callback manager
            **kwargs: Additional parameters

        Returns:
            ChatResult containing the model's response
        """
        payload = self._build_payload(messages, stop, **kwargs)

        # Make async request with retries
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.post(
                        self.api_url,
                        headers=self._get_headers(),
                        json=payload,
                        timeout=self.timeout
                    )
                response.raise_for_status()
                data = response.json()
                return self._process_response(data)

            except (httpx.HTTPError, httpx.RequestError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue

        raise RuntimeError(
            f"Failed to call Boeing API after {self.max_retries} attempts: {last_exception}"
        ) from last_exception

    async def _agenerate_with_cache(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """
        Override LangChain's default routing to always use _agenerate directly.

        By default, when _astream is defined on a BaseChatModel, LangChain's
        _agenerate_with_cache routes ALL async generation (including ainvoke)
        through _astream. This breaks the dispatcher and evaluator nodes because
        _astream only handles text deltas — tool call and structured output
        responses yield zero chunks, causing 'No generations found in stream'.

        This override restores the correct behavior:
            ainvoke (dispatcher, evaluator) → _agenerate → httpx non-streaming
            astream (drafter)               → _astream   → httpx streaming
        """
        return await self._agenerate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        payload = self._build_payload(messages, stop, **kwargs)
        payload["stream"] = True

        # Track last token time to avoid infinite hangs
        last_token_time = time.time()
        MAX_IDLE_SECONDS = 30

        try:
            response = requests.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                stream=True,
                timeout=(10, self.timeout),  # (connect_timeout, read_timeout)
                verify=False
            )
            response.raise_for_status()

            for raw_line in response.iter_lines():
                # Idle timeout protection
                if time.time() - last_token_time > MAX_IDLE_SECONDS:
                    break

                if not raw_line:
                    continue

                # Normalize bytes -> str
                line = raw_line.decode("utf-8", errors="ignore").strip()

                # Optional debug
                # print("RAW_STREAM_LINE:", repr(line))

                # (Optional) SSE compatibility
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()

                if line == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    continue

                finish_reason = choices[0].get("finish_reason")

                # IMPORTANT: end stream as soon as finish_reason appears (even if no delta)
                if finish_reason:
                    break

                # Boeing delta extraction (your observed format)
                delta_content = ""
                messages_list = choices[0].get("messages", [])
                if messages_list:
                    delta_content = messages_list[0].get("delta", "") or ""

                if delta_content:
                    last_token_time = time.time()
                    yield ChatGenerationChunk(message=AIMessageChunk(content=delta_content))
                    if run_manager:
                        run_manager.on_llm_new_token(delta_content)


        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Streaming error: {e}") from e

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async streaming generation using httpx.AsyncClient.
        Direct async equivalent of _stream() — same NDJSON delta extraction format.

        BCAI Wire Format (established from diagnostic — all models):
            - Newline-delimited JSON (NDJSON): one complete JSON object per line
            - No 'data:' SSE prefix — raw JSON sent directly
            - No keepalive lines or empty line separators between chunks
            - transfer-encoding: chunked in streaming mode
            - Final two lines carry finish_reason and usage respectively,
              neither contains a delta — timer must reset on these too

        Idle timeout strategy:
            MAX_IDLE_SECONDS resets on every successfully parsed JSON line,
            not just token-bearing lines. This covers:
                - Normal token chunks (delta present)
                - finish_reason chunk (no delta, signals stream end)
                - Final usage chunk (no delta, carries token counts)
                - Any future BCAI-side structural changes
            The timer only fires when the HTTP connection goes completely
            silent — no bytes at all — which indicates a genuine BCAI
            outage or proxy failure, not normal reasoning model latency.

        Timeout values:
            MAX_IDLE_SECONDS = 120 — safety net for genuine dead connections.
                Reasoning models (gpt-5.x) take 6-30s before first token
                under normal load. 120s gives headroom for BCAI load spikes
                without being reckless about truly dead connections.
            httpx read timeout = self.timeout (300s default) — outer hard
                ceiling for the entire request duration.

        Key behavioral notes:
            - Does NOT break on finish_reason — continues reading until the
              usage chunk arrives after finish_reason: "stop". Breaking early
              would lose token counts.
            - Yields one final metadata-only chunk carrying usage data after
              all tokens are exhausted. Server layer reads this for token tracking.
            - Usage chunk has empty content so frontend ignores it naturally.
            - Raises descriptive RuntimeError on idle timeout so the error
              message in logs and SSE events is immediately actionable.
        """
        payload = self._build_payload(messages, stop, **kwargs)
        payload["stream"] = True

        # ── Idle timeout tracking ──────────────────────────────────────────────
        # Resets on every successfully parsed JSON line — not just token lines.
        # Covers finish_reason and usage trailing chunks which have no delta.
        # Only fires when BCAI sends zero bytes — genuine dead connection.
        last_activity_time = time.time()
        MAX_IDLE_SECONDS = 120  # increased from 30 — reasoning models need headroom

        # Accumulated usage — populated from the final chunk after finish_reason.
        # Always the last line of the BCAI stream.
        usage_data: Dict[str, int] = {}

        # Incomplete line buffer — handles the rare case where a TCP chunk
        # boundary splits a JSON line mid-way. Accumulated here until \n arrives.
        line_buffer = ""

        try:
            async with httpx.AsyncClient(
                    verify=False,
                    timeout=httpx.Timeout(10.0, read=self.timeout)
            ) as client:
                async with client.stream(
                        "POST",
                        self.api_url,
                        headers=self._get_headers(),
                        json=payload
                ) as response:
                    response.raise_for_status()

                    async for raw_chunk in response.aiter_bytes():
                        # ── Idle timeout check ────────────────────────────────
                        # Checked on every byte chunk arrival, not per parsed line.
                        # If BCAI sends any bytes — even partial JSON — the
                        # connection is alive. Timer reset happens after parse below.
                        if time.time() - last_activity_time > MAX_IDLE_SECONDS:
                            raise RuntimeError(
                                f"_astream: idle timeout — no activity from BCAI "
                                f"for {MAX_IDLE_SECONDS}s. "
                                f"Model={self.model}. "
                                f"Connection may be dead or BCAI is overloaded."
                            )

                        # ── Decode and buffer ─────────────────────────────────
                        # Append incoming bytes to the line buffer.
                        # BCAI streams NDJSON — one JSON object per \n-terminated line.
                        # A single TCP chunk may contain multiple lines or split one.
                        line_buffer += raw_chunk.decode("utf-8", errors="replace")

                        # ── Process all complete lines in the buffer ──────────
                        while "\n" in line_buffer:
                            line, line_buffer = line_buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue

                            # Strip SSE prefix if ever present (defensive)
                            if line.startswith("data:"):
                                line = line[len("data:"):].strip()

                            if line == "[DONE]":
                                return

                            try:
                                chunk_data = json.loads(line)
                            except json.JSONDecodeError:
                                # Partial line — rare but possible on very large
                                # chunks. Will be completed in the next iteration.
                                line_buffer = line + "\n" + line_buffer
                                continue

                            # ── Reset activity timer on every valid JSON line ─
                            # This is the key fix — resets on finish_reason and
                            # usage lines too, not just token-bearing lines.
                            last_activity_time = time.time()

                            # ── Capture usage from final chunk ────────────────
                            raw_usage = chunk_data.get("usage")
                            if raw_usage and isinstance(raw_usage, dict):
                                usage_data = {
                                    "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                                    "completion_tokens": raw_usage.get("completion_tokens", 0),
                                    "total_tokens": raw_usage.get("total_tokens", 0),
                                }

                            # ── Extract delta token ───────────────────────────
                            # BCAI streaming format: choices[0].messages[0].delta
                            # Differs from standard OpenAI: choices[0].delta.content
                            choices = chunk_data.get("choices", [])
                            if not choices:
                                continue

                            messages_list = choices[0].get("messages", [])
                            delta_content = ""
                            if messages_list:
                                delta_content = messages_list[0].get("delta", "") or ""

                            if delta_content:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=delta_content)
                                )
                                yield chunk
                                if run_manager:
                                    await run_manager.on_llm_new_token(delta_content)

            # ── Yield final usage chunk ────────────────────────────────────────
            # Emitted after the stream ends. Empty content so frontend ignores it.
            # Server layer reads response_metadata["usage"] for token tracking.
            if usage_data:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={"usage": usage_data}
                    ),
                    generation_info={"usage": usage_data}
                )
                logger.info(
                    f"_astream usage — prompt: {usage_data.get('prompt_tokens')}, "
                    f"completion: {usage_data.get('completion_tokens')}, "
                    f"total: {usage_data.get('total_tokens')}"
                )

        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text
            except Exception:
                pass
            raise RuntimeError(
                f"BCAI API error during async stream: {e.response.status_code} - {error_body}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Network error during async stream: {e}"
            ) from e

    # ==================== Tool Calling Support ====================

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, bool]] = None,
        **kwargs: Any,
    ) -> "BoeingChatModel":
        """
        Bind tool-like objects to this chat model.

        Args:
            tools: List of tools to bind (LangChain tools, functions, or dicts)
            tool_choice: Which tool to require(Only supports):
                - "auto" (default): Model decides
                - "none": Don't call any tool
                - "required": Must call a tool
            **kwargs: Additional parameters

        Returns:
            New BoeingChatModel instance with tools bound
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        # Convert all tools to OpenAI format
        formatted_tools: List[Dict[str, Any]] = [
            convert_to_openai_tool(tool) for tool in tools
        ]

        # Process tool_choice parameter(Boeing API Restriction only accepts: "none", "auto" and "required")
        formatted_tool_choice: Optional[str] = None
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                if tool_choice in ("auto", "none", "required"):
                    formatted_tool_choice = tool_choice
                elif tool_choice == "any":
                    formatted_tool_choice = "required"
                else:
                    print(f"⚠ Warning: Boeing API does not support='{tool_choice}'. Using 'required' instead.")
                    formatted_tool_choice = "required"
            elif isinstance(tool_choice, bool):
                formatted_tool_choice = "required" if tool_choice else "none"
            elif isinstance(tool_choice, dict):
                print(f"⚠ Warning: Boeing API does not support dict tool choice. Using 'required' instead.")
                formatted_tool_choice = "required"

        # Create new instance with tools bound
        new_model = self.model_copy(deep=True)
        new_model._bound_tools = formatted_tools
        new_model._tool_choice = formatted_tool_choice

        return new_model

    def with_structured_output(
        self,
        schema: Union[Dict, Type],
        *,
        include_raw: bool = False,
        method: str = "function_calling",
        **kwargs: Any,
    ) -> Union[Runnable, Dict[str, Any]]:
        """
        Enable structured output extraction using tool calling.

        Args:
            schema: Pydantic model, TypedDict, or JSON schema dict
            include_raw: If True, return both raw and parsed output
            method: "function_calling" (only supported method)
            **kwargs: Additional parameters

        Returns:
            Model configured for structured output or Runnable chain
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from langchain_core.output_parsers.openai_tools import (
            JsonOutputKeyToolsParser,
            PydanticToolsParser,
        )
        from langchain_core.runnables import RunnablePassthrough

        if method != "function_calling":
            raise NotImplementedError(
                f"Method {method} not supported. Only 'function_calling' is available."
            )

        if schema is None:
            raise ValueError("schema must be provided for structured output")

        # Convert schema to tool
        tool = convert_to_openai_tool(schema)
        tool_name = tool["function"]["name"]

        # Bind as required tool
        llm = self.bind_tools([tool], tool_choice="required")

        # Select appropriate parser
        if isinstance(schema, type):
            # Pydantic model
            output_parser: Union[PydanticToolsParser, JsonOutputKeyToolsParser] = PydanticToolsParser(
                tools=[schema],
                first_tool_only=True
            )
        else:
            # Dict schema
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name,
                first_tool_only=True
            )

        if include_raw:
            result: Dict[str, Any] = {
                "raw": RunnablePassthrough(),
                "parsed": llm | output_parser,
            }
            return result
        else:
            return llm | output_parser

    # ==================== Token Counting ====================

    def get_num_tokens_from_messages(
        self,
        messages: List[BaseMessage],
        tools: Optional[Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]] = None
    ) -> int:
        """
        Count tokens using Boeing's Token Counter API.

        Args:
            messages: List of messages to count
            tools: Optional tools to include in count

        Returns:
            Total token count
        """
        api_messages = self._convert_messages_to_api_format(messages)
        payload = {
            "messages": api_messages,
            "model": self.model
        }

        try:
            response = requests.post(
                self.token_counter_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            return data.get("tokenCount", 0)

        except requests.exceptions.RequestException as e:
            # Return 0 on failure, but log warning
            print(f"Warning: Token counting failed: {e}")
            return 0

    def get_token_ids(self, text: str) -> List[int]:
        """
        Get token IDs for text.
        Note: Boeing API doesn't expose this, so we return empty list.
        """
        return []

