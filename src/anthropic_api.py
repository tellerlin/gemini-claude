# anthropic_api.py
import asyncio
import time
import uuid
import json
import os
import subprocess
import base64
import shutil
import re
import mimetypes
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ANSI color codes for beautiful logging
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def log_request_beautifully(method: str, path: str, anthropic_model: str, gemini_model: str,
                          num_messages: int, num_tools: int):
    claude_display = f"{Colors.CYAN}{anthropic_model}{Colors.RESET}"
    gemini_display = f"{Colors.GREEN}{gemini_model}{Colors.RESET}"
    tools_str = f"| {Colors.MAGENTA}{num_tools} tools{Colors.RESET}" if num_tools > 0 else ""
    messages_str = f"| {Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    endpoint = path.split("?")[0]
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {claude_display} → {gemini_display} {messages_str} {tools_str}"
    print(log_line)


# ========== Anthropic API Data Models ==========
class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]] # Simplified content

class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]] # Result is not returned here
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None

class TokenCountResponse(BaseModel):
    input_tokens: int

# ========== Claude Code Tool Simulator (No changes needed here) ==========
class ClaudeCodeToolSimulator:
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        os.makedirs(self.working_directory, exist_ok=True)
        logger.info(f"Claude Code simulator initialized in: {self.working_directory}")

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"🔧 TOOL CALL (SIMULATED): {tool_name} with input: {tool_input}")
        # The actual execution logic is now on the client side.
        # This class can be kept for testing or future use.
        # For now, we'll assume the client handles execution.
        if tool_name == 'create_file':
            filename = tool_input.get('filename') or tool_input.get('path')
            if not filename:
                 return {"error": "Filename not provided"}
            try:
                full_path = os.path.join(self.working_directory, os.path.basename(filename))
                with open(full_path, 'w') as f:
                    pass
                return {"success": True, "message": f"File '{filename}' created successfully in {self.working_directory}."}
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Tool '{tool_name}' execution is simulated and not implemented in the adapter."}

# ========== Conversion Classes (HEAVILY MODIFIED) ==========
class ToolConverter:
    def convert_tools_to_openai(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                }
            })
        return openai_tools

    def convert_tool_choice_to_openai(self, tool_choice: Dict[str, Any]) -> str:
        choice_type = tool_choice.get("type", "auto").lower()
        if choice_type == "any" or choice_type == "tool":
            return "required" # In new OpenAI spec, "required" forces a tool call
        return "auto"

class AnthropicToGeminiConverter:
    def __init__(self):
        self.tool_converter = ToolConverter()
    
    def convert_model(self, anthropic_model: str) -> str:
        anthropic_model_lower = anthropic_model.lower()
        if "sonnet" in anthropic_model_lower or "opus" in anthropic_model_lower:
            return "gemini-1.5-pro-latest"
        elif "haiku" in anthropic_model_lower:
            return "gemini-1.5-flash-latest"
        else:
            return "gemini-1.5-pro-latest"

    def convert_request(self, request: MessagesRequest) -> Dict[str, Any]:
        gemini_model = self.convert_model(request.model)
        
        # Convert messages to OpenAI format, which LiteLLM understands
        openai_messages = []
        for msg in request.messages:
            if msg.role == "user":
                # Check if it's a tool result message
                if isinstance(msg.content, list) and any(block.type == 'tool_result' for block in msg.content):
                    for block in msg.content:
                        if block.type == 'tool_result':
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": json.dumps(block.content) if not isinstance(block.content, str) else block.content,
                            })
                else: # Regular user message
                    text_content = ""
                    if isinstance(msg.content, str):
                        text_content = msg.content
                    elif isinstance(msg.content, list):
                         text_content = "\n".join([block.text for block in msg.content if block.type == 'text'])
                    openai_messages.append({"role": "user", "content": text_content})
            
            elif msg.role == "assistant":
                text_content = ""
                tool_calls = []
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if block.type == "text":
                            text_content += block.text
                        elif block.type == "tool_use":
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input)
                                }
                            })
                elif isinstance(msg.content, str):
                    text_content = msg.content

                assistant_msg = {"role": "assistant"}
                if text_content:
                    assistant_msg["content"] = text_content
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                # Add message only if it has content or tool calls
                if "content" in assistant_msg or "tool_calls" in assistant_msg:
                    openai_messages.append(assistant_msg)

        converted = {
            "model": gemini_model,
            "messages": openai_messages,
            "stream": request.stream,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        # Handle system prompt
        system_text = ""
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            elif isinstance(request.system, list):
                system_text = "\n".join([c.text for c in request.system])
        if system_text:
            # Prepend system message as the first message in the list for LiteLLM
            converted["messages"].insert(0, {"role": "system", "content": system_text})
        
        if request.tools:
            converted["tools"] = self.tool_converter.convert_tools_to_openai(request.tools)
            if request.tool_choice:
                converted["tool_choice"] = self.tool_converter.convert_tool_choice_to_openai(request.tool_choice)

        return converted

class GeminiToAnthropicConverter:
    def __init__(self):
        # The simulator is no longer needed here as execution is on the client side
        pass

    def convert_response(self, gemini_response: Dict[str, Any], original_request: MessagesRequest) -> MessagesResponse:
        """
        Converts a non-streaming LiteLLM/OpenAI response to an Anthropic MessagesResponse.
        FIXED: Does NOT execute tools. It returns a tool_use request to the client.
        """
        try:
            choice = gemini_response['choices'][0]
            message = choice.get('message', {})
            finish_reason = choice.get('finish_reason', 'stop')

            anthropic_content = []
            stop_reason: Literal["end_turn", "max_tokens", "tool_use"] = "end_turn"
            
            # 1. Add text content if it exists
            if message.get('content'):
                anthropic_content.append(ContentBlockText(text=message['content']))

            # 2. Add tool calls if they exist
            if message.get('tool_calls'):
                stop_reason = "tool_use"
                for tool_call in message['tool_calls']:
                    function = tool_call.get('function', {})
                    try:
                        arguments = json.loads(function.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        arguments = {} # Default to empty dict on failure
                    
                    anthropic_content.append(ContentBlockToolUse(
                        id=tool_call['id'],
                        name=function.get('name', ''),
                        input=arguments
                    ))
            
            # If after all that, content is empty, add an empty text block
            if not anthropic_content:
                anthropic_content.append(ContentBlockText(text=""))
            
            # Determine final stop reason
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "max_tokens":
                stop_reason = "max_tokens"
            elif stop_reason != "tool_use": # Don't override tool_use
                stop_reason = "end_turn"

            usage_data = gemini_response.get('usage', {})
            usage = Usage(
                input_tokens=usage_data.get('prompt_tokens', 0),
                output_tokens=usage_data.get('completion_tokens', 0)
            )

            return MessagesResponse(
                id=f"msg_{gemini_response.get('id', uuid.uuid4().hex)}",
                model=original_request.model,
                role="assistant",
                content=anthropic_content,
                stop_reason=stop_reason,
                usage=usage
            )
        except (KeyError, IndexError) as e:
            logger.error(f"Error converting Gemini response: {e}\nResponse: {gemini_response}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to convert model response: {e}")

    async def convert_stream_response(self, gemini_stream: AsyncGenerator[Dict, None], original_request: MessagesRequest) -> AsyncGenerator[str, None]:
        """
        NEW: Handles streaming responses, converting LiteLLM/OpenAI stream chunks
        to Anthropic Server-Sent Events (SSE).
        """
        message_id = f"msg_{uuid.uuid4().hex}"
        
        # 1. Send message_start event
        yield self._create_sse_event("message_start", {"message": {
            "id": message_id, "type": "message", "role": "assistant",
            "model": original_request.model, "content": [], "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0} # Placeholder
        }})

        content_blocks = []
        final_stop_reason = "end_turn"
        input_tokens = 0
        output_tokens = 0
        
        try:
            async for chunk in gemini_stream:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                
                # Capture usage info from the first chunk
                if chunk.usage and not input_tokens:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    yield self._create_sse_event("message_delta", {"usage": {"input_tokens": input_tokens}})

                # Handle text delta
                if delta.content:
                    if not content_blocks or not isinstance(content_blocks[-1], ContentBlockText):
                        # Start a new text block
                        new_block = ContentBlockText(text="")
                        content_blocks.append(new_block)
                        yield self._create_sse_event("content_block_start", {"index": len(content_blocks) - 1, "content_block": {"type": "text", "text": ""}})
                    
                    # Send delta
                    yield self._create_sse_event("content_block_delta", {"index": len(content_blocks) - 1, "delta": {"type": "text_delta", "text": delta.content}})
                    content_blocks[-1].text += delta.content

                # Handle tool call delta
                if delta.tool_calls:
                    final_stop_reason = "tool_use"
                    for tool_call_chunk in delta.tool_calls:
                        index = tool_call_chunk.index
                        # Ensure content_blocks has space
                        while len(content_blocks) <= index:
                            content_blocks.append(None)
                        
                        if content_blocks[index] is None:
                            # This is the start of a new tool call
                            tool_id = tool_call_chunk.id
                            tool_name = tool_call_chunk.function.name
                            new_block = ContentBlockToolUse(id=tool_id, name=tool_name, input={})
                            content_blocks[index] = new_block
                            yield self._create_sse_event("content_block_start", {"index": index, "content_block": new_block.model_dump()})
                            # Append empty arguments for delta streaming
                            yield self._create_sse_event("content_block_delta", {"index": index, "delta": {"type": "input_json_delta", "partial_json": ""}})
                        
                        # Stream argument deltas
                        if tool_call_chunk.function.arguments:
                            yield self._create_sse_event("content_block_delta", {"index": index, "delta": {"type": "input_json_delta", "partial_json": tool_call_chunk.function.arguments}})

                # Handle finish reason
                if chunk.choices[0].finish_reason:
                    # Stop any open blocks
                    for i, block in enumerate(content_blocks):
                        if block: # Could be None if there was an error
                            yield self._create_sse_event("content_block_stop", {"index": i})
                    
                    final_finish_reason = chunk.choices[0].finish_reason
                    if final_finish_reason == "tool_calls":
                        final_stop_reason = "tool_use"
                    elif final_finish_reason == "length":
                        final_stop_reason = "max_tokens"
                    
                    if chunk.usage:
                        output_tokens = chunk.usage.completion_tokens or 0
                    
                    yield self._create_sse_event("message_delta", {"delta": {"stop_reason": final_stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}})
                    break # End of stream
        finally:
            # 2. Send message_stop event
            yield self._create_sse_event("message_stop", {})

    def _create_sse_event(self, event_type: str, data: Dict) -> str:
        json_data = json.dumps(data)
        return f"event: {event_type}\ndata: {json_data}\n\n"

class AnthropicAPIConfig:
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        self.claude_code_simulator = ClaudeCodeToolSimulator(self.working_directory)
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter()
        self.tool_converter = ToolConverter()
        logger.info(f"🚀 Anthropic API Compatibility Layer initialized")
        logger.info(f"📁 Working directory for Claude Code: {self.working_directory}")