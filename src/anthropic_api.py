"""
Anthropic API 兼容层 - 增强版
提供完整的 Anthropic Messages API 支持，并专门支持 Claude Code 的文件系统工具调用
"""

import asyncio
import time
import uuid
import json
import os
import subprocess
import base64
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)

# ANSI color codes for beautiful logging
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def log_request_beautifully(method: str, path: str, anthropic_model: str, gemini_model: str,
                          num_messages: int, num_tools: int, status_code: int = 200):
    """Log requests in a beautiful format showing model mapping"""
    claude_display = f"{Colors.CYAN}{anthropic_model}{Colors.RESET}"
    gemini_display = f"{Colors.GREEN}{gemini_model}{Colors.RESET}"
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}" if num_tools > 0 else ""
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    
    endpoint = path.split("?")[0]
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {gemini_display} {tools_str} {messages_str}"
    
    print(log_line)
    print(model_line)

# ========== Anthropic API 数据模型 ==========

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
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

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

class ThinkingConfig(BaseModel):
    enabled: bool

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

# ========== Claude Code 文件系统工具模拟器 ==========

class ClaudeCodeToolSimulator:
    """模拟 Claude Code 的文件系统工具调用"""
    
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        os.makedirs(self.working_directory, exist_ok=True)
        logger.info(f"Claude Code simulator initialized in: {self.working_directory}")
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用并返回结果"""
        try:
            logger.info(f"🔧 TOOL CALL: {tool_name} with input: {tool_input}")
            
            # Map friendly names to actual methods
            tool_map = {
                "create_file": self._create_file,
                "read_file": self._read_file,
                "write_file": self._write_file,
                "delete_file": self._delete_file,
                "list_directory": self._list_directory,
                "create_directory": self._create_directory,
                "run_command": self._run_command,
                "search_files": self._search_files,
                "move_file": self._move_file,
                "copy_file": self._copy_file,
            }
            
            if tool_name in tool_map:
                return await tool_map[tool_name](tool_input)
            else:
                logger.warning(f"Unknown tool: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}

    def _resolve_path(self, path: str) -> str:
        """Securely resolve a path within the working directory."""
        full_path = os.path.abspath(os.path.join(self.working_directory, path))
        if not full_path.startswith(self.working_directory):
            raise PermissionError("Path traversal attempt detected.")
        return full_path

    async def _create_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", input_data.get("file_path", ""))
        content = input_data.get("content", "")
        if not path: return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
        logger.info(f"✅ Created file: {full_path}")
        return {"success": True, "message": f"File created: {path}"}

    async def _read_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", input_data.get("file_path", ""))
        if not path: return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path): return {"error": f"File not found: {path}"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f: content = f.read()
            logger.info(f"📖 Read file: {full_path} ({len(content)} chars)")
            return {"content": content, "path": path}
        except UnicodeDecodeError:
            with open(full_path, 'rb') as f: binary_content = f.read()
            return {"content": base64.b64encode(binary_content).decode('utf-8'), "path": path, "encoding": "base64"}

    async def _write_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", input_data.get("file_path", ""))
        content = input_data.get("content", "")
        if not path: return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
        logger.info(f"✏️ Wrote file: {full_path} ({len(content)} chars)")
        return {"success": True, "message": f"File written: {path}"}

    async def _delete_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", input_data.get("file_path", ""))
        if not path: return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path): return {"error": f"File not found: {path}"}
        os.remove(full_path)
        logger.info(f"🗑️ Deleted file: {full_path}")
        return {"success": True, "message": f"File deleted: {path}"}

    async def _list_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", ".")
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path): return {"error": f"Directory not found: {path}"}
        
        items = []
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            is_dir = os.path.isdir(item_path)
            items.append({"name": item, "type": "directory" if is_dir else "file"})
        logger.info(f"📁 Listed directory: {full_path} ({len(items)} items)")
        return {"items": items, "path": path}

    async def _create_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", input_data.get("directory_path", ""))
        if not path: return {"error": "No directory path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"📁 Created directory: {full_path}")
        return {"success": True, "message": f"Directory created: {path}"}

    async def _run_command(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        command = input_data.get("command", "")
        if not command: return {"error": "No command provided"}
        
        try:
            safe_commands = ["ls", "pwd", "echo", "cat", "touch", "mkdir", "rm", "cp", "mv", "grep", "find"]
            cmd_parts = command.split()
            if not cmd_parts or cmd_parts[0] not in safe_commands:
                return {"error": f"Command not allowed: {cmd_parts[0]}"}
            
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)
            logger.info(f"🖥️ Executed command: '{command}' (exit code: {result.returncode})")
            return {
                "stdout": stdout.decode(), "stderr": stderr.decode(),
                "return_code": result.returncode, "command": command
            }
        except asyncio.TimeoutError:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Command execution failed: {str(e)}"}
            
# ========== 转换器类 ==========

class AnthropicToGeminiConverter:
    """将 Anthropic API 请求转换为 Gemini 格式"""
    
    def __init__(self):
        self.model_mapping = {
            "claude-3-5-sonnet": "gemini-1.5-pro-latest",
            "claude-3-5-haiku": "gemini-1.5-flash-latest",
            "claude-3-opus": "gemini-1.5-pro-latest",
            "claude-3-sonnet": "gemini-1.5-pro-latest",
            "claude-3-haiku": "gemini-1.5-flash-latest",
        }
    
    def convert_model(self, anthropic_model: str) -> str:
        sorted_keys = sorted(self.model_mapping.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if anthropic_model.startswith(key):
                return self.model_mapping[key]
        return "gemini-1.5-pro-latest"

    def convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        converted = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            content = msg.content
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if block.type == "text": parts.append({"text": block.text})
                    elif block.type == "tool_result":
                         parts.append({"function_response": {
                             "name": "tool_code", # A generic name for the response
                             "response": {"content": block.content, "tool_use_id": block.tool_use_id}
                         }})
            converted.append({"role": role, "parts": parts})
        return converted
    
    def convert_request(self, request: MessagesRequest) -> Dict[str, Any]:
        gemini_model = self.convert_model(request.model)
        converted = {
            "messages": self.convert_messages(request.messages),
            "model": f"gemini/{gemini_model}",
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.max_tokens: converted["max_tokens"] = request.max_tokens
        if request.system:
            if isinstance(request.system, str):
                converted["system_instruction"] = {"parts": [{"text": request.system}]}
            elif isinstance(request.system, list):
                 converted["system_instruction"] = {"parts": [{"text": c.text} for c in request.system]}

        return converted

class GeminiToAnthropicConverter:
    """将 Gemini 响应转换为 Anthropic 格式"""
    
    def __init__(self, claude_code_simulator: ClaudeCodeToolSimulator):
        self.claude_code_simulator = claude_code_simulator

    def convert_usage(self, gemini_usage: Optional[Dict[str, int]]) -> Usage:
        if not gemini_usage: return Usage(input_tokens=0, output_tokens=0)
        return Usage(input_tokens=gemini_usage.get("prompt_token_count", 0),
                     output_tokens=gemini_usage.get("candidates_token_count", 0))

    async def _parse_and_execute_tools(self, text: str) -> Tuple[str, List[ContentBlockToolUse]]:
        """从文本中解析出类似 '● command' 的工具调用并执行它们。"""
        tool_uses = []
        # Simple regex to find tool commands starting with a bullet
        import re
        commands = re.findall(r"●\s*(.+)", text)
        if not commands: return text, []

        for command in commands:
            parts = command.split(" ", 1)
            tool_name = parts[0]
            # A simple mapping from shell commands to tool names
            tool_map = {"touch": "create_file", "mkdir": "create_directory", "ls": "list_directory"}
            mapped_tool = tool_map.get(tool_name, "run_command")
            
            tool_input = {}
            if mapped_tool == "create_file": tool_input = {"path": parts[1] if len(parts)>1 else "", "content": ""}
            elif mapped_tool == "create_directory": tool_input = {"path": parts[1] if len(parts)>1 else ""}
            elif mapped_tool == "list_directory": tool_input = {"path": parts[1] if len(parts)>1 else "."}
            else: tool_input = {"command": command}
            
            tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
            tool_uses.append(ContentBlockToolUse(id=tool_use_id, name=mapped_tool, input=tool_input))
            
            # Execute the tool call
            await self.claude_code_simulator.execute_tool(mapped_tool, tool_input)

        return text, tool_uses
    
    async def convert_response(self, gemini_response: Dict[str, Any], original_request: MessagesRequest) -> MessagesResponse:
        """转换完整响应，增强错误处理和工具调用支持"""
        content_text = ""
        stop_reason = "end_turn"
        
        try:
            choice = gemini_response['choices'][0]
            if 'text' in choice: content_text = choice['text']
            elif 'message' in choice and 'content' in choice['message']: content_text = choice['message']['content']
            
            if choice.get("finish_reason") == "max_tokens": stop_reason = "max_tokens"

            response_content, tool_uses = await self._parse_and_execute_tools(content_text)
            
            final_content = []
            if response_content: final_content.append(ContentBlockText(text=response_content))
            if tool_uses:
                final_content.extend(tool_uses)
                stop_reason = "tool_use"

            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                model=original_request.model,
                content=final_content,
                stop_reason=stop_reason,
                usage=self.convert_usage(gemini_response.get('usage', {}))
            )
        except (KeyError, IndexError, Exception) as e:
            logger.error(f"Error converting Gemini response: {e}\nResponse: {gemini_response}")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                model=original_request.model,
                content=[ContentBlockText(text=f"Error processing response: {e}")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0)
            )

# ========== 流式响应生成器 ==========

class StreamingResponseGenerator:
    """生成 Anthropic 格式的流式响应，支持工具调用"""
    
    def __init__(self, original_request: MessagesRequest, claude_code_simulator: ClaudeCodeToolSimulator):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.input_tokens = 0
        self.claude_code_simulator = claude_code_simulator

    async def generate_sse_events(self, gemini_stream: AsyncGenerator[Dict, None]) -> AsyncGenerator[str, None]:
        """生成 SSE 事件流"""
        yield self._create_event("message_start", {
            "message": {"id": self.message_id, "type": "message", "role": "assistant",
                        "model": self.original_request.model, "content": [], "stop_reason": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}}
        })
        yield self._create_event("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
        
        full_response_text = ""
        output_tokens = 0
        try:
            async for chunk in gemini_stream:
                delta = ""
                if 'choices' in chunk and chunk['choices']:
                     if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                         delta = chunk['choices'][0]['delta']['content']
                     elif 'text' in chunk['choices'][0]: # Handle non-delta chunks in stream
                         delta = chunk['choices'][0]['text']

                if delta:
                    full_response_text += delta
                    yield self._create_event("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": delta}})
                
                if 'usage' in chunk and 'completion_tokens' in chunk['usage']:
                    output_tokens = chunk['usage']['completion_tokens']

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield self._create_event("error", {"error": {"type": "stream_error", "message": str(e)}})

        yield self._create_event("content_block_stop", {"index": 0})
        
        # Post-stream tool processing
        _, tool_uses = await GeminiToAnthropicConverter(self.claude_code_simulator)._parse_and_execute_tools(full_response_text)

        if tool_uses:
            for i, tool_use in enumerate(tool_uses):
                yield self._create_event("content_block_start", {"index": i + 1, "content_block": tool_use.model_dump()})
                yield self._create_event("content_block_stop", {"index": i + 1})

        stop_reason = "tool_use" if tool_uses else "end_turn"
        yield self._create_event("message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": output_tokens}})
        yield self._create_event("message_stop", {})
        yield "data: [DONE]\n\n"

    def _create_event(self, event_type: str, data: Dict) -> str:
        """Helper to format SSE events."""
        return f"event: {event_type}\ndata: {json.dumps({'type': event_type, **data})}\n\n"

# ========== 工具支持 ==========

class ToolConverter:
    """工具调用转换器"""
    def convert_tools_to_gemini(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        gemini_tools = []
        for tool in tools:
            gemini_tools.append({
                "function_declarations": [{
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                }]
            })
        return gemini_tools

    def convert_tool_choice_to_gemini(self, tool_choice: Dict[str, Any]) -> str:
        choice_type = tool_choice.get("type", "auto").lower()
        if choice_type == "any": return "ANY"
        if choice_type == "tool": return tool_choice.get("name", "AUTO")
        return "AUTO"

# ========== 配置和初始化 ==========

class AnthropicAPIConfig:
    """API 配置类"""
    
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        # Create a single, shared simulator instance
        self.claude_code_simulator = ClaudeCodeToolSimulator(self.working_directory)
        
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        # Pass the shared instance to the converter
        self.gemini_to_anthropic = GeminiToAnthropicConverter(self.claude_code_simulator)
        self.tool_converter = ToolConverter()
        
        logger.info(f"🚀 Anthropic API Compatibility Layer initialized")
        logger.info(f"📁 Working directory for Claude Code: {self.working_directory}")
        logger.info(f"🔧 Claude Code tools: Enabled")
