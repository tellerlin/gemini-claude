"""
Anthropic API 兼容层
提供完整的 Anthropic Messages API 支持，包括数据模型、转换器和流式响应
"""

import asyncio
import time
import uuid
import json
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

# ========== 转换器类 ==========

class AnthropicToGeminiConverter:
    """将 Anthropic API 请求转换为 Gemini 格式"""
    
    def __init__(self):
        self.model_mapping = {
            "claude-3-5-sonnet": "gemini-2.5-pro",
            "claude-3-5-haiku": "gemini-2.5-flash",
            "claude-3-opus": "gemini-2.5-pro",
            "claude-3-sonnet": "gemini-2.5-pro",
            "claude-3-haiku": "gemini-2.5-flash",
        }
    
    def convert_model(self, anthropic_model: str) -> str:
        """将 Anthropic 模型名转换为 Gemini 模型名"""
        gemini_model = self.model_mapping.get(anthropic_model, "gemini-2.5-pro")
        logger.debug(f"📋 MODEL MAPPING: {anthropic_model} → {gemini_model}")
        return gemini_model
    
    def convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """转换消息格式"""
        converted_messages = []
        
        for msg in messages:
            if isinstance(msg.content, str):
                converted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                # 处理复杂内容块
                content_text = ""
                for block in msg.content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            content_text += block.text + "\n"
                        elif block.type == "image":
                            content_text += "[Image content]\n"
                        elif block.type == "tool_use":
                            content_text += f"[Tool use: {block.name}]\n"
                        elif block.type == "tool_result":
                            content_text += f"[Tool result: {block.tool_use_id}]\n"
                
                converted_messages.append({
                    "role": msg.role,
                    "content": content_text.strip()
                })
        
        return converted_messages
    
    def convert_request(self, request: MessagesRequest) -> Dict[str, Any]:
        """转换完整的请求"""
        gemini_model = self.convert_model(request.model)
        
        converted = {
            "messages": self.convert_messages(request.messages),
            "model": gemini_model,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        
        if request.max_tokens:
            converted["max_tokens"] = request.max_tokens
        
        logger.info(f"🔄 REQUEST CONVERSION: {request.model} → {gemini_model}")
        return converted

class GeminiToAnthropicConverter:
    """将 Gemini 响应转换为 Anthropic 格式"""
    
    def convert_usage(self, gemini_usage) -> Usage:
        """转换使用统计信息"""
        if hasattr(gemini_usage, 'prompt_tokens'):
            input_tokens = gemini_usage.prompt_tokens
        elif hasattr(gemini_usage, 'input_tokens'):
            input_tokens = gemini_usage.input_tokens
        else:
            input_tokens = 0
            
        if hasattr(gemini_usage, 'completion_tokens'):
            output_tokens = gemini_usage.completion_tokens
        else:
            output_tokens = 0
            
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def convert_content(self, gemini_content: str) -> List[ContentBlockText]:
        """转换内容"""
        return [ContentBlockText(type="text", text=gemini_content)]
    
    def convert_response(self, gemini_response, original_request: MessagesRequest) -> MessagesResponse:
        """转换完整响应"""
        try:
            # 提取响应内容
            if hasattr(gemini_response, 'choices'):
                choice = gemini_response.choices[0]
                content = choice.message.content if hasattr(choice.message, 'content') else ""
                usage = gemini_response.usage if hasattr(gemini_response, 'usage') else None
            else:
                # 处理字典格式的响应
                content = gemini_response.get('choices', [{}])[0].get('message', {}).get('content', '')
                usage = gemini_response.get('usage', {})
            
            # 生成响应ID
            response_id = f"msg_{uuid.uuid4().hex[:24]}"
            
            return MessagesResponse(
                id=response_id,
                model=original_request.model,
                role="assistant",
                content=self.convert_content(content),
                stop_reason="end_turn",
                usage=self.convert_usage(usage) if usage else Usage(input_tokens=0, output_tokens=0)
            )
            
        except Exception as e:
            logger.error(f"Error converting response: {e}")
            # 返回错误响应
            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                model=original_request.model,
                role="assistant",
                content=[ContentBlockText(type="text", text=f"Error processing response: {str(e)}")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0)
            )

# ========== 流式响应生成器 ==========

class StreamingResponseGenerator:
    """生成 Anthropic 格式的流式响应"""
    
    def __init__(self, original_request: MessagesRequest):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    async def generate_sse_events(self, gemini_stream) -> AsyncGenerator[str, None]:
        """生成 SSE 事件流"""
        try:
            # 发送 message_start 事件
            yield self._create_message_start()
            
            # 发送 content_block_start 事件
            yield self._create_content_block_start()
            
            # 发送 ping 事件
            yield self._create_ping()
            
            accumulated_text = ""
            output_tokens = 0
            
            # 处理流式内容
            async for chunk in gemini_stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        content = choice.delta.content
                        if content:
                            accumulated_text += content
                            output_tokens += len(content.split())
                            yield self._create_content_block_delta(content)
                
                # 检查是否有使用统计
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    if hasattr(usage, 'completion_tokens'):
                        output_tokens = usage.completion_tokens
            
            # 发送结束事件
            yield self._create_message_delta(output_tokens)
            yield self._create_message_stop()
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield self._create_error_event(str(e))
    
    def _create_message_start(self) -> str:
        """创建 message_start 事件"""
        message_data = {
            'type': 'message_start',
            'message': {
                'id': self.message_id,
                'type': 'message',
                'role': 'assistant',
                'model': self.original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        return f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
    
    def _create_content_block_start(self) -> str:
        """创建 content_block_start 事件"""
        return f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    def _create_ping(self) -> str:
        """创建 ping 事件"""
        return f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    
    def _create_content_block_delta(self, text: str) -> str:
        """创建 content_block_delta 事件"""
        return f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
    
    def _create_message_delta(self, output_tokens: int) -> str:
        """创建 message_delta 事件"""
        return f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    
    def _create_message_stop(self) -> str:
        """创建 message_stop 事件"""
        return f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    
    def _create_error_event(self, error_message: str) -> str:
        """创建错误事件"""
        return f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n" + \
               f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n" + \
               "data: [DONE]\n\n"

# ========== 工具支持 ==========

class ToolConverter:
    """工具调用转换器"""
    
    def convert_tools_to_gemini(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """将 Anthropic 工具转换为 Gemini 格式"""
        gemini_tools = []
        
        for tool in tools:
            gemini_tool = {
                "function_declarations": [{
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                }]
            }
            gemini_tools.append(gemini_tool)
        
        return gemini_tools
    
    def convert_tool_choice_to_gemini(self, tool_choice: Dict[str, Any]) -> str:
        """转换工具选择参数"""
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "AUTO"
        elif choice_type == "any":
            return "ANY"
        elif choice_type == "tool" and "name" in tool_choice:
            return tool_choice["name"]
        else:
            return "AUTO"