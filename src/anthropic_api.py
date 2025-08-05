# src/anthropic_api.py
import uuid
import json
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal
from pydantic import BaseModel, Field
import logging
from fastapi import HTTPException
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentDict, PartDict, Tool as GeminiTool

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ANSI 颜色代码，用于美化日志输出
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
    log_line = f"{Colors.BOLD}{method} {path.split('?')[0]}{Colors.RESET} {claude_display} → {gemini_display} {messages_str} {tools_str}"
    print(log_line)

# ========== Anthropic API 数据模型 ==========
class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockToolUse]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    # --- MODIFIED SECTION START ---
    # 允许 system 字段是字符串或字典列表，以兼容 Claude Code 客户端
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    # --- MODIFIED SECTION END ---
    stream: bool = False
    temperature: float = 1.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    usage: Usage

# ========== 转换器类 ==========
class AnthropicToGeminiConverter:
    def convert_model(self, anthropic_model: str) -> str:
        model_map = {
            "sonnet": "gemini-2.0-flash-exp",
            "opus": "gemini-2.0-flash-exp", 
            "haiku": "gemini-2.0-flash-exp"
        }
        for key, value in model_map.items():
            if key in anthropic_model.lower():
                return value
        return "gemini-2.0-flash-exp"

    def convert_messages(self, messages: List[Message]) -> List[ContentDict]:
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            if isinstance(msg.content, str):
                gemini_messages.append({'role': role, 'parts': [PartDict(text=msg.content)]})
            elif isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if isinstance(block, ContentBlockText):
                        parts.append(PartDict(text=block.text))
                if parts:
                    gemini_messages.append({'role': role, 'parts': parts})
        return gemini_messages

class GeminiToAnthropicConverter:
    def convert_response(self, gemini_response: genai.types.GenerateContentResponse, original_request: MessagesRequest) -> MessagesResponse:
        content_parts = []
        stop_reason = "end_turn"
        
        if gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            if candidate.finish_reason.name in ["MAX_TOKENS", "LENGTH"]:
                stop_reason = "max_tokens"
            
            for part in candidate.content.parts:
                if part.text:
                    content_parts.append(ContentBlockText(text=part.text))
        
        # 如果没有文本内容，确保返回一个空的文本块
        if not content_parts:
            content_parts.append(ContentBlockText(text=""))
        
        usage = Usage(
            input_tokens=gemini_response.usage_metadata.prompt_token_count if gemini_response.usage_metadata else 0,
            output_tokens=gemini_response.usage_metadata.candidates_token_count if gemini_response.usage_metadata else 0
        )
        
        return MessagesResponse(
            id=f"msg_gemini_{uuid.uuid4().hex}",
            model=original_request.model,
            content=content_parts,
            stop_reason=stop_reason,
            usage=usage
        )

    async def convert_stream_response(self, gemini_stream, original_request: MessagesRequest) -> AsyncGenerator[str, None]:
        message_id = f"msg_gemini_{uuid.uuid4().hex}"
        
        # 发送 message_start 事件
        yield self._create_sse_event("message_start", {
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })
        
        # 发送 content_block_start 事件
        yield self._create_sse_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })
        
        final_usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason = "end_turn"

        try:
            async for chunk in gemini_stream:
                if chunk.text:
                    yield self._create_sse_event("content_block_delta", {
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk.text}
                    })
                
                if chunk.usage_metadata:
                    final_usage["input_tokens"] = chunk.usage_metadata.prompt_token_count
                    final_usage["output_tokens"] = chunk.usage_metadata.candidates_token_count
                
                if chunk.candidates and chunk.candidates[0].finish_reason and chunk.candidates[0].finish_reason.name in ["MAX_TOKENS", "LENGTH"]:
                    stop_reason = "max_tokens"
                    
        except Exception as e:
            logger.error(f"Error processing Gemini stream: {e}", exc_info=True)
            yield self._create_sse_event("error", {
                "error": {
                    "type": "internal_server_error",
                    "message": str(e)
                }
            })
            return

        # 发送结束事件
        yield self._create_sse_event("content_block_stop", {"index": 0})
        yield self._create_sse_event("message_delta", {
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": final_usage["output_tokens"]}
        })
        yield self._create_sse_event("message_stop", {})

    def _create_sse_event(self, event: str, data: Dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

class APIConfig:
    def __init__(self):
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter()
        logger.info("🚀 Anthropic-Gemini Native API Compatibility Layer Initialized.")