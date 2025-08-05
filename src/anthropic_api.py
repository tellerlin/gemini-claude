# src/anthropic_api.py
import uuid
import json
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal, Tuple
from pydantic import BaseModel, Field
import logging
from fastapi import HTTPException
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentDict, PartDict, Tool as GeminiTool, FunctionDeclaration

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

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
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

    def _convert_schema_to_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归地将 Anthropic 的 JSON Schema 转换为 Gemini 的格式。
        """
        if "type" not in schema:
            return {}

        # 转换类型名称
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
        }
        gemini_type = type_mapping.get(schema["type"].lower())
        if not gemini_type:
            logger.warning(f"Unsupported schema type ignored: {schema['type']}")
            return {}

        gemini_schema = {"type": gemini_type}

        if "description" in schema:
            gemini_schema["description"] = schema["description"]

        # 递归转换 object 的 properties
        if gemini_type == "OBJECT" and "properties" in schema:
            gemini_schema["properties"] = {
                k: self._convert_schema_to_gemini(v)
                for k, v in schema["properties"].items()
            }
            if "required" in schema:
                gemini_schema["required"] = schema["required"]

        # 递归转换 array 的 items
        if gemini_type == "ARRAY" and "items" in schema:
            gemini_schema["items"] = self._convert_schema_to_gemini(schema["items"])

        return gemini_schema

    def _convert_anthropic_tool_to_gemini(self, tool: Tool) -> Optional[GeminiTool]:
        """
        将单个 Anthropic tool 转换为 Gemini Tool 格式。
        """
        try:
            if not tool.input_schema:
                logger.warning(f"Tool '{tool.name}' is missing input_schema and will be skipped.")
                return None
                
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=self._convert_schema_to_gemini(tool.input_schema)
            )
            return GeminiTool(function_declarations=[function_declaration])
        except Exception as e:
            logger.error(f"Failed to convert tool '{tool.name}': {e}", exc_info=True)
            return None

    def convert_tools(self, tools: Optional[List[Tool]]) -> Optional[List[GeminiTool]]:
        """
        转换工具列表
        """
        if not tools:
            return None
            
        gemini_tools = []
        for tool in tools:
            converted_tool = self._convert_anthropic_tool_to_gemini(tool)
            if converted_tool:
                gemini_tools.append(converted_tool)
        
        return gemini_tools if gemini_tools else None

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
                    elif isinstance(block, ContentBlockToolResult):
                        # 处理工具调用结果
                        result_text = block.content if isinstance(block.content, str) else json.dumps(block.content)
                        parts.append(PartDict(text=f"Tool result: {result_text}"))
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
                # 处理工具调用
                if hasattr(part, 'function_call') and part.function_call.name:
                    tool_call_id = f"toolu_{uuid.uuid4().hex}"
                    tool_input = dict(part.function_call.args) if part.function_call.args else {}
                    content_parts.append(ContentBlockToolUse(
                        id=tool_call_id,
                        name=part.function_call.name,
                        input=tool_input
                    ))
                    stop_reason = "tool_use"
                elif part.text:
                    content_parts.append(ContentBlockText(text=part.text))
        
        # 如果没有内容，确保返回一个空的文本块
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
        
        is_tool_call_in_progress = False
        current_tool_call_id = None
        current_tool_name = None
        accumulated_args = ""
        final_usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason = "end_turn"
        content_block_index = 0

        try:
            async for chunk in gemini_stream:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                part = candidate.content.parts[0]

                # 检查是否是工具调用
                if hasattr(part, 'function_call') and part.function_call.name:
                    if not is_tool_call_in_progress:
                        # 工具调用的第一个数据块
                        is_tool_call_in_progress = True
                        current_tool_name = part.function_call.name
                        current_tool_call_id = f"toolu_{uuid.uuid4().hex}"
                        stop_reason = "tool_use"
                        accumulated_args = ""

                        # 发送 content_block_start 事件
                        yield self._create_sse_event("content_block_start", {
                            "index": content_block_index,
                            "content_block": {
                                "type": "tool_use", 
                                "id": current_tool_call_id, 
                                "name": current_tool_name, 
                                "input": {}
                            }
                        })

                    # 流式传输工具参数
                    if part.function_call.args:
                        args_dict = dict(part.function_call.args)
                        args_str = json.dumps(args_dict, ensure_ascii=False)
                        
                        # 发送参数增量
                        yield self._create_sse_event("content_block_delta", {
                            "index": content_block_index,
                            "delta": {"type": "input_json_delta", "partial_json": args_str}
                        })
                        accumulated_args = args_str

                # 检查是否是常规文本
                elif hasattr(part, 'text') and part.text:
                    if is_tool_call_in_progress:
                        # 结束工具调用块
                        yield self._create_sse_event("content_block_stop", {"index": content_block_index})
                        is_tool_call_in_progress = False
                        content_block_index += 1

                    if not is_tool_call_in_progress:
                        # 开始文本块
                        yield self._create_sse_event("content_block_start", {
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""}
                        })

                    # 发送文本增量
                    yield self._create_sse_event("content_block_delta", {
                        "index": content_block_index,
                        "delta": {"type": "text_delta", "text": part.text}
                    })

                # 更新使用量和停止原因
                if chunk.usage_metadata:
                    final_usage["input_tokens"] = chunk.usage_metadata.prompt_token_count
                    final_usage["output_tokens"] = chunk.usage_metadata.candidates_token_count
                
                if candidate.finish_reason and candidate.finish_reason.name in ["MAX_TOKENS", "LENGTH"]:
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

        # 循环结束后，确保关闭所有打开的块
        if is_tool_call_in_progress or not is_tool_call_in_progress:
            yield self._create_sse_event("content_block_stop", {"index": content_block_index})
        
        # 发送结束事件
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