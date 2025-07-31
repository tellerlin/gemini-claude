# 在 anthropic_api.py 中优化流式响应生成器
class StreamingResponseGenerator:
    """生成 Anthropic 格式的流式响应"""
    
    def __init__(self, original_request: MessagesRequest):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.input_tokens = 0  # 添加输入token计数
    
    async def generate_sse_events(self, gemini_stream) -> AsyncGenerator[str, None]:
        """生成 SSE 事件流"""
        try:
            # 计算输入tokens（粗略估算）
            self.input_tokens = self._estimate_input_tokens()
            
            # 发送 message_start 事件
            yield self._create_message_start()
            
            # 发送 content_block_start 事件
            yield self._create_content_block_start()
            
            # 发送 ping 事件
            yield self._create_ping()
            
            accumulated_text = ""
            output_tokens = 0
            chunk_count = 0
            last_ping_time = time.time()
            
            # 处理流式内容
            async for chunk in gemini_stream:
                chunk_count += 1
                current_time = time.time()
                
                # 每30秒发送一次ping以保持连接
                if current_time - last_ping_time > 30:
                    yield self._create_ping()
                    last_ping_time = current_time
                
                try:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                            content = choice.delta.content
                            if content:
                                accumulated_text += content
                                output_tokens += self._estimate_token_count(content)
                                yield self._create_content_block_delta(content)
                    
                    # 检查是否有使用统计
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
                        if hasattr(usage, 'completion_tokens'):
                            output_tokens = usage.completion_tokens
                        if hasattr(usage, 'prompt_tokens'):
                            self.input_tokens = usage.prompt_tokens
                            
                except Exception as chunk_error:
                    logger.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                    continue
            
            # 发送结束事件
            yield self._create_content_block_stop()
            yield self._create_message_delta(output_tokens)
            yield self._create_message_stop()
            yield "data: [DONE]\n\n"
            
            logger.info(f"Streaming completed: {chunk_count} chunks, {output_tokens} output tokens")
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield self._create_error_event(str(e))
    
    def _estimate_input_tokens(self) -> int:
        """估算输入token数量"""
        total_chars = 0
        for message in self.original_request.messages:
            if isinstance(message.content, str):
                total_chars += len(message.content)
            elif isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'text'):
                        total_chars += len(block.text)
        
        # 粗略估算：每4个字符约等于1个token
        return max(1, total_chars // 4)
    
    def _estimate_token_count(self, text: str) -> int:
        """估算文本的token数量"""
        return max(1, len(text.split()) // 0.75)  # 粗略估算
    
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
                    'input_tokens': self.input_tokens,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        return f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
    
    def _create_content_block_stop(self) -> str:
        """创建 content_block_stop 事件"""
        return f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    
    def _create_error_event(self, error_message: str) -> str:
        """创建错误事件"""
        error_data = {
            'type': 'error',
            'error': {
                'type': 'internal_server_error',
                'message': error_message
            }
        }
        return f"event: error\ndata: {json.dumps(error_data)}\n\n"

# 优化 GeminiToAnthropicConverter 类
class GeminiToAnthropicConverter:
    """将 Gemini 响应转换为 Anthropic 格式"""
    
    def convert_usage(self, gemini_usage) -> Usage:
        """转换使用统计信息"""
        input_tokens = 0
        output_tokens = 0
        
        if gemini_usage:
            # 处理不同的usage格式
            if hasattr(gemini_usage, 'prompt_tokens'):
                input_tokens = gemini_usage.prompt_tokens
            elif hasattr(gemini_usage, 'input_tokens'):
                input_tokens = gemini_usage.input_tokens
            elif isinstance(gemini_usage, dict):
                input_tokens = gemini_usage.get('prompt_tokens', gemini_usage.get('input_tokens', 0))
                
            if hasattr(gemini_usage, 'completion_tokens'):
                output_tokens = gemini_usage.completion_tokens
            elif hasattr(gemini_usage, 'output_tokens'):
                output_tokens = gemini_usage.output_tokens
            elif isinstance(gemini_usage, dict):
                output_tokens = gemini_usage.get('completion_tokens', gemini_usage.get('output_tokens', 0))
            
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def convert_content(self, gemini_content: str) -> List[ContentBlockText]:
        """转换内容，处理空内容的情况"""
        if not gemini_content:
            gemini_content = ""  # 确保不返回None
        return [ContentBlockText(type="text", text=str(gemini_content))]
    
    def convert_response(self, gemini_response, original_request: MessagesRequest) -> MessagesResponse:
        """转换完整响应，增强错误处理"""
        try:
            content = ""
            usage = None
            stop_reason = "end_turn"
            
            # 处理不同格式的响应
            if hasattr(gemini_response, 'choices') and gemini_response.choices:
                choice = gemini_response.choices[0]
                
                # 获取内容
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content or ""
                elif hasattr(choice, 'text'):
                    content = choice.text or ""
                
                # 获取停止原因
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
                    if finish_reason == 'length':
                        stop_reason = "max_tokens"
                    elif finish_reason in ['stop', 'end_turn']:
                        stop_reason = "end_turn"
                    elif finish_reason == 'function_call':
                        stop_reason = "tool_use"
                
                # 获取使用统计
                usage = getattr(gemini_response, 'usage', None)
                
            elif isinstance(gemini_response, dict):
                # 处理字典格式的响应
                choices = gemini_response.get('choices', [])
                if choices:
                    choice = choices[0]
                    message = choice.get('message', {})
                    content = message.get('content', '')
                    
                    finish_reason = choice.get('finish_reason')
                    if finish_reason == 'length':
                        stop_reason = "max_tokens"
                    elif finish_reason in ['stop', 'end_turn']:
                        stop_reason = "end_turn"
                
                usage = gemini_response.get('usage')
            
            # 生成响应ID
            response_id = f"msg_{uuid.uuid4().hex[:24]}"
            
            return MessagesResponse(
                id=response_id,
                model=original_request.model,
                role="assistant",
                content=self.convert_content(content),
                stop_reason=stop_reason,
                usage=self.convert_usage(usage)
            )
            
        except Exception as e:
            logger.error(f"Error converting response: {e}")
            logger.error(f"Response type: {type(gemini_response)}")
            logger.error(f"Response content: {str(gemini_response)[:500]}...")
            
            # 返回错误响应
            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                model=original_request.model,
                role="assistant",
                content=[ContentBlockText(type="text", text=f"Error processing response: {str(e)}")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0)
            )

# 优化 ToolConverter 类
class ToolConverter:
    """工具调用转换器，增强功能"""
    
    def convert_tools_to_gemini(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """将 Anthropic 工具转换为 Gemini 格式"""
        gemini_tools = []
        
        for tool in tools:
            try:
                # 确保schema格式正确
                schema = tool.input_schema
                if not isinstance(schema, dict):
                    logger.warning(f"Invalid schema for tool {tool.name}, skipping")
                    continue
                
                # 处理schema中的required字段
                if 'required' not in schema and 'properties' in schema:
                    schema['required'] = []
                
                gemini_tool = {
                    "function_declarations": [{
                        "name": tool.name,
                        "description": tool.description or f"Execute {tool.name} function",
                        "parameters": schema
                    }]
                }
                gemini_tools.append(gemini_tool)
                logger.debug(f"Converted tool: {tool.name}")
                
            except Exception as e:
                logger.error(f"Error converting tool {tool.name}: {e}")
                continue
        
        return gemini_tools
    
    def convert_tool_choice_to_gemini(self, tool_choice: Dict[str, Any]) -> str:
        """转换工具选择参数，增强错误处理"""
        if not isinstance(tool_choice, dict):
            logger.warning(f"Invalid tool_choice format: {type(tool_choice)}")
            return "AUTO"
        
        choice_type = tool_choice.get("type", "auto").lower()
        
        if choice_type == "auto":
            return "AUTO"
        elif choice_type == "any":
            return "ANY"
        elif choice_type == "tool" and "name" in tool_choice:
            return tool_choice["name"]
        elif choice_type == "none":
            return "NONE"
        else:
            logger.warning(f"Unknown tool_choice type: {choice_type}")
            return "AUTO"
    
    def convert_gemini_tool_calls_to_anthropic(self, gemini_calls) -> List[ContentBlockToolUse]:
        """将Gemini的工具调用转换为Anthropic格式"""
        anthropic_calls = []
        
        for call in gemini_calls:
            try:
                tool_use = ContentBlockToolUse(
                    type="tool_use",
                    id=f"toolu_{uuid.uuid4().hex[:24]}",
                    name=call.get("name", "unknown_tool"),
                    input=call.get("parameters", {})
                )
                anthropic_calls.append(tool_use)
            except Exception as e:
                logger.error(f"Error converting tool call: {e}")
                continue
        
        return anthropic_calls