import asyncio
import time
import uuid
import json
import os
import subprocess
import base64
import shutil
import re
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal, Tuple
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
    stop_sequence: Optional[str] = None
    usage: Usage

# ========== Claude Code 文件系统工具模拟器 ==========
class ClaudeCodeToolSimulator:
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        os.makedirs(self.working_directory, exist_ok=True)
        logger.info(f"Claude Code simulator initialized in: {self.working_directory}")
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"🔧 TOOL CALL: {tool_name} with input: {tool_input}")
            tool_map = {
                "create_file": self._create_file,
                "read_file": self._read_file,
                "write_file": self._write_file,
                "edit_file": self._edit_file,  # 新增编辑文件功能
                "delete_file": self._delete_file,
                "list_directory": self._list_directory,
                "create_directory": self._create_directory,
                "run_command": self._run_command,
                "search_files": self._search_files,
                "move_file": self._move_file,
                "copy_file": self._copy_file,
                "get_file_info": self._get_file_info,  # 新增获取文件信息
                "find_in_files": self._find_in_files,  # 新增文件内容搜索
                "get_working_directory": self._get_working_directory,  # 新增获取工作目录
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
        if os.path.isabs(path):
            full_path = os.path.abspath(path)
        else:
            full_path = os.path.abspath(os.path.join(self.working_directory, path))
        
        # 更宽松的路径检查，允许访问系统目录但防止恶意路径遍历
        if ".." in os.path.normpath(path) and not full_path.startswith(self.working_directory):
            if not any(full_path.startswith(allowed) for allowed in ["/usr", "/bin", "/tmp", "/home"]):
                raise PermissionError(f"Access denied to path: {path}")
        return full_path

    async def _create_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        content = input_data.get("content", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {"success": True, "message": f"File created: {path}", "path": path}

    async def _read_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}
        
        if os.path.isdir(full_path):
            return {"error": f"Path is a directory, not a file: {path}"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取文件统计信息
            stat = os.stat(full_path)
            return {
                "content": content,
                "path": path,
                "size": stat.st_size,
                "modified": stat.st_mtime
            }
        except UnicodeDecodeError:
            # 对于二进制文件，返回base64编码
            with open(full_path, 'rb') as f:
                binary_content = f.read()
            return {
                "content": base64.b64encode(binary_content).decode('utf-8'),
                "path": path,
                "encoding": "base64",
                "size": len(binary_content)
            }

    async def _write_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        content = input_data.get("content", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {"success": True, "message": f"File written: {path}", "path": path}

    async def _edit_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """编辑文件的特定行或进行查找替换"""
        path = input_data.get("path", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}
        
        # 读取现有内容
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 支持不同的编辑模式
        if "line_number" in input_data and "new_content" in input_data:
            # 按行号编辑
            line_num = int(input_data["line_number"]) - 1  # 转换为0索引
            if 0 <= line_num < len(lines):
                lines[line_num] = input_data["new_content"] + "\n"
            else:
                return {"error": f"Line number {input_data['line_number']} out of range"}
        
        elif "find" in input_data and "replace" in input_data:
            # 查找替换模式
            find_text = input_data["find"]
            replace_text = input_data["replace"]
            content = ''.join(lines)
            if find_text in content:
                content = content.replace(find_text, replace_text)
                lines = content.splitlines(keepends=True)
            else:
                return {"error": f"Text '{find_text}' not found in file"}
        
        else:
            return {"error": "Edit operation requires either 'line_number' and 'new_content', or 'find' and 'replace'"}
        
        # 写回文件
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return {"success": True, "message": f"File edited: {path}", "path": path}

    async def _delete_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}
        
        if os.path.isdir(full_path):
            # 如果是目录，递归删除
            shutil.rmtree(full_path)
            return {"success": True, "message": f"Directory deleted: {path}"}
        else:
            os.remove(full_path)
            return {"success": True, "message": f"File deleted: {path}"}

    async def _list_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", ".")
        recursive = input_data.get("recursive", False)
        show_hidden = input_data.get("show_hidden", False)
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"Directory not found: {path}"}
        
        if not os.path.isdir(full_path):
            return {"error": f"Path is not a directory: {path}"}
        
        items = []
        
        if recursive:
            # 递归列出所有文件和目录
            for root, dirs, files in os.walk(full_path):
                # 处理目录
                for dirname in dirs:
                    if not show_hidden and dirname.startswith('.'):
                        continue
                    dir_path = os.path.join(root, dirname)
                    rel_path = os.path.relpath(dir_path, full_path)
                    items.append({
                        "name": dirname,
                        "path": rel_path,
                        "type": "directory",
                        "size": 0
                    })
                
                # 处理文件
                for filename in files:
                    if not show_hidden and filename.startswith('.'):
                        continue
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, full_path)
                    stat = os.stat(file_path)
                    items.append({
                        "name": filename,
                        "path": rel_path,
                        "type": "file",
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
        else:
            # 只列出当前目录
            for item in os.listdir(full_path):
                if not show_hidden and item.startswith('.'):
                    continue
                
                item_path = os.path.join(full_path, item)
                is_dir = os.path.isdir(item_path)
                stat = os.stat(item_path)
                
                items.append({
                    "name": item,
                    "type": "directory" if is_dir else "file",
                    "size": 0 if is_dir else stat.st_size,
                    "modified": stat.st_mtime
                })
        
        return {"items": items, "path": path, "total": len(items)}

    async def _create_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        if not path:
            return {"error": "No directory path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(full_path, exist_ok=True)
        return {"success": True, "message": f"Directory created: {path}", "path": path}

    async def _run_command(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        command = input_data.get("command", "")
        if not command:
            return {"error": "No command provided"}
        
        # 扩展安全命令列表，包含更多开发常用命令
        safe_commands = [
            "ls", "pwd", "echo", "cat", "head", "tail", "touch", "mkdir", "rm", "cp", "mv", 
            "grep", "find", "wc", "sort", "uniq", "diff", "which", "whereis", "file",
            "python", "python3", "pip", "pip3", "node", "npm", "git", "curl", "wget",
            "make", "gcc", "g++", "javac", "java", "rustc", "cargo", "go", "dotnet"
        ]
        
        cmd_parts = command.split()
        if not cmd_parts:
            return {"error": "Empty command"}
        
        base_command = cmd_parts[0]
        
        # 对于一些命令，允许带路径的版本
        if "/" in base_command:
            base_command = os.path.basename(base_command)
        
        if base_command not in safe_commands:
            return {"error": f"Command not allowed: {base_command}"}
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env["PWD"] = self.working_directory
            
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=60)  # 增加超时时间
            
            return {
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "return_code": result.returncode,
                "command": command
            }
        except asyncio.TimeoutError:
            return {"error": "Command execution timed out"}
        except Exception as e:
            return {"error": f"Command execution failed: {str(e)}"}

    async def _search_files(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pattern = input_data.get("pattern", "")
        directory = input_data.get("directory", ".")
        case_sensitive = input_data.get("case_sensitive", True)
        
        if not pattern:
            return {"error": "No search pattern provided"}
        
        full_path = self._resolve_path(directory)
        if not os.path.exists(full_path):
            return {"error": f"Directory not found: {directory}"}
        
        matches = []
        search_pattern = pattern if case_sensitive else pattern.lower()
        
        for root, dirs, files in os.walk(full_path):
            for file in files:
                file_to_check = file if case_sensitive else file.lower()
                if search_pattern in file_to_check:
                    rel_path = os.path.relpath(os.path.join(root, file), self.working_directory)
                    matches.append({
                        "path": rel_path,
                        "name": file,
                        "directory": os.path.relpath(root, self.working_directory)
                    })
        
        return {"matches": matches, "pattern": pattern, "total": len(matches)}

    async def _find_in_files(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """在文件内容中搜索文本"""
        search_text = input_data.get("search_text", "")
        directory = input_data.get("directory", ".")
        file_pattern = input_data.get("file_pattern", "*")
        case_sensitive = input_data.get("case_sensitive", True)
        
        if not search_text:
            return {"error": "No search text provided"}
        
        full_path = self._resolve_path(directory)
        if not os.path.exists(full_path):
            return {"error": f"Directory not found: {directory}"}
        
        matches = []
        search_term = search_text if case_sensitive else search_text.lower()
        
        for root, dirs, files in os.walk(full_path):
            for file in files:
                # 简单的文件模式匹配
                if file_pattern != "*" and file_pattern not in file:
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    file_matches = []
                    for line_num, line in enumerate(lines, 1):
                        line_to_check = line if case_sensitive else line.lower()
                        if search_term in line_to_check:
                            file_matches.append({
                                "line_number": line_num,
                                "line_content": line.strip(),
                                "match_position": line_to_check.find(search_term)
                            })
                    
                    if file_matches:
                        rel_path = os.path.relpath(file_path, self.working_directory)
                        matches.append({
                            "file": rel_path,
                            "matches": file_matches,
                            "total_matches": len(file_matches)
                        })
                
                except (UnicodeDecodeError, PermissionError):
                    # 跳过二进制文件或无权限文件
                    continue
        
        return {
            "results": matches,
            "search_text": search_text,
            "total_files": len(matches),
            "total_matches": sum(result["total_matches"] for result in matches)
        }

    async def _get_file_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取文件详细信息"""
        path = input_data.get("path", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}
        
        stat = os.stat(full_path)
        is_dir = os.path.isdir(full_path)
        
        info = {
            "path": path,
            "name": os.path.basename(path),
            "type": "directory" if is_dir else "file",
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "permissions": oct(stat.st_mode)[-3:],
        }
        
        if not is_dir:
            # 对于文件，添加更多信息
            _, ext = os.path.splitext(path)
            info["extension"] = ext
            
            # 尝试获取行数（仅对文本文件）
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                info["lines"] = len(lines)
                info["encoding"] = "utf-8"
            except UnicodeDecodeError:
                info["encoding"] = "binary"
        else:
            # 对于目录，统计子项数量
            try:
                items = os.listdir(full_path)
                info["items_count"] = len(items)
            except PermissionError:
                info["items_count"] = "permission_denied"
        
        return info

    async def _get_working_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取当前工作目录"""
        return {
            "working_directory": self.working_directory,
            "absolute_path": os.path.abspath(self.working_directory)
        }

    async def _move_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        source = input_data.get("source", "")
        destination = input_data.get("destination", "")
        
        if not source or not destination:
            return {"error": "Source and destination paths required"}
        
        src_path = self._resolve_path(source)
        dst_path = self._resolve_path(destination)
        
        if not os.path.exists(src_path):
            return {"error": f"Source file not found: {source}"}
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)
        
        return {"success": True, "message": f"File moved: {source} → {destination}"}

    async def _copy_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        source = input_data.get("source", "")
        destination = input_data.get("destination", "")
        
        if not source or not destination:
            return {"error": "Source and destination paths required"}
        
        src_path = self._resolve_path(source)
        dst_path = self._resolve_path(destination)
        
        if not os.path.exists(src_path):
            return {"error": f"Source file not found: {source}"}
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        
        return {"success": True, "message": f"File copied: {source} → {destination}"}

# ========== 转换器类 ==========
class AnthropicToGeminiConverter:
    def __init__(self):
        pass
    
    def convert_model(self, anthropic_model: str) -> str:
        anthropic_model_lower = anthropic_model.lower()
        if "opus" in anthropic_model_lower or "sonnet" in anthropic_model_lower:
            return "gemini-2.5-pro"
        elif "haiku" in anthropic_model_lower:
            return "gemini-2.5-flash"
        else:
            logger.warning(f"Model '{anthropic_model}' does not contain 'opus', 'sonnet', or 'haiku'. "
                           f"Falling back to default 'gemini-2.5-pro'.")
            return "gemini-2.5-pro"

    def convert_request(self, request: MessagesRequest) -> Dict[str, Any]:
        gemini_model = self.convert_model(request.model)
        gemini_messages = []
        
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            parts = []
            
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if block.type == "text":
                        parts.append({"text": block.text})
                    elif block.type == "image":
                        # 处理图像内容
                        parts.append({"inline_data": block.source})
                    elif block.type == "tool_use":
                        # 处理工具使用
                        parts.append({
                            "function_call": {
                                "name": block.name,
                                "args": block.input
                            }
                        })
                    elif block.type == "tool_result":
                        # 处理工具结果
                        parts.append({
                            "function_response": {
                                "name": "tool_result",
                                "response": {
                                    "content": json.dumps(block.content) if not isinstance(block.content, str) else block.content,
                                    "tool_use_id": block.tool_use_id
                                }
                            }
                        })
            
            gemini_messages.append({"role": role, "parts": parts})
        
        converted = {
            "messages": gemini_messages,
            "model": gemini_model,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        
        if request.max_tokens:
            converted["max_tokens"] = request.max_tokens
        
        # 处理系统指令
        system_text = ""
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            elif isinstance(request.system, list) and request.system:
                system_text = "\n".join([c.text for c in request.system])
        
        if system_text:
            converted["system_instruction"] = {"parts": [{"text": system_text}]}
        
        # 处理工具
        if request.tools:
            tool_converter = ToolConverter()
            converted["tools"] = tool_converter.convert_tools_to_gemini(request.tools)
            
            if request.tool_choice:
                converted["tool_choice"] = tool_converter.convert_tool_choice_to_gemini(request.tool_choice)
        
        return converted

class GeminiToAnthropicConverter:
    def __init__(self, claude_code_simulator: ClaudeCodeToolSimulator):
        self.claude_code_simulator = claude_code_simulator

    def convert_usage(self, gemini_usage: Optional[Dict[str, int]]) -> Usage:
        if not gemini_usage:
            return Usage(input_tokens=0, output_tokens=0)
        
        input_tokens = gemini_usage.get("prompt_token_count", gemini_usage.get("input_tokens", 0))
        output_tokens = gemini_usage.get("candidates_token_count", 
                                       gemini_usage.get("completion_tokens", 
                                                       gemini_usage.get("output_tokens", 0)))
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    async def _parse_and_execute_tools(self, text: str) -> Tuple[str, List[ContentBlockToolUse]]:
        """解析文本中的工具调用并执行"""
        tool_uses = []
        
        if not isinstance(text, str):
            return "", []
        
        # 改进的正则表达式来匹配各种工具调用格式
        patterns = [
            r"<function_calls>\s*<invoke name=\"([^\"]+)\">\s*<parameter name=\"([^\"]+)\">([^<]*)</parameter>.*?</invoke>\s*</function_calls>",
            r"●\s*(.+)",  # 保持原有的简单命令格式
            r"```(\w+)\s*(.*?)```",  # 代码块格式
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            
            if pattern.startswith("<function_calls>"):
                # 处理正式的工具调用格式
                for match in matches:
                    tool_name, param_name, param_value = match
                    tool_input = {param_name: param_value}
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    tool_uses.append(ContentBlockToolUse(id=tool_use_id, name=tool_name, input=tool_input))
                    await self.claude_code_simulator.execute_tool(tool_name, tool_input)
            
            elif pattern.startswith("●"):
                # 处理简单命令格式
                for command in matches:
                    parts = command.strip().split(" ", 1)
                    tool_name_cmd = parts[0]
                    
                    # 命令到工具的映射
                    tool_map = {
                        "touch": "create_file",
                        "mkdir": "create_directory", 
                        "ls": "list_directory",
                        "cat": "read_file",
                        "rm": "delete_file",
                        "cp": "copy_file",
                        "mv": "move_file",
                        "find": "search_files",
                        "grep": "find_in_files",
                        "pwd": "get_working_directory"
                    }
                    
                    mapped_tool = tool_map.get(tool_name_cmd, "run_command")
                    tool_input = {}
                    
                    if mapped_tool == "create_file":
                        tool_input = {"path": parts[1] if len(parts) > 1 else "", "content": ""}
                    elif mapped_tool == "create_directory":
                        tool_input = {"path": parts[1] if len(parts) > 1 else ""}
                    elif mapped_tool == "list_directory":
                        tool_input = {"path": parts[1] if len(parts) > 1 else "."}
                    elif mapped_tool == "read_file":
                        tool_input = {"path": parts[1] if len(parts) > 1 else ""}
                    elif mapped_tool == "delete_file":
                        tool_input = {"path": parts[1] if len(parts) > 1 else ""}
                    elif mapped_tool == "copy_file" and len(parts) > 1:
                        file_parts = parts[1].split(" ", 1)
                        if len(file_parts) == 2:
                            tool_input = {"source": file_parts[0], "destination": file_parts[1]}
                    elif mapped_tool == "move_file" and len(parts) > 1:
                        file_parts = parts[1].split(" ", 1)
                        if len(file_parts) == 2:
                            tool_input = {"source": file_parts[0], "destination": file_parts[1]}
                    elif mapped_tool == "search_files":
                        search_parts = parts[1].split(" ") if len(parts) > 1 else []
                        if search_parts:
                            tool_input = {"pattern": search_parts[-1], "directory": "."}
                    elif mapped_tool == "find_in_files":
                        tool_input = {"search_text": parts[1] if len(parts) > 1 else "", "directory": "."}
                    elif mapped_tool == "get_working_directory":
                        tool_input = {}
                    else:
                        tool_input = {"command": command}
                    
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    tool_uses.append(ContentBlockToolUse(id=tool_use_id, name=mapped_tool, input=tool_input))
                    await self.claude_code_simulator.execute_tool(mapped_tool, tool_input)
            
            elif pattern.startswith("```"):
                # 处理代码块格式（可能包含文件创建等操作）
                for match in matches:
                    language, code_content = match
                    if language in ["bash", "sh", "shell"]:
                        # 将shell命令作为run_command执行
                        for line in code_content.strip().split('\n'):
                            if line.strip():
                                tool_input = {"command": line.strip()}
                                tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                tool_uses.append(ContentBlockToolUse(id=tool_use_id, name="run_command", input=tool_input))
                                await self.claude_code_simulator.execute_tool("run_command", tool_input)
        
        return text, tool_uses
    
    async def convert_response(self, gemini_response: Dict[str, Any], original_request: MessagesRequest) -> MessagesResponse:
        content_text = ""
        stop_reason = "end_turn"
        
        try:
            # 处理不同的响应格式
            if 'choices' in gemini_response and gemini_response['choices']:
                choice = gemini_response['choices'][0]
                if choice.get('message') and choice['message'].get('content'):
                    content_text = choice['message']['content']
                
                # 检查停止原因
                if choice.get("finish_reason") == "max_tokens":
                    stop_reason = "max_tokens"
                elif choice.get("finish_reason") == "stop":
                    stop_reason = "end_turn"
                elif choice.get("finish_reason") == "function_call":
                    stop_reason = "tool_use"
            
            elif 'candidates' in gemini_response and gemini_response['candidates']:
                # Gemini API 原生格式
                candidate = gemini_response['candidates'][0]
                if candidate.get('content') and candidate['content'].get('parts'):
                    parts = candidate['content']['parts']
                    content_text = ''.join([part.get('text', '') for part in parts if 'text' in part])
                
                if candidate.get('finishReason') == 'MAX_TOKENS':
                    stop_reason = "max_tokens"
                elif candidate.get('finishReason') == 'STOP':
                    stop_reason = "end_turn"
            
            # 解析并执行工具调用
            response_content, tool_uses = await self._parse_and_execute_tools(content_text)
            
            # 构建最终响应内容
            final_content = []
            if response_content.strip():
                final_content.append(ContentBlockText(type="text", text=response_content))
            
            if tool_uses:
                final_content.extend(tool_uses)
                stop_reason = "tool_use"
            
            return MessagesResponse(
                id=f"msg_{gemini_response.get('id', uuid.uuid4().hex)}",
                model=original_request.model,
                role="assistant",
                content=final_content,
                stop_reason=stop_reason,
                usage=self.convert_usage(gemini_response.get('usage'))
            )
            
        except (KeyError, IndexError, Exception) as e:
            logger.error(f"Error converting Gemini response: {e}\nResponse: {gemini_response}")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                model=original_request.model,
                role="assistant",
                content=[ContentBlockText(type="text", text=f"Error processing response: {e}")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0)
            )

class StreamingResponseGenerator:
    def __init__(self, original_request: MessagesRequest, claude_code_simulator: ClaudeCodeToolSimulator):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.claude_code_simulator = claude_code_simulator
        self.converter = GeminiToAnthropicConverter(claude_code_simulator)

    async def generate_sse_events(self, gemini_stream: AsyncGenerator[Dict, None]) -> AsyncGenerator[str, None]:
        # 发送消息开始事件
        yield self._create_event("message_start", {
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.original_request.model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })
        
        # 发送内容块开始事件
        yield self._create_event("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })
        
        full_response_text = ""
        output_tokens = 0
        input_tokens = 0
        
        try:
            async for chunk in gemini_stream:
                delta = ""
                
                # 处理不同的流响应格式
                if 'choices' in chunk and chunk['choices']:
                    choice = chunk['choices'][0]
                    if choice.get('delta') and choice['delta'].get('content'):
                        delta = choice['delta']['content']
                elif 'candidates' in chunk and chunk['candidates']:
                    candidate = chunk['candidates'][0]
                    if candidate.get('content') and candidate['content'].get('parts'):
                        for part in candidate['content']['parts']:
                            if 'text' in part:
                                delta += part['text']
                
                if delta:
                    full_response_text += delta
                    yield self._create_event("content_block_delta", {
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta}
                    })
                
                # 更新token计数
                if chunk.get('usage'):
                    usage = chunk['usage']
                    output_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                    input_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield self._create_event("error", {
                "error": {"type": "stream_error", "message": str(e)}
            })
        
        # 结束当前内容块
        yield self._create_event("content_block_stop", {"index": 0})
        
        # 解析并执行工具调用
        _, tool_uses = await self.converter._parse_and_execute_tools(full_response_text)
        
        # 为每个工具调用创建内容块
        if tool_uses:
            for i, tool_use in enumerate(tool_uses):
                yield self._create_event("content_block_start", {
                    "index": i + 1,
                    "content_block": tool_use.model_dump()
                })
                yield self._create_event("content_block_stop", {"index": i + 1})
        
        # 确定停止原因
        stop_reason = "tool_use" if tool_uses else "end_turn"
        
        # 发送消息增量事件
        yield self._create_event("message_delta", {
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_tokens}
        })
        
        # 发送消息停止事件
        yield self._create_event("message_stop", {})
        yield "data: [DONE]\n\n"

    def _create_event(self, event_type: str, data: Dict) -> str:
        event_data = {'type': event_type, **data}
        return f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

class ToolConverter:
    def convert_tools_to_gemini(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """将Anthropic工具格式转换为Gemini格式"""
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
        """将Anthropic工具选择格式转换为Gemini格式"""
        choice_type = tool_choice.get("type", "auto").lower()
        if choice_type == "any":
            return "ANY"
        elif choice_type == "tool":
            return tool_choice.get("name", "AUTO")
        return "AUTO"

class AnthropicAPIConfig:
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        self.claude_code_simulator = ClaudeCodeToolSimulator(self.working_directory)
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter(self.claude_code_simulator)
        self.tool_converter = ToolConverter()
        self.streaming_generator = StreamingResponseGenerator
        
        logger.info(f"🚀 Anthropic API Compatibility Layer initialized")
        logger.info(f"📁 Working directory for Claude Code: {self.working_directory}")
        logger.info(f"🔧 Claude Code tools: Enabled")
        logger.info(f"📝 Available tools: {list(self.get_available_tools().keys())}")
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """返回所有可用工具的描述"""
        return {
            "create_file": {
                "description": "Create a new file with specified content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to create"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["path"]
                }
            },
            "read_file": {
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"]
                }
            },
            "write_file": {
                "description": "Write content to a file (overwrite existing content)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["path", "content"]
                }
            },
            "edit_file": {
                "description": "Edit a file by line number or find/replace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "line_number": {"type": "integer", "description": "Line number to edit (1-based)"},
                        "new_content": {"type": "string", "description": "New content for the line"},
                        "find": {"type": "string", "description": "Text to find and replace"},
                        "replace": {"type": "string", "description": "Replacement text"}
                    },
                    "required": ["path"]
                }
            },
            "delete_file": {
                "description": "Delete a file or directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file or directory to delete"}
                    },
                    "required": ["path"]
                }
            },
            "list_directory": {
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the directory to list"},
                        "recursive": {"type": "boolean", "description": "List recursively"},
                        "show_hidden": {"type": "boolean", "description": "Show hidden files"}
                    }
                }
            },
            "create_directory": {
                "description": "Create a new directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the directory to create"}
                    },
                    "required": ["path"]
                }
            },
            "run_command": {
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"}
                    },
                    "required": ["command"]
                }
            },
            "search_files": {
                "description": "Search for files by name pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "directory": {"type": "string", "description": "Directory to search in"},
                        "case_sensitive": {"type": "boolean", "description": "Case sensitive search"}
                    },
                    "required": ["pattern"]
                }
            },
            "find_in_files": {
                "description": "Search for text within file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_text": {"type": "string", "description": "Text to search for"},
                        "directory": {"type": "string", "description": "Directory to search in"},
                        "file_pattern": {"type": "string", "description": "File pattern to match"},
                        "case_sensitive": {"type": "boolean", "description": "Case sensitive search"}
                    },
                    "required": ["search_text"]
                }
            },
            "move_file": {
                "description": "Move a file or directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source path"},
                        "destination": {"type": "string", "description": "Destination path"}
                    },
                    "required": ["source", "destination"]
                }
            },
            "copy_file": {
                "description": "Copy a file or directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source path"},
                        "destination": {"type": "string", "description": "Destination path"}
                    },
                    "required": ["source", "destination"]
                }
            },
            "get_file_info": {
                "description": "Get detailed information about a file or directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to get information about"}
                    },
                    "required": ["path"]
                }
            },
            "get_working_directory": {
                "description": "Get the current working directory",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
