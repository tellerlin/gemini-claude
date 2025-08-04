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

# --- ADDED: Token counting classes for better organization ---
class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None

class TokenCountResponse(BaseModel):
    input_tokens: int

# ========== Claude Code File System Tool Simulator (UPGRADED) ==========
class ClaudeCodeToolSimulator:
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        os.makedirs(self.working_directory, exist_ok=True)
        logger.info(f"Claude Code simulator initialized in: {self.working_directory}")

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"🔧 TOOL CALL: {tool_name} with input: {tool_input}")
            tool_map = {
                # Basic File Operations
                "create_file": self._create_file,
                "read_file": self._read_file,
                "write_file": self._write_file,
                "edit_file": self._edit_file,
                "delete_file": self._delete_file,
                "move_file": self._move_file,
                "copy_file": self._copy_file,
                "get_file_info": self._get_file_info,
                
                # Directory Operations
                "list_directory": self._list_directory,
                "create_directory": self._create_directory,
                "get_working_directory": self._get_working_directory,
                "change_directory": self._change_directory,

                # Search
                "search_files": self._search_files,
                "find_in_files": self._find_in_files,
                "grep": self._grep,

                # Command Execution
                "run_command": self._run_command,
                "run_bash": self._run_bash,
                "run_python": self._run_python,

                # Git Operations
                "git_status": self._git_status,
                "git_add": self._git_add,
                "git_commit": self._git_commit,
                "git_push": self._git_push,
                "git_pull": self._git_pull,
                "git_branch": self._git_branch,
                "git_checkout": self._git_checkout,
                "git_diff": self._git_diff,
                "git_log": self._git_log,

                # Environment Management
                "install_package": self._install_package,
                "list_packages": self._list_packages,
                
                # Networking
                "download_file": self._download_file,
                "curl": self._curl,
            }
            if tool_name in tool_map:
                return await tool_map[tool_name](tool_input)
            else:
                logger.warning(f"Unknown tool: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return {"error": str(e)}

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            full_path = os.path.abspath(path)
        else:
            full_path = os.path.abspath(os.path.join(self.working_directory, path))
        
        if ".." in os.path.normpath(path) and not full_path.startswith(os.path.abspath(self.working_directory)):
            # Allow access to common system directories for tools like compilers, etc.
            if not any(full_path.startswith(allowed) for allowed in ["/usr", "/bin", "/tmp", "/home", "/opt"]):
                raise PermissionError(f"Access denied to path: {path}. Path traversal outside of working directory is restricted.")
        return full_path

    # ========== Basic File Operations ==========
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
            
            stat = os.stat(full_path)
            return {
                "content": content, "path": path, "size": stat.st_size, "modified": stat.st_mtime
            }
        except UnicodeDecodeError:
            with open(full_path, 'rb') as f:
                binary_content = f.read()
            return {
                "content": base64.b64encode(binary_content).decode('utf-8'),
                "path": path, "encoding": "base64", "size": len(binary_content)
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
        path = input_data.get("path", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}
        
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if "line_number" in input_data and "new_content" in input_data:
            line_num = int(input_data["line_number"]) - 1
            if 0 <= line_num < len(lines):
                lines[line_num] = input_data["new_content"] + "\n"
            else:
                return {"error": f"Line number {input_data['line_number']} out of range"}
        elif "find" in input_data and "replace" in input_data:
            content = ''.join(lines).replace(input_data["find"], input_data["replace"])
            lines = content.splitlines(keepends=True)
        else:
            return {"error": "Edit op requires 'line_number' & 'new_content', or 'find' & 'replace'"}

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
            shutil.rmtree(full_path)
            return {"success": True, "message": f"Directory deleted: {path}"}
        else:
            os.remove(full_path)
            return {"success": True, "message": f"File deleted: {path}"}

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

    async def _get_file_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        if not path:
            return {"error": "No file path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {path}"}
        
        stat = os.stat(full_path)
        is_dir = os.path.isdir(full_path)
        
        info = {
            "path": path, "name": os.path.basename(path), "type": "directory" if is_dir else "file",
            "size": stat.st_size, "created": stat.st_ctime, "modified": stat.st_mtime,
            "accessed": stat.st_atime, "permissions": oct(stat.st_mode)[-3:],
        }
        
        if not is_dir:
            _, ext = os.path.splitext(path)
            info["extension"] = ext
            info["mime_type"] = mimetypes.guess_type(path)[0]
        return info

    # ========== Directory Operations ==========
    async def _list_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", ".")
        recursive = input_data.get("recursive", False)
        show_hidden = input_data.get("show_hidden", False)
        
        full_path = self._resolve_path(path)
        if not os.path.isdir(full_path):
            return {"error": f"Directory not found or not a directory: {path}"}
        
        items = []
        if recursive:
            for root, dirs, files in os.walk(full_path):
                all_nodes = dirs + files
                for node in all_nodes:
                    if not show_hidden and node.startswith('.'):
                        continue
                    node_path = os.path.join(root, node)
                    rel_path = os.path.relpath(node_path, self.working_directory)
                    stat = os.stat(node_path)
                    items.append({
                        "path": rel_path, "type": "directory" if os.path.isdir(node_path) else "file",
                        "size": stat.st_size, "modified": stat.st_mtime
                    })
        else:
            for item in os.listdir(full_path):
                if not show_hidden and item.startswith('.'):
                    continue
                item_path = os.path.join(full_path, item)
                stat = os.stat(item_path)
                items.append({
                    "path": os.path.relpath(item_path, self.working_directory),
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": stat.st_size, "modified": stat.st_mtime
                })
        return {"items": items, "path": path, "total": len(items)}

    async def _create_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        if not path:
            return {"error": "No directory path provided"}
        
        full_path = self._resolve_path(path)
        os.makedirs(full_path, exist_ok=True)
        return {"success": True, "message": f"Directory created: {path}", "path": path}

    async def _get_working_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"working_directory": self.working_directory}

    async def _change_directory(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = input_data.get("path", "")
        if not path:
            return {"error": "No directory path provided"}
        
        full_path = self._resolve_path(path)
        if not os.path.isdir(full_path):
            return {"error": f"Directory not found or not a directory: {path}"}
        
        self.working_directory = full_path
        os.chdir(full_path) # Also change process CWD
        return {"success": True, "new_working_directory": self.working_directory}

    # ========== Search Operations ==========
    async def _search_files(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pattern = input_data.get("pattern", "")
        directory = input_data.get("directory", ".")
        if not pattern:
            return {"error": "No search pattern provided"}
        
        full_path = self._resolve_path(directory)
        matches = []
        for root, _, files in os.walk(full_path):
            for file in files:
                if re.search(pattern, file):
                    matches.append(os.path.relpath(os.path.join(root, file), self.working_directory))
        return {"matches": matches}

    async def _find_in_files(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        search_text = input_data.get("search_text", "")
        directory = input_data.get("directory", ".")
        if not search_text:
            return {"error": "No search text provided"}
            
        return await self._grep({"pattern": search_text, "path": directory, "recursive": True})

    async def _grep(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pattern = input_data.get("pattern", "")
        path = input_data.get("path", ".")
        recursive = input_data.get("recursive", False)
        case_insensitive = input_data.get("case_insensitive", False)
        
        if not pattern:
            return {"error": "No pattern provided for grep"}
            
        grep_cmd = f"grep {'-r' if recursive else ''} {'-i' if case_insensitive else ''} -n --color=never -E '{pattern}' {path}"
        return await self._run_command({"command": grep_cmd})

    # ========== Command Execution ==========
    async def _run_command(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        command = input_data.get("command", "")
        if not command:
            return {"error": "No command provided"}

        # Basic security: prevent complex shell commands like pipelines, redirection etc. in the base command
        if any(c in command.split()[0] for c in ";|&<>"):
             return {"error": f"Complex shell operators not allowed in base command."}

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            
            return {
                "stdout": stdout.decode(errors='replace'),
                "stderr": stderr.decode(errors='replace'),
                "return_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"error": "Command timed out after 60 seconds."}
        except Exception as e:
            return {"error": f"Command failed: {str(e)}"}

    async def _run_bash(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        script = input_data.get("script", "")
        if not script:
            return {"error": "No script provided"}
        return await self._run_command({"command": script})
    
    async def _run_python(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code = input_data.get("code", "")
        if not code:
            return {"error": "No python code provided"}
        # Using a temporary file is safer than -c for complex scripts
        tmp_file = f"temp_script_{uuid.uuid4().hex}.py"
        await self._create_file({"path": tmp_file, "content": code})
        result = await self._run_command({"command": f"python3 {tmp_file}"})
        await self._delete_file({"path": tmp_file})
        return result

    # ========== Git Operations ==========
    async def _git_status(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_command({"command": "git status --porcelain"})

    async def _git_add(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        files = input_data.get("files", ["."])
        return await self._run_command({"command": f"git add {' '.join(files)}"})

    async def _git_commit(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        message = input_data.get("message", "")
        if not message:
            return {"error": "Commit message is required"}
        return await self._run_command({"command": f'git commit -m "{message}"'})

    async def _git_push(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        remote = input_data.get("remote", "origin")
        branch = input_data.get("branch", "")
        return await self._run_command({"command": f"git push {remote} {branch}".strip()})

    async def _git_pull(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        remote = input_data.get("remote", "origin")
        branch = input_data.get("branch", "")
        return await self._run_command({"command": f"git pull {remote} {branch}".strip()})
    
    async def _git_branch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_command({"command": "git branch"})

    async def _git_checkout(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        target = input_data.get("target", "")
        create_new = input_data.get("create_new", False)
        if not target:
            return {"error": "Target branch or path is required"}
        return await self._run_command({"command": f"git checkout {'-b' if create_new else ''} {target}"})

    async def _git_diff(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        file_path = input_data.get("file", "")
        staged = input_data.get("staged", False)
        return await self._run_command({"command": f"git diff {'--staged' if staged else ''} {file_path}"})
    
    async def _git_log(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        limit = input_data.get("limit", 10)
        return await self._run_command({"command": f"git log -n {limit} --oneline"})

    # ========== Environment Management ==========
    async def _install_package(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        package = input_data.get("package", "")
        manager = input_data.get("manager", "pip")
        if not package:
            return {"error": "Package name required"}
        if manager not in ["pip", "pip3", "npm", "yarn"]:
            return {"error": f"Unsupported package manager: {manager}"}
        return await self._run_command({"command": f"{manager} install {package}"})

    async def _list_packages(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        manager = input_data.get("manager", "pip")
        if manager not in ["pip", "pip3", "npm", "yarn"]:
            return {"error": f"Unsupported package manager: {manager}"}
        return await self._run_command({"command": f"{manager} list"})

    # ========== Networking ==========
    async def _download_file(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data.get("url", "")
        output_path = input_data.get("output", "")
        if not url:
            return {"error": "URL required"}
        return await self._run_command({"command": f"curl -L -o {output_path}" if output_path else f"curl -L -O {url}"})
    
    async def _curl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data.get("url", "")
        if not url:
            return {"error": "URL required"}
        return await self._run_command({"command": f"curl -L {url}"})


# ========== Conversion Classes ==========
class AnthropicToGeminiConverter:
    def __init__(self):
        pass
    
    def convert_model(self, anthropic_model: str) -> str:
        """
        Converts an Anthropic model name to a corresponding Gemini model name
        with flexible matching.

        - "sonnet" and "opus" variants map to "gemini-2.5-pro".
        - "haiku" variants map to "gemini-2.0-flash".
        - Defaults to "gemini-2.5-pro" for any other model.
        """
        anthropic_model_lower = anthropic_model.lower()
        
        if "sonnet" in anthropic_model_lower or "opus" in anthropic_model_lower:
            return "gemini-2.5-pro"
        elif "haiku" in anthropic_model_lower:
            return "gemini-2.0-flash"
        else:
            logger.warning(f"Model '{anthropic_model}' not found in specific mappings, falling back to default 'gemini-2.5-pro'.")
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
                        parts.append({"inline_data": block.source})
                    elif block.type == "tool_use":
                        parts.append({
                            "function_call": {
                                "name": block.name,
                                "args": block.input
                            }
                        })
                    elif block.type == "tool_result":
                        parts.append({
                            "function_response": {
                                "name": block.name if hasattr(block, 'name') else 'tool_result',
                                "response": {
                                    "name": block.name if hasattr(block, 'name') else 'tool_result',
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
        
        system_text = ""
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            elif isinstance(request.system, list):
                system_text = "\n".join([c.text for c in request.system])
        
        if system_text:
            converted["system_instruction"] = {"parts": [{"text": system_text}]}
        
        if request.tools:
            tool_converter = ToolConverter()
            converted["tools"] = tool_converter.convert_tools_to_gemini(request.tools)
            if request.tool_choice:
                converted["tool_config"] = {"function_calling_config": {"mode": tool_converter.convert_tool_choice_to_gemini(request.tool_choice)}}

        return converted

class GeminiToAnthropicConverter:
    def __init__(self, claude_code_simulator: ClaudeCodeToolSimulator):
        self.claude_code_simulator = claude_code_simulator

    def convert_usage(self, gemini_usage: Optional[Dict[str, int]]) -> Usage:
        if not gemini_usage:
            return Usage(input_tokens=0, output_tokens=0)
        
        return Usage(
            input_tokens=gemini_usage.get("prompt_token_count", 0),
            output_tokens=gemini_usage.get("candidates_token_count", 0)
        )

    async def _parse_and_handle_tool_calls(self, gemini_parts: list) -> Tuple[List, str]:
        anthropic_content = []
        stop_reason = "end_turn"

        for part in gemini_parts:
            if "text" in part and part["text"]:
                anthropic_content.append(ContentBlockText(type="text", text=part["text"]))
            elif "function_call" in part:
                tool_call = part["function_call"]
                tool_use_id = f"toolu_{uuid.uuid4().hex[:8]}"
                anthropic_content.append(ContentBlockToolUse(
                    id=tool_use_id,
                    name=tool_call["name"],
                    input=tool_call.get("args", {})
                ))
                stop_reason = "tool_use"
        
        return anthropic_content, stop_reason

    async def convert_response(self, gemini_response: Dict[str, Any], original_request: MessagesRequest) -> MessagesResponse:
        try:
            candidate = gemini_response['candidates'][0]
            gemini_parts = candidate.get('content', {}).get('parts', [])
            
            final_content, stop_reason = await self._parse_and_handle_tool_calls(gemini_parts)

            if candidate.get('finishReason') == 'MAX_TOKENS':
                stop_reason = "max_tokens"
            elif candidate.get('finishReason') == 'STOP' and stop_reason != 'tool_use':
                stop_reason = "end_turn"
            
            # If no text or tool calls, add an empty text block as per Anthropic spec
            if not final_content:
                final_content.append(ContentBlockText(text=""))
                
            return MessagesResponse(
                id=f"msg_{gemini_response.get('id', uuid.uuid4().hex)}",
                model=original_request.model,
                role="assistant",
                content=final_content,
                stop_reason=stop_reason,
                usage=self.convert_usage(gemini_response.get('usageMetadata'))
            )
        except (KeyError, IndexError) as e:
            logger.error(f"Error converting Gemini response: {e}\nResponse: {gemini_response}")
            return MessagesResponse(
                id=f"msg_error_{uuid.uuid4().hex[:8]}", model=original_request.model,
                role="assistant", content=[ContentBlockText(text=f"Error processing response: {str(e)}")],
                stop_reason="end_turn", usage=Usage(input_tokens=0, output_tokens=0)
            )

class StreamingResponseGenerator:
    def __init__(self, original_request: MessagesRequest, claude_code_simulator: ClaudeCodeToolSimulator):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:8]}"
        self.claude_code_simulator = claude_code_simulator

    async def generate_sse_events(self, gemini_stream: AsyncGenerator[Dict, None]) -> AsyncGenerator[str, None]:
        yield self._create_event("message_start", {"message": {
            "id": self.message_id, "type": "message", "role": "assistant",
            "model": self.original_request.model, "content": [], "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }})
        
        full_response_parts = []
        input_tokens = 0
        output_tokens = 0

        async for chunk in gemini_stream:
            usage_metadata = chunk.get('usageMetadata', {})
            if "prompt_token_count" in usage_metadata:
                input_tokens = usage_metadata["prompt_token_count"]
                yield self._create_event("message_delta", {"usage": {"input_tokens": input_tokens}})

            if not chunk.get('candidates'):
                continue
            
            delta_parts = chunk['candidates'][0].get('content', {}).get('parts', [])
            
            for part in delta_parts:
                if "text" in part and part["text"]:
                    yield self._start_content_block_if_needed(full_response_parts, "text")
                    delta_text = part["text"]
                    yield self._create_event("content_block_delta", {"index": len(full_response_parts) - 1, "delta": {"type": "text_delta", "text": delta_text}})
                    self._update_full_response(full_response_parts, "text", delta_text)
                
                elif "function_call" in part:
                    yield self._stop_last_content_block_if_needed(full_response_parts)
                    tool_call = part["function_call"]
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:8]}"
                    tool_use_content = ContentBlockToolUse(id=tool_use_id, name=tool_call['name'], input=tool_call.get('args', {}))
                    yield self._create_event("content_block_start", {"index": len(full_response_parts), "content_block": tool_use_content.model_dump()})
                    full_response_parts.append(tool_use_content)
                    yield self._create_event("content_block_stop", {"index": len(full_response_parts) - 1})

        yield self._stop_last_content_block_if_needed(full_response_parts)
        
        final_chunk = await gemini_stream.__anext__()
        final_usage = final_chunk.get('usageMetadata', {})
        output_tokens = final_usage.get('candidates_token_count', output_tokens)

        stop_reason = "tool_use" if any(isinstance(p, ContentBlockToolUse) for p in full_response_parts) else "end_turn"
        
        yield self._create_event("message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": output_tokens}})
        yield self._create_event("message_stop", {})

    def _update_full_response(self, parts, type, text_delta):
        if parts and isinstance(parts[-1], ContentBlockText):
            parts[-1].text += text_delta
        else:
             parts.append(ContentBlockText(text=text_delta))

    def _start_content_block_if_needed(self, parts, type):
        if not parts or not isinstance(parts[-1], ContentBlockText):
            return self._create_event("content_block_start", {"index": len(parts), "content_block": {"type": "text", "text": ""}})
        return "" # Return empty string if no event is needed

    def _stop_last_content_block_if_needed(self, parts):
         if parts and isinstance(parts[-1], ContentBlockText):
            return self._create_event("content_block_stop", {"index": len(parts) - 1})
         return ""

    def _create_event(self, event_type: str, data: Dict) -> str:
        if not data: return ""
        json_data = json.dumps({'type': event_type, **data})
        return f"event: {event_type}\ndata: {json_data}\n\n"

class ToolConverter:
    def convert_tools_to_gemini(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Converts Anthropic tools to the standardized OpenAI tool format that LiteLLM expects.
        LiteLLM will then handle the conversion to the provider-specific format (Gemini).
        """
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

    def convert_tool_choice_to_gemini(self, tool_choice: Dict[str, Any]) -> str:
        choice_type = tool_choice.get("type", "auto").lower()
        if choice_type == "any":
            return "ANY"
        elif choice_type == "tool":
            return "ANY" # Gemini does not support forcing a specific tool by name yet
        return "AUTO"

class AnthropicAPIConfig:
    def __init__(self, working_directory: str = "."):
        self.working_directory = os.path.abspath(working_directory)
        self.claude_code_simulator = ClaudeCodeToolSimulator(self.working_directory)
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter(self.claude_code_simulator)
        self.tool_converter = ToolConverter()
        
        logger.info(f"🚀 Anthropic API Compatibility Layer initialized")
        logger.info(f"📁 Working directory for Claude Code: {self.working_directory}")
        logger.info(f"🔧 Available tools: {list(self.get_available_tools().keys())}")
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Returns the schemas for all available tools."""
        return {
            "create_file": {"description": "Create a new file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path"]}},
            "read_file": {"description": "Read content from a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            "write_file": {"description": "Write (overwrite) content to a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            "edit_file": {"description": "Edit a file by line or find/replace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "line_number": {"type": "integer"}, "new_content": {"type": "string"}, "find": {"type": "string"}, "replace": {"type": "string"}}, "required": ["path"]}},
            "delete_file": {"description": "Delete a file or directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            "move_file": {"description": "Move a file or directory.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
            "copy_file": {"description": "Copy a file or directory.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
            "get_file_info": {"description": "Get detailed info about a file/dir.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            "list_directory": {"description": "List directory contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "recursive": {"type": "boolean", "default": False}, "show_hidden": {"type": "boolean", "default": False}}}},
            "create_directory": {"description": "Create a new directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            "get_working_directory": {"description": "Get the current working directory.", "parameters": {"type": "object", "properties": {}}},
            "change_directory": {"description": "Change the current working directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            "search_files": {"description": "Search for files by regex pattern.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "directory": {"type": "string", "default": "."}}, "required": ["pattern"]}},
            "find_in_files": {"description": "Find text within files (uses grep).", "parameters": {"type": "object", "properties": {"search_text": {"type": "string"}, "directory": {"type": "string", "default": "."}}, "required": ["search_text"]}},
            "grep": {"description": "Execute grep command.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}, "recursive": {"type": "boolean", "default": False}, "case_insensitive": {"type": "boolean", "default": False}}, "required": ["pattern"]}},
            "run_command": {"description": "Execute a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            "run_bash": {"description": "Execute a bash script.", "parameters": {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"]}},
            "run_python": {"description": "Execute python code.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
            "git_status": {"description": "Get git status.", "parameters": {"type": "object", "properties": {}}},
            "git_add": {"description": "Add files to git.", "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}, "default": ["."]}}}},
            "git_commit": {"description": "Commit changes.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
            "git_push": {"description": "Push changes to a remote.", "parameters": {"type": "object", "properties": {"remote": {"type": "string", "default": "origin"}, "branch": {"type": "string", "default": ""}}}},
            "git_pull": {"description": "Pull changes from a remote.", "parameters": {"type": "object", "properties": {"remote": {"type": "string", "default": "origin"}, "branch": {"type": "string", "default": ""}}}},
            "git_branch": {"description": "List git branches.", "parameters": {"type": "object", "properties": {}}},
            "git_checkout": {"description": "Checkout a branch or path.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "create_new": {"type": "boolean", "default": False}}, "required": ["target"]}},
            "git_diff": {"description": "Show git diff.", "parameters": {"type": "object", "properties": {"file": {"type": "string", "default": ""}, "staged": {"type": "boolean", "default": False}}}},
            "git_log": {"description": "Show git log.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}},
            "install_package": {"description": "Install a package.", "parameters": {"type": "object", "properties": {"package": {"type": "string"}, "manager": {"type": "string", "enum": ["pip", "npm"], "default": "pip"}}, "required": ["package"]}},
            "list_packages": {"description": "List installed packages.", "parameters": {"type": "object", "properties": {"manager": {"type": "string", "enum": ["pip", "npm"], "default": "pip"}}}},
            "download_file": {"description": "Download a file from a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "output": {"type": "string", "default": ""}}, "required": ["url"]}},
            "curl": {"description": "Make a GET request to a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        }
