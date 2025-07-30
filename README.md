# Gemini Claude Adapter

高性能的 Gemini Claude 适配器，专为 Claude Code 和本地客户端设计，支持多 API 密钥轮换、自动故障恢复和流式响应。

[🇨🇳 中文版本](README.md) | [🇺🇸 English Version](README.en.md)

## ✨ 核心特性

- 🚀 **极速响应** - 优化的请求处理和密钥轮换算法
- 🔑 **智能密钥管理** - 失败密钥立即冷却，自动切换到下一个可用密钥
- 🌐 **完全兼容** - 兼容 Claude Code 和 OpenAI API 格式
- ⚡ **流式支持** - 原生支持流式聊天响应
- 🛡️ **企业级特性** - 代理支持、CORS 处理、错误恢复
- 📊 **实时监控** - 详细的服务状态和密钥使用统计

## 🎯 设计目标

- **响应最快**: 优化的密钥轮换策略，失败立即切换
- **兼容性最强**: 支持 Claude Code 和各种客户端
- **稳定性最高**: 自动故障恢复和密钥冷却机制

## 🚀 快速开始

### 开发环境

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd gemini-claude
   ```

2. **创建虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   nano .env
   ```
   添加你的 Gemini API 密钥：
   ```bash
   GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...
   ```

5. **启动开发服务器**
   ```bash
   python main.py
   ```

服务器将在 `http://localhost:8000` 启动。

### 生产部署

使用自动化部署脚本：

#### Ubuntu/Debian 系统
```bash
# 方法 1：使用 Git Clone（推荐）
export GITHUB_REPO_URL=https://github.com/tellerlin/gemini-claude.git
sudo bash scripts/deploy.sh

# 方法 2：手动上传
scp -r gemini-claude/ user@your-vps-ip:~/
ssh user@your-vps-ip
cd gemini-claude
sudo bash scripts/deploy.sh
```

#### CentOS/RHEL 系统
```bash
# 方法 1：使用 Git Clone（推荐）
export GITHUB_REPO_URL=https://github.com/tellerlin/gemini-claude.git
sudo bash scripts/deploy-centos.sh

# 方法 2：手动上传
scp -r gemini-claude/ user@your-vps-ip:~/
ssh user@your-vps-ip
cd gemini-claude
sudo bash scripts/deploy-centos.sh
```

**支持的系统版本：**
- Ubuntu 20.04, 22.04, 24.04
- Debian 11, 12
- CentOS Stream 8, 9
- RHEL 8, 9
- Rocky Linux 8, 9
- AlmaLinux 8, 9

### 📝 部署后配置

部署脚本会自动创建 `.env` 文件（从 `.env.example` 复制），你需要编辑它来配置你的 API 密钥：

```bash
# 编辑 .env 文件
sudo nano /home/gemini/gemini-claude/.env

# 添加你的 Gemini API 密钥
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...

# 启动服务
gemini-manage restart

# 检查服务状态
gemini-manage status
```

**⚠️ 故障排除：**

如果 `gemini-manage` 命令不可用，请尝试以下方法：

1. **重新运行部署脚本**（确保以 root 身份运行）：
   ```bash
   sudo bash scripts/deploy.sh          # Ubuntu/Debian
   sudo bash scripts/deploy-centos.sh   # CentOS/RHEL
   ```

2. **手动检查服务状态**：
   ```bash
   # 检查进程是否运行
   ps aux | grep uvicorn
   
   # 检查端口是否监听
   netstat -tlnp | grep 8000
   
   # 检查服务健康状态
   curl http://localhost:8000/health
   ```

3. **手动启动服务**（如果 Supervisor 不可用）：
   ```bash
   cd /home/gemini/gemini-claude
   sudo -u gemini bash -c "source venv/bin/activate && nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &"
   ```

4. **使用快速修复脚本**（推荐）：
   ```bash
   # 运行快速修复脚本，自动诊断和修复问题
   ./quick-fix.sh
   ```

**重要配置项：**
- `GEMINI_API_KEYS`: 你的 Gemini API 密钥，多个密钥用逗号分隔
- `PROXY_URL`: 如果需要使用代理，取消注释并设置
- `MAX_FAILURES`: 失败次数阈值（推荐：1-3）
- `COOLING_PERIOD`: 冷却时间秒数（推荐：300）
- `REQUEST_TIMEOUT`: 请求超时秒数（推荐：45）

## 📡 API 端点

### 聊天完成
```
POST /v1/chat/completions
```

请求格式（兼容 OpenAI）：
```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "model": "gemini-2.5-pro",
  "temperature": 0.7,
  "stream": false
}
```

### 健康检查
```
GET /health
```

### 统计信息
```
GET /stats
```

### 可用模型
```
GET /v1/models
```

## 🔧 配置说明

### 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `GEMINI_API_KEYS` | Gemini API 密钥列表，用逗号分隔 | 必填 |
| `PROXY_URL` | 代理服务器地址 | 可选 |
| `MAX_FAILURES` | 失败次数阈值 | 1 |
| `COOLING_PERIOD` | 冷却时间（秒） | 300 |
| `REQUEST_TIMEOUT` | 请求超时（秒） | 45 |
| `MAX_RETRIES` | 重试次数 | 0 |

### API 密钥格式支持

支持多种格式：
```
AIzaSyABC123...,AIzaSyDEF456...
"AIzaSyABC123...","AIzaSyDEF456..."
'AIzaSyABC123...','AIzaSyDEF456...'
AIzaSyABC123..., "AIzaSyDEF456...", 'AIzaSyGHI789...'
```

## 🛠️ 管理命令

### 本地开发
```bash
# 启动开发服务器
python main.py

# 使用客户端测试
python client/client.py http://localhost:8000
```

### 生产环境
```bash
# 服务管理
gemini-manage start
gemini-manage stop
gemini-manage restart
gemini-manage status

# 查看日志
gemini-manage logs
gemini-manage error-logs

# 更新依赖
gemini-manage update

# 备份配置
gemini-manage backup
```

## 🎮 客户端使用

### 交互式聊天
```bash
python client/client.py http://your-vps-ip
```

### 程序化使用
```python
from client.client import RemoteGeminiClient, ClientConfig

config = ClientConfig(
    vps_url="http://your-vps-ip",
    timeout=120,
    retries=3,
    preferred_model="gemini-2.5-pro"
)

client = RemoteGeminiClient(config)

# 发送消息
response = await client.chat_completion([
    {"role": "user", "content": "Hello!"}
])
```

## 🔒 安全特性

- 🔐 API 密钥安全存储和传输
- 🛡️ CORS 保护
- 🚫 请求频率限制
- 📝 详细的访问日志
- 🔒 防火墙配置

## 📊 监控和日志

- 实时密钥状态监控
- 详细的请求日志
- 错误追踪和分析
- 性能指标统计

## 🐛 故障排除

### 常见问题

1. **API 密钥无效**
   - 检查 `.env` 文件中的密钥格式
   - 确保密钥有效且未过期

2. **连接超时**
   - 检查网络连接
   - 考虑使用代理
   - 调整 `REQUEST_TIMEOUT` 值

3. **服务不可用**
   - 检查服务器状态：`gemini-manage status`
   - 查看错误日志：`gemini-manage error-logs`

4. **`gemini-manage` 命令不可用**
   - 确保以 root 身份运行部署脚本
   - 检查 `/usr/local/bin/gemini-manage` 是否存在
   - 尝试重新运行部署脚本

5. **Supervisor 服务未启动**
   - 检查 Supervisor 状态：`systemctl status supervisord`
   - 手动启动 Supervisor：`systemctl start supervisord`
   - 如果 Supervisor 不可用，使用手动启动方法

6. **端口 8000 被占用**
   - 检查端口占用：`netstat -tlnp | grep 8000`
   - 终止占用进程：`kill -9 <PID>`
   - 修改 `.env` 文件中的 `PORT` 值

7. **服务启动失败或管理脚本错误**
   - 运行快速修复脚本：`./quick-fix.sh`
   - 该脚本会自动诊断并修复常见问题
   - 检查依赖、权限、配置和服务状态

### 日志位置

- 应用日志：`/home/gemini/gemini-claude/logs/app.log`
- 错误日志：`/home/gemini/gemini-claude/logs/error.log`
- 部署日志：`/tmp/gemini_deployment.log`

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📞 支持

如有问题，请查看文档或提交 Issue。

---

**[🇨🇳 中文版本](README.md)** | **[🇺🇸 Switch to English](README.en.md)**
