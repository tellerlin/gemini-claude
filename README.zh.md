# Gemini Claude Adapter v2.1.0

高性能的 Gemini Claude 适配器，专为 Claude Code 和本地客户端设计，支持多 API 密钥轮换、自动故障恢复和流式响应。

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

## ✨ 核心特性

-   🤖 **完整 Anthropic API 兼容性** - 完全支持 Anthropic Messages API (`/v1/messages`)，包含正确的流式格式
-   🔑 **智能密钥管理** - 失败的 Gemini 密钥立即进入冷却期，自动故障转移到下一个可用密钥
-   🛡️ **强大安全性** - 对所有敏感端点强制执行 API 密钥认证，使用客户端和管理员密钥
-   🌐 **双 API 支持** - 兼容 Anthropic 和 OpenAI API 格式，实现最大灵活性
-   ⚡ **流式支持** - 原生支持 Anthropic 风格的流式响应，包含所有必需的事件类型
-   🛠️ **工具调用支持** - Anthropic 和 Gemini 格式之间的完整工具/函数调用支持
-   📊 **实时监控** - 服务健康、密钥状态和使用统计的端点
-   🐳 **简化 Docker 部署** - 使用 Docker 和 Docker Compose 快速安全设置

### 🚀 v2.1.0 新特性

-   **性能优化** - 智能响应缓存和 HTTP 连接池
-   **增强错误处理** - 智能错误分类和断路器模式
-   **高级监控** - 综合指标收集和性能跟踪
-   **结构化配置** - 具有环境支持的分层配置系统
-   **改进可靠性** - 更好的容错和自动恢复机制

## 🔒 安全优先：理解认证机制

此适配器强制 API 密钥认证来保护您的服务。有两个访问级别：

1.  **客户端密钥 (`ADAPTER_API_KEYS`)**：面向标准用户。这些密钥授予核心功能访问权限，如聊天完成 (`/v1/chat/completions`) 和列出模型 (`/v1/models`)。
2.  **管理员密钥 (`ADMIN_API_KEYS`)**：面向管理员。这些密钥授予所有端点的访问权限，包括受保护的管理端点，如重置 Gemini 密钥 (`/admin/reset-key/{prefix}`)。

如果未设置 `ADMIN_API_KEYS`，客户端密钥也将有管理员端点的访问权限。对于生产环境，**强烈建议**设置单独的管理员密钥。

认证通过 `X-API-Key` 头或 `Authorization: Bearer <token>` 头处理。

## 🚀 快速开始指南

### 先决条件

-   **Docker** 和 **Docker Compose** 已安装在您的系统上
-   **Google Gemini API 密钥** ([在此获取](https://makersuite.google.com/app/apikey))
-   **Git** 用于克隆仓库

### 步骤 1：获取您的 API 密钥

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 使用您的 Google 账户登录
3. 创建一个或多个 API 密钥
4. 复制密钥（以 `AIza...` 开头）

### 步骤 2：部署服务

```bash
# 克隆仓库
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# 复制并配置环境变量
cp .env.example .env

# 编辑配置（参见下面的配置部分）
nano .env  # 或使用您喜欢的编辑器

# 启动服务
docker-compose up -d

# 检查是否正在运行
docker-compose ps
docker-compose logs -f
```

服务将在 `http://localhost:8000`（或您服务器的 IP）可用。

### 步骤 3：测试您的部署

```bash
# 基本健康检查（无需认证）
curl http://localhost:8000/health

# 使用您的 API 密钥测试
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-client-key-123"
```

## ⚙️ 基本配置

编辑您的 `.env` 文件，使用这些**必需**的设置：

```env
# =============================================
# 必需：Gemini API 配置
# =============================================
# 从以下获取您的 API 密钥：https://makersuite.google.com/app/apikey
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...

# =============================================
# 必需：安全配置
# =============================================
# 生成强密钥：openssl rand -hex 32
SECURITY_ADAPTER_API_KEYS=your-client-key-123,your-client-key-456

# =============================================
# 可选：管理员访问
# =============================================
# 管理端点的可选管理员密钥
SECURITY_ADMIN_API_KEYS=your-admin-key-abc,your-admin-key-def

# =============================================
# 可选：服务配置
# =============================================
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
SERVICE_LOG_LEVEL=INFO
```

**⚠️ 重要安全说明：**
- 保护您的 `.env` 文件，切勿提交到版本控制
- 为 `SECURITY_ADAPTER_API_KEYS` 使用强而唯一的 API 密钥
- 为生产环境设置 `SECURITY_ADMIN_API_KEYS`
- 使用以下方式生成安全密钥：`openssl rand -hex 32`

### 步骤 3：启动服务

配置好 `.env` 文件后，使用 Docker Compose 启动服务。

```bash
docker-compose up -d
```

服务现在将在后台运行。API 将在 `http://localhost:8000`（或您服务器的 IP 地址）访问。

### 步骤 4：管理服务

以下是用于管理服务的基本 Docker Compose 命令：

-   **检查日志**：`docker-compose logs -f`
-   **停止服务**：`docker-compose down`
-   **重启服务**：`docker-compose restart`

## 🔄 更新项目

当 Gemini Claude Adapter 发布新更新时，请按照以下步骤更新您的部署：

### 方法 1：Git Pull（推荐）

此方法保留您的自定义配置，同时更新应用程序代码：

```bash
# 导航到您的项目目录
cd gemini-claude

# 停止运行的服务
docker-compose down

# 拉取最新更改
git pull origin main

# 重新构建并重启服务
docker-compose up -d --build

# 检查服务状态
docker-compose ps
docker-compose logs -f
```

### 方法 2：手动更新

如果您对代码进行了自定义修改：

```bash
# 停止服务
docker-compose down

# 备份您的 .env 文件（重要！）
cp .env .env.backup

# 删除旧的项目目录（可选）
rm -rf gemini-claude

# 克隆最新版本
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# 恢复您的配置
cp ../.env.backup .env

# 启动服务
docker-compose up -d
```

### 重要说明

- **配置保留**：您的 `.env` 文件包含您的 API 密钥和设置。更新前请务必备份。
- **数据库/日志**：`logs/` 目录被挂载为卷，因此您的日志将被保留。
- **Docker 镜像**：`--build` 标志确保 Docker 使用最新代码重新构建镜像。
- **破坏性更改**：检查项目的发布说明或提交历史记录，了解可能需要配置更新的破坏性更改。

### 什么会被更新

- 应用程序代码和功能
- 安全补丁和改进
- Docker 配置
- 依赖项和要求

### 什么会被保留

- 您的 `.env` 配置文件
- `logs/` 目录中的应用程序日志
- Docker 卷和数据
- 您的自定义设置

## ⚙️ 完整配置参考

### 环境变量

| 变量 | 必需 | 默认值 | 描述 |
|------|------|--------|------|
| `GEMINI_API_KEYS` | 是 | - | 逗号分隔的 Google Gemini API 密钥 |
| `ADAPTER_API_KEYS` | 是 | - | 客户端认证密钥（使用 `openssl rand -hex 32` 生成） |
| `ADMIN_API_KEYS` | 否 | - | 管理员认证密钥（生产环境推荐） |
| `HOST` | 否 | `0.0.0.0` | 容器内绑定的主机 |
| `PORT` | 否 | `8000` | 容器内绑定的端口 |
| `MAX_FAILURES` | 否 | `1` | 密钥冷却前的连续失败次数 |
| `COOLING_PERIOD` | 否 | `300` | 失败密钥冷却的秒数 |
| `REQUEST_TIMEOUT` | 否 | `45` | Gemini API 请求超时秒数 |
| `MAX_RETRIES` | 否 | `0` | 失败请求的重试次数 |
| `PROXY_URL` | 否 | - | Gemini API 调用的 HTTP 代理 URL |

### API 密钥格式示例

```bash
# 多个 Gemini 密钥
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...

# 多个客户端密钥
ADAPTER_API_KEYS=client-key-123,client-key-456,client-key-abc

# 管理员密钥（与客户端密钥分离）
ADMIN_API_KEYS=admin-key-secure-1,admin-key-secure-2

# 使用代理
PROXY_URL=http://proxy.example.com:8080
```

## 🔧 配置您的客户端（Claude Code 示例）

要连接支持 Anthropic API 格式的 Claude Code 客户端，请按照以下步骤操作：

1.  **打开客户端设置**：导航到您的代码编辑器或客户端的设置面板。
2.  **查找 API 配置**：查找"Anthropic API 设置"或"Claude API 设置"部分。
3.  **设置 API 端点**：
    -   在"API Base URL"或"Endpoint"字段中，输入您的适配器 URL：
        `http://<your-vps-ip>:8000/v1`
4.  **设置 API 密钥**：
    -   在"API Key"字段中，输入您在 `ADAPTER_API_KEYS` 中定义的**客户端密钥**之一。
5.  **保存并测试**：保存设置并尝试聊天完成以确认工作正常。

## 📡 API 端点

### 主要端点（需要认证）

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/messages` | POST | **主要 Anthropic Messages API** - 完全兼容 |
| `/v1/messages/count_tokens` | POST | 发送前计算令牌数 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | OpenAI 兼容端点（遗留） |
| `/stats` | GET | 使用统计 |
| `/metrics` | GET | 详细性能指标 |

### 管理员端点（需要管理员密钥）

| 端点 | 方法 | 描述 |
|------|------|------|
| `/admin/reset-key/{prefix}` | POST | 重置失败的密钥 |
| `/admin/recover-key/{prefix}` | POST | 恢复永久失败的密钥 |
| `/admin/security-status` | GET | 安全配置状态 |

### 公开端点（无需认证）

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 基本健康检查 |
| `/health/detailed` | GET | 详细健康状态 |

#### 示例：使用 `curl` 检查统计信息

```bash
curl http://localhost:8000/stats \
  -H "Authorization: Bearer your-client-key-123"
```

#### 示例：使用 `curl` 重置密钥

```bash
curl -X POST http://localhost:8000/admin/reset-key/AIza \
  -H "Authorization: Bearer your-admin-key-abc"
```

## 🐛 故障排除

-   **"Invalid API key"**：确保您在客户端中使用的密钥列在 `.env` 文件的 `ADAPTER_API_KEYS`（或管理员端点的 `ADMIN_API_KEYS`）中。请记住在更改 `.env` 文件后重启服务（`docker-compose restart`）。
-   **"Service Unavailable" 或 502/503 错误**：这通常意味着您的所有 Gemini API 密钥都处于"冷却"状态。检查日志（`docker-compose logs -f`）以查看错误。您还可以检查 `/health` 端点获取状态，或使用 `/stats` 端点查看每个密钥的状态。
-   **连接被拒绝**：验证 Docker 容器是否正在运行（`docker-compose ps`）。检查您是否使用了正确的服务器 IP 地址和端口。如果在云提供商上运行，确保防火墙规则允许端口 8000 上的流量。

---

## 📁 项目结构

```
gemini-claude/
├── main.py                 # 开发入口点
├── src/
│   ├── main.py            # FastAPI 应用服务器
│   ├── anthropic_api.py   # Anthropic API 兼容层
│   ├── config.py          # 配置管理
│   ├── error_handling.py  # 增强错误处理
│   └── performance.py     # 性能优化
├── requirements.txt        # Python 依赖
├── .env.example           # 配置模板
├── docker-compose.yml     # Docker 部署配置
├── Dockerfile             # Docker 镜像配置
├── logs/                  # 应用日志（自动创建）
└── README.md              # 此文件
```

## 🔒 安全最佳实践

1. **使用强 API 密钥**：使用 `openssl rand -hex 32` 生成
2. **分离管理员密钥**：为 `SECURITY_ADMIN_API_KEYS` 设置不同的密钥
3. **保护您的服务器**：使用防火墙规则限制访问
4. **监控访问**：定期检查日志以发现未授权的尝试
5. **保持更新**：定期使用 `git pull && docker-compose up -d --build` 拉取更新

## 📊 高级配置

<details>
<summary>点击展开完整配置选项</summary>

```env
# =============================================
# 服务配置
# =============================================
SERVICE_ENVIRONMENT=production
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
SERVICE_WORKERS=1
SERVICE_LOG_LEVEL=INFO
SERVICE_ENABLE_METRICS=true
SERVICE_ENABLE_HEALTH_CHECK=true
SERVICE_CORS_ORIGINS=*

# =============================================
# Gemini API 配置
# =============================================
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...
GEMINI_MAX_FAILURES=3
GEMINI_COOLING_PERIOD=300
GEMINI_HEALTH_CHECK_INTERVAL=60
GEMINI_REQUEST_TIMEOUT=45
GEMINI_MAX_RETRIES=2
GEMINI_PROXY_URL=http://proxy.example.com:8080

# =============================================
# 安全配置
# =============================================
SECURITY_ADAPTER_API_KEYS=your-client-key-123,your-client-key-456
SECURITY_ADMIN_API_KEYS=your-admin-key-abc,your-admin-key-def
SECURITY_ENABLE_IP_BLOCKING=true
SECURITY_MAX_FAILED_ATTEMPTS=5
SECURITY_BLOCK_DURATION=300
SECURITY_ENABLE_RATE_LIMITING=true
SECURITY_RATE_LIMIT_REQUESTS=100
SECURITY_RATE_LIMIT_WINDOW=60

# =============================================
# 性能优化
# =============================================
CACHE_ENABLED=true
CACHE_MAX_SIZE=1000
CACHE_TTL=300
CACHE_KEY_PREFIX=gemini_adapter

PERF_MAX_KEEPALIVE_CONNECTIONS=20
PERF_MAX_CONNECTIONS=100
PERF_KEEPALIVE_EXPIRY=30.0
PERF_CONNECT_TIMEOUT=10.0
PERF_READ_TIMEOUT=45.0
PERF_WRITE_TIMEOUT=10.0
PERF_POOL_TIMEOUT=5.0
PERF_HTTP2_ENABLED=true

# =============================================
# 可选 Redis 配置
# =============================================
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password
```

</details>

---

**[🇨🇳 中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**