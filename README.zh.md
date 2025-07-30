# Gemini Claude Adapter

高性能的 Gemini Claude 适配器，专为 Claude Code 和本地客户端设计，支持多 API 密钥轮换、自动故障恢复和流式响应。

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

## ✨ 核心特性

-   🚀 **极速响应** - 优化的请求处理和密钥轮换算法
-   🔑 **智能密钥管理** - 失败密钥立即冷却，自动切换到下一个可用密钥
-   🛡️ **强大安全性** - 强制 API 密钥认证，支持客户端和管理员密钥
-   🌐 **完全兼容** - 兼容 Claude Code 和 OpenAI API 格式
-   ⚡ **流式支持** - 原生支持流式聊天响应
-   📊 **实时监控** - 详细的服务状态和密钥使用统计
-   🐳 **简化 Docker 部署** - 使用 Docker 和 Docker Compose 快速安全部署

## 🔒 安全优先：理解认证机制

此适配器强制 API 密钥认证来保护您的服务。有两个访问级别：

1.  **客户端密钥 (`ADAPTER_API_KEYS`)**：面向标准用户。这些密钥授予核心功能访问权限，如聊天完成 (`/v1/chat/completions`) 和列出模型 (`/v1/models`)。
2.  **管理员密钥 (`ADMIN_API_KEYS`)**：面向管理员。这些密钥授予所有端点的访问权限，包括受保护的管理端点，如重置 Gemini 密钥 (`/admin/reset-key/{prefix}`)。

如果未设置 `ADMIN_API_KEYS`，客户端密钥也将有管理员端点的访问权限。对于生产环境，**强烈建议**设置单独的管理员密钥。

认证通过 `X-API-Key` 头或 `Authorization: Bearer <token>` 头处理。

## 🚀 使用 Docker 部署（推荐）

使用 Docker 部署是最简单和安全的方法。

### 先决条件

-   Git
-   Docker
-   Docker Compose

### 步骤 1：克隆仓库

```bash
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude
```

### 步骤 2：配置环境变量

通过复制示例创建 `.env` 文件。此文件将存储您的所有秘密和配置。

```bash
cp .env.example .env
```

现在，使用文本编辑器（`nano .env` 或 `vim .env`）编辑 `.env` 文件并填写必需的值。

```env
# .env

# --- Gemini API 密钥 ---
# 在此处添加您的 Gemini API 密钥，用逗号分隔。
# 示例：GEMINI_API_KEYS=AIzaSyABC...,AIzaSyDEF...
GEMINI_API_KEYS=

# --- 适配器安全密钥 ---
# 生产环境必需。这些密钥供您的客户端（如 Claude Code）用于访问适配器。
# 使用 'openssl rand -hex 32' 生成强密钥
# 示例：ADAPTER_API_KEYS=client-key-123,client-key-456
ADAPTER_API_KEYS=

# 可选但推荐：用于管理员访问的单独密钥。
# 示例：ADMIN_API_KEYS=admin-key-abc,admin-key-def
ADMIN_API_KEYS=

# --- 网络配置 ---
# 服务在 Docker 容器内运行的主机和端口。
HOST=0.0.0.0
PORT=8000

# --- 密钥管理 ---
# Gemini 密钥被冷却前的连续失败次数。
MAX_FAILURES=1
# 失败的 Gemini 密钥的冷却时间（秒）。
COOLING_PERIOD=300
# 请求超时时间（秒）。
REQUEST_TIMEOUT=45

# --- 代理（可选）---
# 如果需要通过代理路由 Gemini API 流量，取消注释并设置 URL。
# PROXY_URL=http://your-proxy-url:port
```

**重要提示**：保护您的 `.env` 文件。它包含敏感密钥。

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

## 🔧 配置您的客户端（Claude Code 示例）

要连接支持 OpenAI API 格式的客户端（如 Claude Code），请按照以下步骤操作：

1.  **打开客户端设置**：导航到您的代码编辑器或客户端的设置面板。
2.  **查找 API 配置**：查找"OpenAI API 设置"或类似部分。
3.  **设置 API 端点**：
    -   在"API Base URL"或"Endpoint"字段中，输入您的适配器 URL：
        `http://<your-vps-ip>:8000/v1`
4.  **设置 API 密钥**：
    -   在"API Key"字段中，输入您在 `ADAPTER_API_KEYS` 中定义的**客户端密钥**之一。
5.  **保存并测试**：保存设置并尝试聊天完成以确认工作正常。

## 📡 API 端点

### 公开端点
*无需认证。*

-   `GET /`：返回基本服务信息。
-   `GET /health`：用于监控的健康检查端点。如果至少有一个 Gemini 密钥处于活动状态，则返回 `200 OK`。

### 受保护端点
*需要**客户端 API 密钥**（`X-API-Key` 或 `Bearer` 令牌）。*

-   `POST /v1/chat/completions`：聊天完成的主要端点。
-   `GET /v1/models`：列出适配器中配置的可用 Gemini 模型。
-   `GET /stats`：返回有关密钥使用情况、失败和状态的详细统计信息。

### 管理员端点
*需要**管理员 API 密钥**。*

-   `POST /admin/reset-key/{key_prefix}`：手动将失败或冷却的 Gemini 密钥重置回活动状态。`key_prefix` 必须至少为 4 个字符。
-   `GET /admin/security-status`：显示适配器的当前安全配置。

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
│   └── main.py            # FastAPI 应用程序（主服务器）
├── requirements.txt       # Python 依赖
├── .env.example          # 环境配置模板
├── docker-compose.yml    # Docker Compose 配置
├── Dockerfile            # Docker 镜像配置
├── logs/                 # 应用日志
├── README.md             # 主要项目文档（英文）
├── README.zh.md          # 中文文档
├── CLAUDE.md             # Claude Code 项目说明
└── security_guide.md     # 安全配置指南
```

---

**[🇨🇳 中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**