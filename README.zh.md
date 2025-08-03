# Gemini Claude Adapter v2.1.0

高性能的 Gemini Claude 适配器，专为 Claude Code 和本地客户端设计，支持多 API 密钥轮换、自动故障恢复和流式响应。

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

---

## ✨ 主要特性

-   [cite_start]🤖 **完全兼容 Anthropic API**: 完全支持 Anthropic Messages API (`/v1/messages`)，并提供正确的流式响应格式 [cite: 1, 2]。
-   [cite_start]🔑 **智能密钥管理**: 失效的 Gemini 密钥会立即进入冷却期，并自动切换到下一个可用密钥 [cite: 1, 2]。
-   [cite_start]🛡️ **强大的安全性**: 为所有敏感端点强制启用 API 密钥认证，分为客户端密钥和管理员密钥 [cite: 1, 2]。
-   [cite_start]⚡ **流式传输支持**: 原生支持 Anthropic 风格的流式响应，包含所有必需的事件类型 [cite: 1, 2]。
-   [cite_start]🛠️ **工具调用支持**: 完美兼容 Anthropic 与 Gemini 格式之间的工具（函数）调用转换 [cite: 1, 2]。
-   [cite_start]📊 **实时监控**: 提供服务健康状况、密钥状态和使用统计等监控端点 [cite: 1, 2]。
-   [cite_start]🐳 **简化的 Docker 部署**: 使用 Docker 和 Docker Compose 进行快速、安全的设置 [cite: 1, 2]。

---

## 🚀 快速上手指南

### 先决条件

-   系统已安装 **Docker** 和 **Docker Compose**。
-   拥有 **Google Gemini API 密钥** ([在此获取](https://makersuite.google.com/app/apikey))。
-   安装 **Git** 用于克隆代码仓库。

### 步骤 1: 克隆与配置

```bash
# 克隆代码仓库
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# 从示例文件创建你的配置文件
cp .env.example .env

# 编辑 .env 文件并填入你的密钥
# 可以使用你喜欢的编辑器，例如：nano .env
nano .env
````

你 **必须** 在 `.env` 文件中设置 `GEMINI_API_KEYS` 和 `SECURITY_ADAPTER_API_KEYS`。详情请参阅下方的配置章节。

### 步骤 2: 部署服务

```bash
# 构建并后台启动服务
docker-compose up -d --build

# 检查服务是否正常运行
docker-compose ps
```

服务将运行在 `http://localhost:80` (或你服务器的 IP 地址)。要实时查看日志，请运行 `docker-compose logs -f`。

### 步骤 3: 测试你的部署 🧪

项目提供了一个简单的测试脚本 `test_endpoints.sh` 来验证所有功能。

**1. 在脚本中设置你的密钥**
用编辑器打开测试脚本：

```bash
nano test_endpoints.sh
```

在脚本内部，将 `CLIENT_KEY` 和 `ADMIN_KEY` 的占位符值替换为您在 `.env` 文件中设置的真实密钥。

**2. 赋予脚本执行权限**
这个命令只需要运行一次。

```bash
chmod +x test_endpoints.sh
```

**3. 运行测试**

```bash
./test_endpoints.sh
```

脚本将对你的服务运行一系列测试并打印结果。如果所有测试都顺利通过，那么你的服务就已经准备就绪了！

-----

## ⚙️ 核心配置

请编辑你的 `.env` 文件，完成以下设置：

```env
# =============================================
# 必需: Gemini API 配置
# =============================================
# 从这里获取你的 API 密钥: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
# 你可以添加多个密钥，用逗号分隔
GEMINI_API_KEYS=your-google-ai-api-key-1,your-google-ai-api-key-2

# =============================================
# 必需: 安全配置
# =============================================
# 这是你的客户端应用（例如 Cursor）将用于认证的密钥
# 推荐使用以下命令生成一个强密钥: openssl rand -hex 32
SECURITY_ADAPTER_API_KEYS=your-client-key

# =============================================
# 推荐: 管理员访问权限
# =============================================
# 一个独立的、私有的密钥，用于访问管理端点
SECURITY_ADMIN_API_KEYS=your-admin-key
```

**⚠️ 重要安全提示:**

  - 请妥善保管你的 `.env` 文件，**绝不要** 将它提交到版本控制系统（如 Git）。
  - 为 `SECURITY_ADAPTER_API_KEYS` 使用强大且唯一的 API 密钥。
  - 强烈建议在生产环境中设置 `SECURITY_ADMIN_API_KEYS`。

-----

## 🔧 客户端配置 (以 Cursor IDE 为例)

要连接一个使用 Anthropic API 格式的客户端：

1.  **打开客户端设置**: 进入你的 IDE 或客户端的设置面板。
2.  **找到 API 配置**: 寻找 "Anthropic" 或 "Claude" API 相关的设置。
3.  **设置 API 端点**:
      - 在 "API Base URL" 或 "Endpoint" 字段中，填入你的适配器地址：
        `http://<your-server-ip>:80/v1`
4.  **设置 API 密钥**:
      - 在 "API Key" 字段中，填入你在 `SECURITY_ADAPTER_API_KEYS` 中定义的**客户端密钥**。
5.  **保存并测试**。

### 支持的模型

本适配器会自动将常见的 Anthropic 模型名称映射到 Gemini 模型：

  - `claude-3-5-sonnet` → `gemini-2.5-pro`
  - `claude-3-5-haiku` → `gemini-2.5-flash`
  - 以及其他 Claude 3 模型。

-----

## 📡 API 端点

### 主要端点 (需要客户端密钥)

| 端点 | 方法 | 描述 |
|---|---|---|
| `/v1/messages` | POST | **核心 Anthropic Messages API** |
| `/v1/models` | GET | 列出可用的模型 |
| `/stats` | GET | 查看密钥使用统计 |
| `/metrics` | GET | 获取详细的性能指标 |

### 管理端点 (需要管理员密钥)

| 端点 | 方法 | 描述 |
|---|---|---|
| `/admin/reset-key/{prefix}` | POST | 重置一个失效的 Gemini 密钥 |
| `/cache/clear` | POST | 清除响应缓存 |
| `/errors/recent` | GET | 查看近期的错误日志 |

### 公共端点

| 端点 | 方法 | 描述 |
|---|---|---|
| `/` | GET | 服务信息 |
| `/health` | GET | 基础健康检查 |

---

**[🇨🇳 中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**
