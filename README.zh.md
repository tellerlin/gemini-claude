# Gemini Claude Adapter v3.0.0

一款专为 Claude Code 及其他 Anthropic 客户端设计的高性能、安全的 Gemini 适配器，具有 **完整的 Anthropic API 兼容性**。其特性包括智能多 API 密钥轮换、自动故障转移、强大的安全性、流式传输支持以及高级优化功能。

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

---

## ✨ 主要特性

  - 🤖 **完全兼容 Anthropic API**: 完全支持 Anthropic Messages API (`/v1/messages`)，并提供正确的流式格式。
  - 🔑 **智能密钥管理**: 因临时问题（如速率限制）失败的 Gemini 密钥会立即进入冷却期，并自动切换到下一个可用密钥。
  - 🛡️ **强大的安全性**: 所有敏感端点强制使用客户端和管理员密钥进行 API 密钥认证。
  - ⚡ **流式传输支持**: 原生支持 Anthropic 风格的流式响应，包含所有必需的事件类型。
  - 🛠️ **工具调用支持**: 完全支持在 Anthropic 和 Gemini 格式之间进行工具/函数调用。
  - 📊 **实时监控**: 提供服务健康状况、密钥状态和使用统计的监控端点。
  - 🚀 **性能优化**: 包括响应缓存和可配置的超时/重试选项。
  - 🐳 **简化的 Docker 部署**: 使用 Docker 和 Docker Compose 进行快速、安全的设置。

## 🚀 快速入门指南

### 先决条件

  - 系统中已安装 **Docker** 和 **Docker Compose**
  - **Google Gemini API 密钥** ([在此获取](https://aistudio.google.com/app/apikey))
  - 用于克隆仓库的 **Git**

### 第 1 步：克隆并配置

```bash
# 克隆仓库
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# 从示例文件创建您的配置文件
cp .env.example .env

# 编辑 .env 文件并添加您的密钥
nano .env
```

您 **必须** 在 `.env` 文件中设置 `GEMINI_API_KEYS` 和 `SECURITY_ADAPTER_API_KEYS`。详情请参见下面的配置部分。

### 第 2 步：部署服务

```bash
# 在后台构建并启动服务
docker-compose up -d --build

# 检查服务是否正常运行
docker-compose ps

# 查看实时日志
docker-compose logs -f
```

服务将在 `http://localhost:8000` (或您服务器的 IP 和配置的端口) 上可用。

## 🔄 更新您的部署

### 对于新部署

请遵循上方的“快速入门指南”。

### 对于现有部署

要将您的服务更新到最新版本以获取新功能和错误修复：

```bash
# 导航到您的项目目录
cd gemini-claude

# 从仓库拉取最新的更改
git pull origin master

# 停止当前服务
docker-compose down

# 使用 --no-cache 重新构建以确保应用最新更改
docker-compose build --no-cache

# 启动更新后的服务
docker-compose up -d

# 查看最近的日志以确保启动成功
docker-compose logs --since 5m
```

## 🧪 测试您的部署

项目包含一个全面的测试脚本 (`test_endpoints.sh`) 来验证所有功能。

### 1\. 在脚本中设置您的密钥

在编辑器中打开测试脚本：

```bash
nano test_endpoints.sh
```

在脚本内部，将 `CLIENT_KEY` 和 `ADMIN_KEY` 的占位符值替换为您在 `.env` 文件中设置的实际密钥。

### 2\. 使脚本可执行

此命令只需运行一次：

```bash
chmod +x test_endpoints.sh
```

### 3\. 运行测试

```bash
# 首先确保您的适配器正在运行！
./test_endpoints.sh
```

脚本将对您的实时服务运行一系列测试并打印结果。如果所有测试都通过，您的服务就可以使用了！

## 🩺 故障排查与诊断

如果您遇到问题，这些诊断脚本可以帮助您识别问题。在您的项目根目录（包含 `docker-compose.yml` 的目录）中运行这些命令。

### 1\. 检查 API 密钥有效性

`api_key_checker.py` 脚本会测试您 `.env` 文件中的每一个 `GEMINI_API_KEYS`。它会验证密钥是否有效、是否有配额以及是否能访问必要的模型。然后，它会提议创建一个新的 `.env.updated` 文件，其中只包含有效的密钥。

```bash
docker-compose run --rm gemini-claude-adapter python api_key_checker.py
```

该脚本是交互式的，将指导您完成验证密钥和保存清理后列表的过程。

### 2\. 常规健康与导入检查

`diagnose_script.py` 会对您的设置进行常规健康检查。它会验证所有必需的项目文件是否存在，以及 `requirements.txt` 中的所有 Python 依赖项是否可以在 Docker 环境中正确导入。

```bash
docker-compose run --rm gemini-claude-adapter python diagnose_script.py
```

这有助于快速发现与安装损坏或文件丢失相关的问题。脚本将输出一个检查列表，用 `✓` 表示成功，`✗` 表示失败，帮助您精确定位问题。

## ⚙️ 必要配置

编辑您的 `.env` 文件并进行以下设置：

```dotenv
# =============================================
# 必需：Gemini API 配置
# =============================================
# 从这里获取您的密钥: https://aistudio.google.com/app/apikey
# 您可以添加多个密钥，用逗号分隔。不要使用引号。
GEMINI_API_KEYS=your-google-ai-api-key-1,your-google-ai-api-key-2

# =============================================
# 必需：安全配置
# =============================================
# 这是您的客户端应用程序将用于认证的密钥。
# 使用以下命令生成一个强密钥: openssl rand -hex 32
SECURITY_ADAPTER_API_KEYS=your-secure-client-key

# =============================================
# 推荐：管理员访问权限
# =============================================
# 一个独立的、用于访问管理端点的私有密钥。
SECURITY_ADMIN_API_KEYS=your-secure-admin-key

# =============================================
# 可选：性能与行为
# =============================================
# 因临时问题（如速率限制）失败的密钥在被重用前需要等待的秒数
GEMINI_COOLING_PERIOD=300
# 单次请求 Gemini API 的超时秒数
GEMINI_REQUEST_TIMEOUT=120
# 如果请求失败，使用不同密钥重试的次数
GEMINI_MAX_RETRIES=2
# 启用或禁用响应缓存
CACHE_ENABLED=true
# 缓存响应的存活时间（秒）
CACHE_TTL=300
```

**⚠️ 重要安全提示：**

  - 保护好您的 `.env` 文件，**永远不要** 将其提交到版本控制系统。
  - 为 `SECURITY_ADAPTER_API_KEYS` 使用强大且唯一的 API 密钥。
  - 强烈建议为生产环境设置 `SECURITY_ADMIN_API_KEYS`。

## 🔧 客户端配置

### 通用客户端设置 (例如 Cursor IDE)

要连接使用 Anthropic API 格式的客户端：

1.  **打开客户端设置**: 导航到您的 IDE 或客户端的设置界面。
2.  **查找 API 配置**: 寻找 "Anthropic" 或 "Claude" API 设置。
3.  **设置 API 端点**:
      - 在 "API Base URL" 或 "Endpoint" 字段中，输入您的适配器 URL：
        `http://<your-server-ip>:8000/v1`
        *(注意：如果您在 `.env` 或 `docker-compose.yml` 文件中自定义了端口，这里的端口可能会不同)*
4.  **设置 API 密钥**:
      - 在 "API Key" 字段中，输入您在 `SECURITY_ADAPTER_API_KEYS` 中定义的 **客户端密钥**。
5.  **保存并测试**

### Claude Code 配置

Claude Code 是 Anthropic 用于代理式编码的命令行工具。要将其配置为使用您的适配器：

#### 1\. 配置环境变量

编辑您的 shell 配置文件（根据您的 shell 选择一个）：

```bash
# 对于 bash 用户
nano ~/.bash_profile

# 对于 zsh 用户 (macOS 默认)
nano ~/.zshrc

# 对于其他 shell，编辑相应的配置文件
```

#### 2\. 添加以下行

将这些环境变量添加到您的配置文件中：

```bash
# Gemini Claude Adapter 配置
export ANTHROPIC_BASE_URL="http://your-server-ip:8000/v1"
export ANTHROPIC_AUTH_TOKEN="your-secure-client-key"
```

替换：

  - `your-server-ip` 为您服务器的 IP 地址或域名。
  - `your-secure-client-key` 为您在 `SECURITY_ADAPTER_API_KEYS` 中设置的密钥。

#### 3\. 应用更改

```bash
# 重新加载您的 shell 配置文件
source ~/.bash_profile  # 或对于 zsh 用户使用 ~/.zshrc

# 验证配置
echo $ANTHROPIC_BASE_URL
echo $ANTHROPIC_AUTH_TOKEN
```

#### 4\. 测试 Claude Code

```bash
# 使用您的适配器测试 Claude Code
claude-code --help

# 运行一个简单的测试
claude-code "创建一个 hello world Python 脚本"
```

有关 Claude Code 的更多信息，请查看[官方文档](https://docs.anthropic.com/en/docs/claude-code)。

### 支持的模型

适配器会自动将常见的 Anthropic 模型名称映射到一个兼容的 Gemini 模型。根据当前代码，以下模型会被映射：

  - `claude-3-5-sonnet` → `gemini-2.0-flash-exp`
  - `claude-3-opus` → `gemini-2.0-flash-exp`
  - `claude-3-haiku` → `gemini-2.0-flash-exp`
  - 任何其他 Anthropic 模型名称也将默认映射到 `gemini-2.0-flash-exp`。

## 📡 API 端点

### 主要端点 (需要客户端密钥)

| 端点 | 方法 | 描述 |
|:---|:---|:---|
| `/v1/messages` | `POST` | **主要的 Anthropic Messages API** |
| `/v1/models` | `GET` | 列出可用的（已映射的）模型 |
| `/stats` | `GET` | 查看密钥使用和性能统计 |
| `/metrics` | `GET` | 详细的性能指标 |

### 管理员端点 (需要管理员密钥)

| 端点 | 方法 | 描述 |
|:---|:---|:---|
| `/admin/reset-key/{prefix}` | `POST` | 通过前缀重置一个失败/冷却中的 Gemini 密钥 |

### 公共端点

| 端点 | 方法 | 描述 |
|:---|:---|:---|
| `/` | `GET` | 服务信息 |
| `/health` | `GET` | 基本健康检查，指示是否有可用的活动密钥 |

## 🛠️ 开发与部署

### 本地开发

```bash
# 从 requirements.txt 安装依赖
pip install -r requirements.txt

# 创建并配置您的 .env 文件
cp .env.example .env
# 编辑 .env 文件进行配置

# 使用 Uvicorn 运行应用
# 应用将在 http://localhost:8000 上运行
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker 部署

```bash
# 在本地构建 Docker 镜像
docker build -t gemini-claude-adapter .

# 使用您的 .env 文件运行容器
# 这会将容器的 8000 端口映射到您主机的 8000 端口
docker run -d \
  --name gemini-claude-adapter \
  -p 8000:8000 \
  --env-file .env \
  gemini-claude-adapter
```

## 📈 监控与管理

### 健康监控

检查服务健康状况：

```bash
curl http://your-server-ip:8000/health
```

### 密钥统计

查看当前密钥使用情况和统计数据：

```bash
curl -H "X-API-Key: your-client-key" \
     http://your-server-ip:8000/stats
```

### 性能指标

获取详细的性能指标：

```bash
curl -H "X-API-Key: your-client-key" \
     http://your-server-ip:8000/metrics
```

### 管理操作

重置一个失败的密钥 (需要管理员密钥)：

```bash
curl -X POST \
     -H "X-API-Key: your-admin-key" \
     http://your-server-ip:8000/admin/reset-key/AIzaSy
```

## 🔒 安全最佳实践

1.  **使用强 API 密钥**: 使用 `openssl rand -hex 32` 生成加密安全的密钥。
2.  **分离客户端和管理员密钥**: 为客户端访问和管理操作使用不同的密钥。
3.  **网络安全**: 考虑在反向代理（如 nginx, Cloudflare 等）后面运行服务。
4.  **速率限制**: 内置的速率限制有助于防止滥用。
5.  **监控**: 定期检查日志和指标以发现异常活动。
6.  **更新**: 保持适配器更新，以获取最新的安全补丁。

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。对于重大更改，请先开启一个 issue 来讨论您想要更改的内容。

## 📄 许可证

本项目根据 MIT 许可证授权 - 详情请见 LICENSE 文件。

## 🆘 支持

如果您遇到任何问题：

1.  检查上面的“故障排查”部分。
2.  运行诊断脚本。
3.  检查 Docker 日志: `docker-compose logs -f`。
4.  在 GitHub 上开启一个 issue，并提供您问题的详细信息。

## 📊 性能提示

  - **多个 API 密钥**: 使用多个 Gemini API 密钥以获得更好的吞吐量和可靠性。
  - **缓存**: 为重复的请求启用响应缓存。
  - **监控**: 定期检查 `/stats` 和 `/health` 端点。
  - **资源分配**: 确保为您的预期负载分配足够的系统资源。

---

**[🇨🇳 中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**
