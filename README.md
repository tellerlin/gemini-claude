# Gemini Claude Adapter v2.1.0

A high-performance, secure Gemini adapter with **complete Anthropic API compatibility**, designed for Claude Code and other Anthropic clients. Features intelligent multi-API key rotation, automatic failover, robust security, streaming support, and advanced optimizations.

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

## ✨ Key Features

-   🤖 **Full Anthropic API Compatibility**: Complete support for Anthropic Messages API (`/v1/messages`) with proper streaming format
-   🔑 **Smart Key Management**: Failed Gemini keys are immediately placed in a cool-down period, with automatic failover to the next available key
-   🛡️ **Robust Security**: Enforced API key authentication for all sensitive endpoints using client and admin keys
-   🌐 **Dual API Support**: Compatible with both Anthropic and OpenAI API formats for maximum flexibility
-   ⚡ **Streaming Support**: Native support for Anthropic-style streaming responses with all required event types
-   🛠️ **Tool Calling Support**: Complete tool/function calling support between Anthropic and Gemini formats
-   📊 **Real-time Monitoring**: Endpoints for service health, key status, and usage statistics
-   🐳 **Simplified Docker Deployment**: Quick and secure setup using Docker and Docker Compose

### 🚀 New in v2.1.0

-   **Performance Optimizations**: Intelligent response caching and HTTP connection pooling
-   **Enhanced Error Handling**: Smart error classification and circuit breaker patterns
-   **Advanced Monitoring**: Comprehensive metrics collection and performance tracking
-   **Structured Configuration**: Hierarchical configuration system with environment support
-   **Improved Reliability**: Better fault tolerance and automatic recovery mechanisms

## 🔒 Security First: Understanding Authentication

This adapter enforces API key authentication to protect your service. There are two levels of access:

1.  **Client Keys (`ADAPTER_API_KEYS`)**: For standard users. These keys grant access to core functionalities like Anthropic Messages API (`/v1/messages`), token counting (`/v1/messages/count_tokens`), and listing models (`/v1/models`).
2.  **Admin Keys (`ADMIN_API_KEYS`)**: For administrators. These keys grant access to all endpoints, including protected management endpoints like resetting a Gemini key (`/admin/reset-key/{prefix}`).

If admin keys are not set, the client keys will also have access to admin endpoints. For production environments, it is **highly recommended** to set separate admin keys.

Authentication is handled via the `X-API-Key` header or an `Authorization: Bearer <token>` header.

### Enhanced Security Features (v2.1.0)

- **IP Blocking**: Automatic IP blocking after repeated failed attempts
- **Rate Limiting**: Configurable rate limiting per client key
- **Circuit Breaker**: Prevents cascading failures during service issues
- **Smart Error Classification**: Intelligent error handling and logging

## 🚀 Quick Start Guide

### Prerequisites

-   **Docker** and **Docker Compose** installed on your system
-   **Google Gemini API Keys** ([Get yours here](https://makersuite.google.com/app/apikey))
-   **Git** for cloning the repository

### Step 1: Get Your API Keys

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create one or more API keys
4. Copy the keys (they start with `AIza...`)

### Step 2: Deploy the Service

```bash
# Clone the repository
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# Copy and configure environment variables
cp .env.example .env

# Edit the configuration (see configuration section below)
nano .env  # or use your preferred editor

# Start the service
docker-compose up -d

# Check if it's running
docker-compose ps
docker-compose logs -f
```

The service will be available at `http://localhost:80` (or your server's IP).

### Step 3: Test Your Deployment

```bash
# Basic health check (no authentication required)
curl http://localhost:80/health

# Test with your API key
curl http://localhost:80/v1/models \
  -H "Authorization: Bearer your-client-key"
```

### Step 4: Test Q&A Functionality

Test the complete Q&A functionality using the Anthropic Messages API:

```bash
# Simple Q&A test
curl -X POST http://localhost:80/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-client-key" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user", 
        "content": "Hello! Can you explain what is machine learning in simple terms?"
      }
    ]
  }'

# Multi-turn conversation test
curl -X POST http://localhost:80/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-client-key" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user", 
        "content": "What are the main benefits of Python programming?"
      },
      {
        "role": "assistant",
        "content": "Python offers several key benefits:\n\n1. **Simple and readable syntax** - Easy to learn and understand\n2. **Versatile** - Used for web development, data science, AI, automation\n3. **Large ecosystem** - Extensive libraries and frameworks\n4. **Cross-platform** - Runs on Windows, macOS, and Linux\n5. **Strong community support** - Active developer community and resources"
      },
      {
        "role": "user",
        "content": "Can you give me a simple example of Python code?"
      }
    ]
  }'

# Test with different models
curl -X POST http://localhost:80/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-client-key" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-haiku",
    "max_tokens": 500,
    "messages": [
      {
        "role": "user", 
        "content": "Write a haiku about programming"
      }
    ]
  }'
```

## ⚙️ Essential Configuration

Edit your `.env` file with these **required** settings:

```env
# =============================================
# REQUIRED: Gemini API Configuration
# =============================================
# Get your API keys from: https://makersuite.google.com/app/apikey
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...

# =============================================
# REQUIRED: Security Configuration
# =============================================
# Generate strong keys: openssl rand -hex 32
ADAPTER_API_KEYS=your-client-key

# =============================================
# OPTIONAL: Admin Access
# =============================================
# Optional admin keys for management endpoints
ADMIN_API_KEYS=your-admin-key

# =============================================
# OPTIONAL: Service Configuration
# =============================================
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
SERVICE_LOG_LEVEL=INFO
```

**⚠️ Important Security Notes:**
- Keep your `.env` file secure and never commit it to version control
- Use strong, unique API keys for `ADAPTER_API_KEYS`
- Set `ADMIN_API_KEYS` for production environments
- Generate secure keys with: `openssl rand -hex 32`

### Step 3: Launch the Service

With your `.env` file configured, start the service using Docker Compose.

```bash
docker-compose up -d
```

The service will now be running in the background. The API will be accessible at `http://localhost:80` (or your server's IP address).

### Step 4: Manage the Service

Here are the essential Docker Compose commands for managing your service:

-   **Check Logs**: `docker-compose logs -f`
-   **Stop Service**: `docker-compose down`
-   **Restart Service**: `docker-compose restart`

The project includes optimized Docker configuration files (`docker-compose.yml` and `Dockerfile`) that handle:
- Production-ready server setup with gunicorn + uvicorn
- Security best practices (non-privileged user execution)
- Log persistence through volume mounting
- Automatic restart policies for high availability

## 🔄 Updating the Project

When new updates are released for the Gemini Claude Adapter, follow these steps to update your deployment:

### Method 1: Git Pull (Recommended)

This method preserves your custom configurations while updating the application code:

```bash
# Navigate to your project directory
cd gemini-claude

# Stop the running service
docker-compose down

# Pull the latest changes
git pull origin main

# Rebuild and restart the service
docker-compose up -d --build

# Check the service status
docker-compose ps
docker-compose logs -f
```

### Method 2: Manual Update

If you have made custom modifications to the code:

```bash
# Stop the service
docker-compose down

# Backup your .env file (important!)
cp .env .env.backup

# Remove the old project directory (optional)
rm -rf gemini-claude

# Clone the latest version
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# Restore your configuration
cp ../.env.backup .env

# Start the service
docker-compose up -d
```

### Important Notes

- **Configuration Preservation**: Your `.env` file contains your API keys and settings. Always back it up before updating.
- **Database/Logs**: The `logs/` directory is mounted as a volume, so your logs will be preserved.
- **Docker Images**: The `--build` flag ensures Docker uses the latest code to rebuild the image.
- **Breaking Changes**: Check the project's release notes or commit history for any breaking changes that might require configuration updates.

### What Gets Updated

- Application code and features
- Security patches and improvements
- Docker configuration
- Dependencies and requirements

### What Gets Preserved

- Your `.env` configuration file
- Application logs in `logs/` directory
- Docker volumes and data
- Your custom settings

## ⚙️ Complete Configuration Reference

### Environment Variables (v2.1.0)

#### Service Configuration
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SERVICE_ENVIRONMENT` | No | `development` | Runtime environment (development/production) |
| `SERVICE_HOST` | No | `0.0.0.0` | Host to bind to |
| `SERVICE_PORT` | No | `8000` | Port to bind to |
| `SERVICE_WORKERS` | No | `1` | Number of worker processes |
| `SERVICE_LOG_LEVEL` | No | `INFO` | Logging level |
| `SERVICE_ENABLE_METRICS` | No | `true` | Enable metrics collection |
| `SERVICE_ENABLE_HEALTH_CHECK` | No | `true` | Enable health check endpoint |

#### Gemini API Configuration
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEYS` | Yes | - | Comma-separated Google Gemini API keys |
| `GEMINI_MAX_FAILURES` | No | `3` | Consecutive failures before key cooling |
| `GEMINI_COOLING_PERIOD` | No | `300` | Seconds to keep failed key in cooling |
| `GEMINI_HEALTH_CHECK_INTERVAL` | No | `60` | Health check interval in seconds |
| `GEMINI_REQUEST_TIMEOUT` | No | `45` | Timeout for Gemini API requests |
| `GEMINI_MAX_RETRIES` | No | `2` | Number of retry attempts |
| `GEMINI_PROXY_URL` | No | - | HTTP proxy URL for API calls |

#### Security Configuration
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ADAPTER_API_KEYS` | Yes | - | Client authentication keys |
| `ADMIN_API_KEYS` | No | - | Admin authentication keys |
| `SECURITY_ENABLE_IP_BLOCKING` | No | `true` | Enable IP blocking |
| `SECURITY_MAX_FAILED_ATTEMPTS` | No | `5` | Max failed attempts before IP block |
| `SECURITY_BLOCK_DURATION` | No | `300` | IP block duration in seconds |
| `SECURITY_ENABLE_RATE_LIMITING` | No | `true` | Enable rate limiting |
| `SECURITY_RATE_LIMIT_REQUESTS` | No | `100` | Rate limit requests per window |
| `SECURITY_RATE_LIMIT_WINDOW` | No | `60` | Rate limit window in seconds |

#### Performance Configuration
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CACHE_ENABLED` | No | `true` | Enable response caching |
| `CACHE_MAX_SIZE` | No | `1000` | Maximum cache size |
| `CACHE_TTL` | No | `300` | Cache TTL in seconds |
| `CACHE_KEY_PREFIX` | No | `gemini_adapter` | Cache key prefix |
| `PERF_MAX_KEEPALIVE_CONNECTIONS` | No | `20` | Max keepalive connections |
| `PERF_MAX_CONNECTIONS` | No | `100` | Max total connections |
| `PERF_KEEPALIVE_EXPIRY` | No | `30.0` | Keepalive expiry time |
| `PERF_CONNECT_TIMEOUT` | No | `10.0` | Connection timeout |
| `PERF_READ_TIMEOUT` | No | `45.0` | Read timeout |
| `PERF_WRITE_TIMEOUT` | No | `10.0` | Write timeout |
| `PERF_POOL_TIMEOUT` | No | `5.0` | Pool timeout |
| `PERF_HTTP2_ENABLED` | No | `true` | Enable HTTP/2 |

#### Optional Database Configuration
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | No | - | Redis URL for distributed caching |
| `REDIS_PASSWORD` | No | - | Redis password |
| `REDIS_DB` | No | `0` | Redis database number |
| `REDIS_MAX_CONNECTIONS` | No | `10` | Max Redis connections |

### Configuration Examples

```bash
# Multiple Gemini API keys
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...

# Client authentication keys
ADAPTER_API_KEYS=your-client-key

# Admin authentication keys (optional)
ADMIN_API_KEYS=your-admin-key

# Performance optimization settings
CACHE_ENABLED=true
PERF_HTTP2_ENABLED=true
SERVICE_ENABLE_METRICS=true
```

## 🔧 Configuring Your Client (Claude Code Example)

To connect Claude Code which uses the Anthropic API format, follow these steps:

1.  **Open Client Settings**: Navigate to the settings panel of your code editor or client.
2.  **Find API Configuration**: Look for "Anthropic API Settings" or "Claude API Settings" section.
3.  **Set the API Endpoint**:
    -   In the "API Base URL" or "Endpoint" field, enter the URL of your adapter:
        `http://<your-vps-ip>:80/v1`
4.  **Set the API Key**:
    -   In the "API Key" field, enter the **client key** you defined in `ADAPTER_API_KEYS` (e.g., `your-client-key`).
5.  **Save and Test**: Save the settings and try a chat completion to confirm it's working.

### Supported Models

The adapter maps Anthropic model names to the latest Gemini models:

- `claude-3-5-sonnet` → `gemini-2.5-pro`
- `claude-3-5-haiku` → `gemini-2.5-flash`
- `claude-3-opus` → `gemini-2.5-pro`
- `claude-3-sonnet` → `gemini-2.5-pro`
- `claude-3-haiku` → `gemini-2.5-flash`

The adapter uses the latest Gemini 2.5 models for optimal performance and capabilities.

## 📡 API Endpoints

### Main Endpoints (Require Authentication)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | **Primary Anthropic Messages API** - Full compatibility |
| `/v1/messages/count_tokens` | POST | Count tokens before sending |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI-compatible endpoint (legacy) |
| `/stats` | GET | Usage statistics |
| `/metrics` | GET | Detailed performance metrics |

### Admin Endpoints (Require Admin Keys)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/reset-key/{prefix}` | POST | Reset a failed key |
| `/admin/recover-key/{prefix}` | POST | Recover a permanently failed key |
| `/admin/security-status` | GET | Security configuration status |

### Public Endpoints (No Authentication)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Detailed health status |

### Usage Examples

#### Check Service Health
```bash
curl http://localhost:80/health \
  -H "Authorization: Bearer your-client-key"
```

#### View Performance Metrics
```bash
curl http://localhost:80/metrics \
  -H "Authorization: Bearer your-client-key"
```

#### Check Cache Performance
```bash
curl http://localhost:80/cache/stats \
  -H "Authorization: Bearer your-client-key"
```

#### Reset a Failed Key
```bash
curl -X POST http://localhost:80/admin/reset-key/AIza \
  -H "Authorization: Bearer your-admin-key"
```

#### Clear Cache
```bash
curl -X POST http://localhost:80/cache/clear \
  -H "Authorization: Bearer your-admin-key"
```

## 🐛 Troubleshooting

### Common Issues

-   **"Invalid API key"**: Ensure the key you are using is listed in `ADAPTER_API_KEYS` (or `ADMIN_API_KEYS` for admin endpoints) in your `.env` file. Restart the service after configuration changes.

-   **"Service Unavailable" or 502/503 Errors**: Usually indicates all Gemini API keys are in a "cooling" state. Check logs (`docker-compose logs -f`) for error details. Use the `/health/detailed` endpoint for comprehensive status.

-   **Connection Refused**: Verify Docker container is running (`docker-compose ps`). Check IP address, port, and firewall rules. Ensure port 80 is accessible.

### Performance Issues

-   **High Response Times**: Check `/metrics` endpoint for performance statistics. Consider increasing `PERF_MAX_CONNECTIONS` or enabling caching if disabled.

-   **Cache Not Working**: Verify `CACHE_ENABLED=true` in configuration. Check `/cache/stats` for cache hit rates.

-   **Memory Usage**: Monitor `CACHE_MAX_SIZE` setting. Reduce cache size if memory usage is high.

### Advanced Diagnostics

#### Check Recent Errors
```bash
curl http://localhost:80/errors/recent \
  -H "Authorization: Bearer your-admin-key"
```

#### Monitor Real-time Logs
```bash
docker-compose logs -f --tail=100
```

### Configuration Validation

After making changes to `.env`, always restart the service:
```bash
docker-compose down
docker-compose up -d
```

Verify configuration is loaded correctly by checking service info:
```bash
curl http://localhost:80/
```

---

## 📁 Project Structure

```
gemini-claude/
├── main.py                 # Development entry point
├── src/
│   ├── main.py            # FastAPI application server
│   ├── anthropic_api.py   # Anthropic API compatibility layer
│   ├── config.py          # Configuration management
│   ├── error_handling.py  # Enhanced error handling
│   └── performance.py     # Performance optimizations
├── requirements.txt        # Python dependencies
├── .env.example           # Configuration template
├── docker-compose.yml     # Docker deployment config
├── Dockerfile             # Docker image config
├── logs/                  # Application logs (auto-created)
└── README.md              # This file
```

### Key Modules (v2.1.0)

- **`config.py`**: Hierarchical configuration system with environment variable support
- **`error_handling.py`**: Smart error classification, circuit breaker patterns, and monitoring
- **`performance.py`**: Response caching, connection pooling, and performance metrics

## 🔒 Security Best Practices

1. **Use Strong API Keys**: Generate with `openssl rand -hex 32`
2. **Separate Admin Keys**: Set different keys for `ADMIN_API_KEYS`
3. **Secure Your Server**: Use firewall rules to restrict access
4. **Monitor Access**: Check logs regularly for unauthorized attempts
5. **Keep Updated**: Pull updates regularly with `git pull && docker-compose up -d --build`

## 📊 Advanced Configuration

<details>
<summary>Click to expand full configuration options</summary>

```env
# =============================================
# Service Configuration
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
# Gemini API Configuration
# =============================================
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...
GEMINI_MAX_FAILURES=3
GEMINI_COOLING_PERIOD=300
GEMINI_HEALTH_CHECK_INTERVAL=60
GEMINI_REQUEST_TIMEOUT=45
GEMINI_MAX_RETRIES=2
GEMINI_PROXY_URL=http://proxy.example.com:8080

# =============================================
# Security Configuration
# =============================================
ADAPTER_API_KEYS=your-client-key
ADMIN_API_KEYS=your-admin-key
SECURITY_ENABLE_IP_BLOCKING=true
SECURITY_MAX_FAILED_ATTEMPTS=5
SECURITY_BLOCK_DURATION=300
SECURITY_ENABLE_RATE_LIMITING=true
SECURITY_RATE_LIMIT_REQUESTS=100
SECURITY_RATE_LIMIT_WINDOW=60

# =============================================
# Performance Optimization
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
# Optional Redis Configuration
# =============================================
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password
```

</details>

---

**[🇨🇳 切换到中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**