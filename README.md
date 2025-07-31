# Gemini Claude Adapter v2.0.0

A high-performance, secure Gemini adapter with **complete Anthropic API compatibility**, designed for Claude Code and other Anthropic clients. Features intelligent multi-API key rotation, automatic failover, robust security, and streaming support.

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

## ✨ Key Features

-   🤖 **Full Anthropic API Compatibility**: Complete support for Anthropic Messages API (`/v1/messages`) with proper streaming format
-   🔑 **Smart Key Management**: Failed Gemini keys are immediately placed in a cool-down period, with automatic failover to the next available key.
-   🛡️ **Robust Security**: Enforced API key authentication for all sensitive endpoints using client and admin keys.
-   🌐 **Dual API Support**: Compatible with both Anthropic and OpenAI API formats for maximum flexibility.
-   ⚡ **Streaming Support**: Native support for Anthropic-style streaming responses with all required event types.
-   🛠️ **Tool Calling Support**: Complete tool/function calling support between Anthropic and Gemini formats.
-   📊 **Real-time Monitoring**: Endpoints for service health, key status, and usage statistics.
-   🐳 **Simplified Docker Deployment**: Quick and secure setup using Docker and Docker Compose.

## 🔒 Security First: Understanding Authentication

This adapter enforces API key authentication to protect your service. There are two levels of access:

1.  **Client Keys (`ADAPTER_API_KEYS`)**: For standard users. These keys grant access to core functionalities like Anthropic Messages API (`/v1/messages`), token counting (`/v1/messages/count_tokens`), and listing models (`/v1/models`).
2.  **Admin Keys (`ADMIN_API_KEYS`)**: For administrators. These keys grant access to all endpoints, including protected management endpoints like resetting a Gemini key (`/admin/reset-key/{prefix}`).

If `ADMIN_API_KEYS` are not set, the client keys will also have access to admin endpoints. For production environments, it is **highly recommended** to set separate admin keys.

Authentication is handled via the `X-API-Key` header or an `Authorization: Bearer <token>` header.

## 🚀 Deployment with Docker (Recommended)

Deploying with Docker is the simplest and most secure method.

### Prerequisites

-   Git
-   Docker
-   Docker Compose

### Step 1: Clone the Repository

```bash
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude
```

### Step 2: Configure Environment Variables

Create a `.env` file by copying the example. This file will store all your secrets and configurations.

```bash
cp .env.example .env
```

Now, edit the `.env` file with a text editor (`nano .env` or `vim .env`) and fill in the required values.

```env
# .env

# --- Gemini API Keys ---
# Add your Gemini API keys here, separated by commas.
# Example: GEMINI_API_KEYS=AIzaSyABC...,AIzaSyDEF...
GEMINI_API_KEYS=

# --- Adapter Security Keys ---
# Required for production. These keys are used by your clients (e.g., Claude Code) to access the adapter.
# Generate strong keys using 'openssl rand -hex 32'
# Example: ADAPTER_API_KEYS=client-key-123,client-key-456
ADAPTER_API_KEYS=

# Optional but Recommended: Separate keys for admin access.
# Example: ADMIN_API_KEYS=admin-key-abc,admin-key-def
ADMIN_API_KEYS=

# --- Network Configuration ---
# The host and port the service will run on inside the Docker container.
HOST=0.0.0.0
PORT=8000

# --- Key Management ---
# Number of consecutive failures before a Gemini key is put into cooling.
MAX_FAILURES=1
# Cooldown period in seconds for a failed Gemini key.
COOLING_PERIOD=300
# Request timeout in seconds.
REQUEST_TIMEOUT=45

# --- Proxy (Optional) ---
# If you need to route Gemini API traffic through a proxy, uncomment and set the URL.
# PROXY_URL=http://your-proxy-url:port
```

**Important**: Protect your `.env` file. It contains sensitive keys.

### Step 3: Launch the Service

With your `.env` file configured, start the service using Docker Compose.

```bash
docker-compose up -d
```

The service will now be running in the background. The API will be accessible at `http://localhost:8000` (or your server's IP address).

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

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEYS` | Yes | - | Comma-separated Google Gemini API keys |
| `ADAPTER_API_KEYS` | Yes | - | Client authentication keys (generate with `openssl rand -hex 32`) |
| `ADMIN_API_KEYS` | No | - | Admin authentication keys (recommended for production) |
| `HOST` | No | `0.0.0.0` | Host to bind to inside container |
| `PORT` | No | `8000` | Port to bind to inside container |
| `MAX_FAILURES` | No | `1` | Consecutive failures before key cooling |
| `COOLING_PERIOD` | No | `300` | Seconds to keep failed key in cooling |
| `REQUEST_TIMEOUT` | No | `45` | Timeout for Gemini API requests in seconds |
| `MAX_RETRIES` | No | `0` | Number of retry attempts for failed requests |
| `PROXY_URL` | No | - | HTTP proxy URL for Gemini API calls |

### API Key Format Examples

```bash
# Multiple Gemini keys
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...

# Multiple client keys
ADAPTER_API_KEYS=client-key-123,client-key-456,client-key-abc

# Admin keys (separate from client keys)
ADMIN_API_KEYS=admin-key-secure-1,admin-key-secure-2

# With proxy
PROXY_URL=http://proxy.example.com:8080
```

## 🔧 Configuring Your Client (Claude Code Example)

To connect Claude Code which uses the Anthropic API format, follow these steps:

1.  **Open Client Settings**: Navigate to the settings panel of your code editor or client.
2.  **Find API Configuration**: Look for "Anthropic API Settings" or "Claude API Settings" section.
3.  **Set the API Endpoint**:
    -   In the "API Base URL" or "Endpoint" field, enter the URL of your adapter:
        `http://<your-vps-ip>:8000/v1`
4.  **Set the API Key**:
    -   In the "API Key" field, enter one of the **client keys** you defined in `ADAPTER_API_KEYS`.
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

### Public Endpoints
*No authentication required.*

-   `GET /`: Returns basic service information.
-   `GET /health`: A health check endpoint for monitoring. Returns `200 OK` if at least one Gemini key is active.

### Protected Endpoints
*Requires a **Client API Key** (`X-API-Key` or `Bearer` token).*

#### Anthropic Messages API (Primary)
-   `POST /v1/messages`: **Primary endpoint** for Anthropic Messages API with complete compatibility.
-   `POST /v1/messages/count_tokens`: Count tokens for messages before sending.
-   `GET /v1/models`: Lists available Anthropic models (mapped to Gemini models).

#### Legacy OpenAI-Compatible API (Backward Compatibility)
-   `POST /v1/chat/completions`: Legacy OpenAI-compatible endpoint for backward compatibility.
-   `GET /stats`: Returns detailed statistics about key usage, failures, and status.

### Admin Endpoints
*Requires an **Admin API Key**.*

-   `POST /admin/reset-key/{key_prefix}`: Manually resets a failed or cooling Gemini key back to active status. `key_prefix` must be at least 4 characters.
-   `GET /admin/security-status`: Shows the current security configuration of the adapter.

#### Example: Checking Stats with `curl`

```bash
curl http://localhost:8000/stats \
  -H "Authorization: Bearer your-client-key-123"
```

#### Example: Resetting a Key with `curl`

```bash
curl -X POST http://localhost:8000/admin/reset-key/AIza \
  -H "Authorization: Bearer your-admin-key-abc"
```

## 🐛 Troubleshooting

-   **"Invalid API key"**: Ensure the key you are using in your client is listed in `ADAPTER_API_KEYS` (or `ADMIN_API_KEYS` for admin endpoints) in your `.env` file. Remember to restart the service (`docker-compose restart`) after changing the `.env` file.
-   **"Service Unavailable" or 502/503 Errors**: This usually means all your Gemini API keys are in a "cooling" state. Check the logs (`docker-compose logs -f`) to see the errors. You can also check the `/health` endpoint for status or use the `/stats` endpoint to see the state of each key.
-   **Connection Refused**: Verify that the Docker container is running (`docker-compose ps`). Check that you are using the correct IP address and port for your server. If running on a cloud provider, ensure the firewall rules allow traffic on port 8000.

---

## 📁 Project Structure

```
gemini-claude/
├── main.py                 # Development entry point
├── src/
│   ├── main.py            # FastAPI application (main server)
│   └── anthropic_api.py   # Anthropic API compatibility layer
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Docker image configuration
├── logs/                 # Application logs
├── README.md             # Main project documentation (English)
├── README.zh.md          # Chinese documentation
├── CLAUDE.md             # Project instructions for Claude Code
└── security_guide.md     # Security configuration guide
```

---

**[🇨🇳 切换到中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**