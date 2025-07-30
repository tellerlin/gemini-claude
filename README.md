# Gemini Claude Adapter

A high-performance, secure Gemini adapter designed for clients like Claude Code, featuring multi-API key rotation, automatic failover, robust security, and streaming support.

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

## ✨ Key Features

-   🚀 **Ultra-fast Response**: Optimized request handling and smart key rotation.
-   🔑 **Smart Key Management**: Failed Gemini keys are immediately placed in a cool-down period, with automatic failover to the next available key.
-   🛡️ **Robust Security**: Enforced API key authentication for all sensitive endpoints using client and admin keys.
-   🌐 **Full Compatibility**: Compatible with clients that use the OpenAI API format, including Claude Code.
-   ⚡ **Streaming Support**: Native support for streaming chat responses.
-   📊 **Real-time Monitoring**: Endpoints for service health, key status, and usage statistics.
-   🐳 **Simplified Docker Deployment**: Quick and secure setup using Docker and Docker Compose.

## 🔒 Security First: Understanding Authentication

This adapter enforces API key authentication to protect your service. There are two levels of access:

1.  **Client Keys (`ADAPTER_API_KEYS`)**: For standard users. These keys grant access to core functionalities like chat completions (`/v1/chat/completions`) and listing models (`/v1/models`).
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

## 🔧 Configuring Your Client (Claude Code Example)

To connect a client like Claude Code that supports the OpenAI API format, follow these steps:

1.  **Open Client Settings**: Navigate to the settings panel of your code editor or client.
2.  **Find API Configuration**: Look for "OpenAI API Settings" or a similar section.
3.  **Set the API Endpoint**:
    -   In the "API Base URL" or "Endpoint" field, enter the URL of your adapter:
        `http://<your-vps-ip>:8000/v1`
4.  **Set the API Key**:
    -   In the "API Key" field, enter one of the **client keys** you defined in `ADAPTER_API_KEYS`.
5.  **Save and Test**: Save the settings and try a chat completion to confirm it's working.

## 📡 API Endpoints

### Public Endpoints
*No authentication required.*

-   `GET /`: Returns basic service information.
-   `GET /health`: A health check endpoint for monitoring. Returns `200 OK` if at least one Gemini key is active.

### Protected Endpoints
*Requires a **Client API Key** (`X-API-Key` or `Bearer` token).*

-   `POST /v1/chat/completions`: The main endpoint for chat completions.
-   `GET /v1/models`: Lists the available Gemini models configured in the adapter.
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
│   └── main.py            # FastAPI application (main server)
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