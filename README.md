# Gemini Claude Adapter v2.1.0

A high-performance, secure Gemini adapter with **complete Anthropic API compatibility**, designed for Claude Code and other Anthropic clients. Features intelligent multi-API key rotation, automatic failover, robust security, streaming support, and advanced optimizations.

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

## ✨ Key Features

-   🤖 **Full Anthropic API Compatibility**: Complete support for the Anthropic Messages API (`/v1/messages`) with proper streaming format.
-   🔑 **Smart Key Management**: Failed Gemini keys are immediately placed in a cool-down period, with automatic failover to the next available key.
-   🛡️ **Robust Security**: Enforced API key authentication for all sensitive endpoints using client and admin keys.
-   ⚡ **Streaming Support**: Native support for Anthropic-style streaming responses with all required event types.
-   🛠️ **Tool Calling Support**: Complete tool/function calling support between Anthropic and Gemini formats.
-   📊 **Real-time Monitoring**: Endpoints for service health, key status, and usage statistics.
-   🐳 **Simplified Docker Deployment**: Quick and secure setup using Docker and Docker Compose.

---

## 🚀 Quick Start Guide

### Prerequisites

-   **Docker** and **Docker Compose** installed on your system.
-   **Google Gemini API Keys** ([Get yours here](https://makersuite.google.com/app/apikey)).
-   **Git** for cloning the repository.

### Step 1: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude

# Create your configuration file from the example
cp .env.example .env

# Edit the .env file and add your keys
# Use your favorite editor, for example: nano .env
nano .env
````

You **must** set `GEMINI_API_KEYS` and `SECURITY_ADAPTER_API_KEYS` in your `.env` file. See the configuration section below for details.

### Step 2: Deploy the Service

```bash
# Build and start the service in the background
docker-compose up -d --build

# Check if it's running correctly
docker-compose ps
```

The service will be available at `http://localhost:80` (or your server's IP). To check the logs in real-time, run `docker-compose logs -f`.

### Step 3: Test Your Deployment 🧪

The project includes a comprehensive test script (`test_endpoints.sh`) to verify all functionalities.

**1. Set Your Keys in the Script**
Open the test script in an editor:

```bash
nano test_endpoints.sh
```

Inside the script, replace the placeholder values for `CLIENT_KEY` and `ADMIN_KEY` with the actual keys you set in your `.env` file.

**2. Make the Script Executable**
This command only needs to be run once.

```bash
chmod +x test_endpoints.sh
```

**3. Run the Tests**

```bash
./test_endpoints.sh
```

The script will now run a series of tests against your live service and print the results. If all tests pass, your service is ready to use\!

-----

## ⚙️ Essential Configuration

Edit your `.env` file with these settings:

```env
# =============================================
# REQUIRED: Gemini API Configuration
# =============================================
# Get your keys from: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
# You can add multiple keys, separated by commas
GEMINI_API_KEYS=your-google-ai-api-key-1,your-google-ai-api-key-2

# =============================================
# REQUIRED: Security Configuration
# =============================================
# This is the key your client application will use to authenticate.
# Generate a strong key with: openssl rand -hex 32
SECURITY_ADAPTER_API_KEYS=your-client-key

# =============================================
# RECOMMENDED: Admin Access
# =============================================
# A separate, private key for accessing management endpoints.
SECURITY_ADMIN_API_KEYS=your-admin-key
```

**⚠️ Important Security Notes:**

  - Keep your `.env` file secure and **never** commit it to version control.
  - Use strong, unique API keys for `SECURITY_ADAPTER_API_KEYS`.
  - It's highly recommended to set `SECURITY_ADMIN_API_KEYS` for production environments.

-----

## 🔧 Client Configuration (e.g., Cursor IDE)

To connect a client that uses the Anthropic API format:

1.  **Open Client Settings**: Navigate to the settings of your IDE or client.
2.  **Find API Configuration**: Look for "Anthropic" or "Claude" API settings.
3.  **Set the API Endpoint**:
      - In the "API Base URL" or "Endpoint" field, enter the URL of your adapter:
        `http://<your-server-ip>:80/v1`
4.  **Set the API Key**:
      - In the "API Key" field, enter the **client key** you defined in `SECURITY_ADAPTER_API_KEYS`.
5.  **Save and Test**.

### Supported Models

The adapter automatically maps common Anthropic model names to Gemini:

  - `claude-3-5-sonnet` → `gemini-2.5-pro`
  - `claude-3-5-haiku` → `gemini-2.5-flash`
  - And other Claude 3 models.

-----

## 📡 API Endpoints

### Main Endpoints (Client Key required)

| Endpoint                | Method | Description                        |
| ----------------------- | ------ | ---------------------------------- |
| `/v1/messages`          | POST   | **Primary Anthropic Messages API** |
| `/v1/models`            | GET    | List available models              |
| `/stats`                | GET    | View key usage statistics          |
| `/metrics`              | GET    | Detailed performance metrics       |

### Admin Endpoints (Admin Key required)

| Endpoint                    | Method | Description                   |
| --------------------------- | ------ | ----------------------------- |
| `/admin/reset-key/{prefix}` | POST   | Reset a failed Gemini key     |
| `/cache/clear`              | POST   | Clear the response cache      |
| `/errors/recent`            | GET    | View recent error logs        |

### Public Endpoints

| Endpoint | Method | Description         |
| -------- | ------ | ------------------- |
| `/`      | GET    | Service information |
| `/health`  | GET    | Basic health check  |


---

**[🇨🇳 切换到中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**
