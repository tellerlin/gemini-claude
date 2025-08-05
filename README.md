# Gemini Claude Adapter v3.0.0

A high-performance, secure Gemini adapter with **complete Anthropic API compatibility**, designed for Claude Code and other Anthropic clients. Features intelligent multi-API key rotation, automatic failover, robust security, streaming support, and advanced optimizations.

[🇨🇳 中文版本](README.zh.md) | [🇺🇸 English Version](README.md)

-----

## ✨ Key Features

  - 🤖 **Full Anthropic API Compatibility**: Complete support for the Anthropic Messages API (`/v1/messages`) with proper streaming format.
  - 🔑 **Smart Key Management**: Failed Gemini keys are immediately placed in a cool-down period, with automatic failover to the next available key.
  - 🛡️ **Robust Security**: Enforced API key authentication for all sensitive endpoints using client and admin keys.
  - ⚡ **Streaming Support**: Native support for Anthropic-style streaming responses with all required event types.
  - 🛠️ **Tool Calling Support**: Complete tool/function calling support between Anthropic and Gemini formats.
  - 📊 **Real-time Monitoring**: Endpoints for service health, key status, and usage statistics.
  - 🚀 **Performance Optimization**: Includes options for response caching and configurable timeouts/retries.
  - 🐳 **Simplified Docker Deployment**: Quick and secure setup using Docker and Docker Compose.

## 🚀 Quick Start Guide (Optimized for Simplicity)

This guide provides the simplest path to get your service running.

### Prerequisites

  - **Docker** and **Docker Compose** installed on your system.
  - **Google Gemini API Keys** ([Get yours here](https://aistudio.google.com/app/apikey)).
  - **Git** for cloning the repository.

### Step 1: Clone the Repository

Get the latest code from GitHub.

```bash
# Clone the repository and navigate into the directory
git clone https://github.com/tellerlin/gemini-claude.git
cd gemini-claude
```

### Step 2: Configure Your API Keys

Create your `.env` configuration file and add your essential API keys.

```bash
# Create your configuration file from the example
cp .env.example .env

# Open the .env file in a text editor (e.g., nano)
nano .env
```

Inside the `.env` file, you **must** set the following two values:

  * `GEMINI_API_KEYS`: Your Google Gemini API key(s). You can add multiple, separated by commas.
  * `SECURITY_ADAPTER_API_KEYS`: A secure key for your clients (e.g., Cursor) to access the adapter. Generate one with `openssl rand -hex 32`.

### Step 3: Build and Deploy the Service

This single command will make the necessary scripts executable, build the Docker image, and start the service in the background.

```bash
# Set permissions and deploy the service
chmod +x start.sh && docker-compose up -d --build
```

### Step 4: Verify the Deployment

Check that the container is running and view the logs to confirm a successful startup.

```bash
# Check the status of your container (should show 'running')
docker-compose ps

# View the latest logs to ensure it started without errors
docker-compose logs --since 5m
```

If the logs show "Application startup complete," your service is live and ready\! It will be available at `http://localhost:8000` (or your server's IP address).

-----

## 🔄 Updating Your Deployment (Optimized)

Follow these steps to update your existing deployment to the latest version.

### Step 1: Pull the Latest Code

Navigate to your project directory and pull the latest changes from the master branch.

```bash
# Navigate to your project directory
cd gemini-claude

# Pull the latest changes
git pull origin master
```

### Step 2: Stop, Rebuild, and Restart

This single command ensures that your startup script has the correct permissions, stops the old version, rebuilds the image with the latest code, and restarts the service.

```bash
# Set permissions, stop, rebuild with no cache, and restart
chmod +x start.sh && docker-compose down && docker-compose up -d --build --force-recreate
```

  * **Note:** We use `--build` to create a new image from the updated code and `--force-recreate` to ensure the old container is replaced.

### Step 3: Verify the Update

Check the logs to make sure the new version has started correctly.

```bash
# Check recent logs to ensure a successful startup
docker-compose logs --since 5m
```

This streamlined process ensures your deployment is always up-to-date with the latest features and security fixes.

-----

## 🧪 Test Your Deployment

The project includes a comprehensive test script (`test_endpoints.sh`) to verify all functionalities.

### 1\. Set Your Keys in the Script

Open the test script in an editor:

```bash
nano test_endpoints.sh
```

Inside the script, replace the placeholder values for `CLIENT_KEY` and `ADMIN_KEY` with the actual keys you set in your `.env` file.

### 2\. Make the Script Executable

This command only needs to be run once:

```bash
chmod +x test_endpoints.sh
```

### 3\. Run the Tests

```bash
# Make sure your adapter is running first!
./test_endpoints.sh
```

The script will run a series of tests against your live service and print the results. If all tests pass, your service is ready to use\!

## 🩺 Troubleshooting & Diagnostics

If you encounter issues, these diagnostic scripts can help you identify the problem. Run these commands from your project's root directory (the one containing `docker-compose.yml`).

### 1\. Checking API Key Validity

The `api_key_checker.py` script tests each `GEMINI_API_KEYS` from your `.env` file. It verifies if a key is valid, has quota, and can access the necessary models. It then offers to create a new `.env.updated` file containing only the working keys.

```bash
docker-compose run --rm gemini-claude-adapter python api_key_checker.py
```

The script is interactive and will guide you through the process of validating keys and saving the cleaned list.

### 2\. General Health & Import Check

The `diagnose_script.py` performs a general health check on your setup. It verifies that all required project files exist and that all Python dependencies from `requirements.txt` can be imported correctly inside the Docker environment.

```bash
docker-compose run --rm gemini-claude-adapter python diagnose_script.py
```

This helps quickly find issues related to a broken installation or missing files. The script will output a list of checks with a `✓` for success or `✗` for failure, helping you pinpoint the problem.

## ⚙️ Essential Configuration

Edit your `.env` file with these settings:

```dotenv
# =============================================
# REQUIRED: Gemini API Configuration
# =============================================
# Get your keys from: https://aistudio.google.com/app/apikey
# You can add multiple keys, separated by commas. Do not use quotes.
GEMINI_API_KEYS=your-google-ai-api-key-1,your-google-ai-api-key-2

# =============================================
# REQUIRED: Security Configuration
# =============================================
# This is the key your client application will use to authenticate.
# Generate a strong key with: openssl rand -hex 32
SECURITY_ADAPTER_API_KEYS=your-secure-client-key

# =============================================
# RECOMMENDED: Admin Access
# =============================================
# A separate, private key for accessing management endpoints.
SECURITY_ADMIN_API_KEYS=your-secure-admin-key

# =============================================
# Optional: Performance & Behavior
# =============================================
# Time in seconds to wait before reusing a key that failed due to a temporary issue (e.g., rate limit)
GEMINI_COOLING_PERIOD=300
# Timeout in seconds for a single request to the Gemini API
GEMINI_REQUEST_TIMEOUT=120
# Number of retries with different keys if a request fails
GEMINI_MAX_RETRIES=2
# Enable or disable response caching
CACHE_ENABLED=true
# Time-to-live for cached responses in seconds
CACHE_TTL=300
```

**⚠️ Important Security Notes:**

  - Keep your `.env` file secure and **never** commit it to version control.
  - Use strong, unique API keys for `SECURITY_ADAPTER_API_KEYS`.
  - It's highly recommended to set `SECURITY_ADMIN_API_KEYS` for production environments.

## 🔧 Client Configuration

### General Client Setup (e.g., Cursor IDE)

To connect a client that uses the Anthropic API format:

1.  **Open Client Settings**: Navigate to the settings of your IDE or client.
2.  **Find API Configuration**: Look for "Anthropic" or "Claude" API settings.
3.  **Set the API Endpoint**:
      - In the "API Base URL" or "Endpoint" field, enter the URL of your adapter:
        `http://<your-server-ip>:8000/v1`
        *(Note: The port may differ if you've customized it in `.env` or your `docker-compose.yml` file)*
4.  **Set the API Key**:
      - In the "API Key" field, enter the **client key** you defined in `SECURITY_ADAPTER_API_KEYS`.
5.  **Save and Test**

### Claude Code Configuration

Claude Code is Anthropic's command line tool for agentic coding. To configure it with your adapter:

#### 1\. Configure Environment Variables

Edit your shell profile file (choose one based on your shell):

```bash
# For bash users
nano ~/.bash_profile

# For zsh users (macOS default)
nano ~/.zshrc

# For other shells, edit the appropriate profile file
```

#### 2\. Add the Following Lines

Add these environment variables to your profile file:

```bash
# Gemini Claude Adapter Configuration
export ANTHROPIC_BASE_URL="http://your-server-ip:8000/v1"
export ANTHROPIC_AUTH_TOKEN="your-secure-client-key"
```

Replace:

  - `your-server-ip` with your server's IP address or domain name
  - `your-secure-client-key` with the key you set in `SECURITY_ADAPTER_API_KEYS`

#### 3\. Apply the Changes

```bash
# Reload your shell profile
source ~/.bash_profile  # or ~/.zshrc for zsh users

# Verify the configuration
echo $ANTHROPIC_BASE_URL
echo $ANTHROPIC_AUTH_TOKEN
```

#### 4\. Test Claude Code

```bash
# Test Claude Code with your adapter
claude-code --help

# Run a simple test
claude-code "Create a hello world Python script"
```

For more information about Claude Code, check the [official documentation](https://docs.anthropic.com/en/docs/claude-code).

### Supported Models

The adapter automatically maps common Anthropic model names to a compatible Gemini model. Based on the current code, the following models are mapped:

  - `claude-3-5-sonnet` → `gemini-2.0-flash-exp`
  - `claude-3-opus` → `gemini-2.0-flash-exp`
  - `claude-3-haiku` → `gemini-2.0-flash-exp`
  - Any other Anthropic model name will also default to `gemini-2.0-flash-exp`

## 📡 API Endpoints

### Main Endpoints (Client Key required)

| Endpoint       | Method | Description                        |
| :------------- | :----- | :--------------------------------- |
| `/v1/messages` | `POST` | **Primary Anthropic Messages API** |
| `/v1/models`   | `GET`  | List available models (mapped)     |
| `/stats`       | `GET`  | View key usage and performance statistics |
| `/metrics`     | `GET`  | Detailed performance metrics       |

### Admin Endpoints (Admin Key required)

| Endpoint                  | Method | Description                                    |
| :------------------------ | :----- | :--------------------------------------------- |
| `/admin/reset-key/{prefix}` | `POST` | Reset a failed/cooling Gemini key by its prefix |

### Public Endpoints

| Endpoint | Method | Description                                        |
| :------- | :----- | :------------------------------------------------- |
| `/`      | `GET`  | Service information                                |
| `/health`| `GET`  | Basic health check, indicates if active keys are available |

## 🛠️ Development and Deployment

### Local Development

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Create and configure your .env file
cp .env.example .env
# Edit .env with your configuration

# Run the application using Uvicorn
# The app will run on http://localhost:8000
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build the Docker image locally
docker build -t gemini-claude-adapter .

# Run the container using your .env file
# This maps the container's port 8000 to your machine's port 8000
docker run -d \
  --name gemini-claude-adapter \
  -p 8000:8000 \
  --env-file .env \
  gemini-claude-adapter
```

## 📈 Monitoring and Management

### Health Monitoring

Check the service health:

```bash
curl http://your-server-ip:8000/health
```

### Key Statistics

View current key usage and statistics:

```bash
curl -H "X-API-Key: your-client-key" \
     http://your-server-ip:8000/stats
```

### Performance Metrics

Get detailed performance metrics:

```bash
curl -H "X-API-Key: your-client-key" \
     http://your-server-ip:8000/metrics
```

### Admin Operations

Reset a failed key (requires admin key):

```bash
curl -X POST \
     -H "X-API-Key: your-admin-key" \
     http://your-server-ip:8000/admin/reset-key/AIzaSy
```

## 🔒 Security Best Practices

1.  **Use Strong API Keys**: Generate cryptographically secure keys using `openssl rand -hex 32`.
2.  **Separate Client and Admin Keys**: Use different keys for client access and administrative operations.
3.  **Network Security**: Consider running the service behind a reverse proxy (nginx, Cloudflare, etc.).
4.  **Rate Limiting**: The built-in rate limiting helps prevent abuse.
5.  **Monitoring**: Regularly check logs and metrics for unusual activity.
6.  **Updates**: Keep the adapter updated with the latest security patches.

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1.  Check the troubleshooting section above.
2.  Run the diagnostic scripts.
3.  Check the Docker logs: `docker-compose logs -f`.
4.  Open an issue on GitHub with detailed information about your problem.

## 📊 Performance Tips

  - **Multiple API Keys**: Use multiple Gemini API keys for better throughput and reliability.
  - **Caching**: Enable response caching for repeated requests.
  - **Monitoring**: Regularly check `/stats` and `/health` endpoints.
  - **Resource Allocation**: Ensure adequate system resources for your expected load.

---

**[🇨🇳 切换到中文版本](README.zh.md)** | **[🇺🇸 English Version](README.md)**
