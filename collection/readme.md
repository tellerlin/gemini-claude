# Gemini Claude Adapter - Complete Deployment and Usage Guide

## 1\. System Overview

Gemini Claude Adapter is a high-performance, self-hosted API adapter designed to integrate tools like Claude Code with Google's Gemini API. It provides a robust, enterprise-grade solution for managing multiple API keys, ensuring high availability and optimal performance through features like intelligent key rotation, automatic failover, and comprehensive monitoring.

### 1.1 Core Features

  - **Intelligent Multi-Key Management**: Automatically rotates through a list of Gemini API keys to distribute load and avoid rate limits.
  - **Automatic Failover & Cooldown**: Detects failing API keys (due to quota issues or errors), places them in a temporary "cooling" period, and automatically reactivates them later.
  - **Resilient Request Retries**: If a request fails with one key, the system automatically retries with the next available key, enhancing reliability.
  - **Full Streaming Support**: Natively supports streaming chat responses for real-time, interactive experiences, as well as standard non-streaming requests.
  - **Proxy Integration**: Supports routing all outgoing traffic to the Gemini API through an HTTP/HTTPS proxy.
  - **Comprehensive API & Monitoring**:
      - Offers an OpenAI-compatible API endpoint (`/v1/chat/completions`).
      - Provides real-time endpoints for health checks (`/health`) and detailed usage statistics (`/stats`).
      - Includes an administrative endpoint to manually reset the status of a specific API key.
  - **Automated Deployment**: A comprehensive shell script automates the entire server setup process, including dependency installation, service configuration, and security hardening.
  - **Advanced Client & Tools**: Comes with a feature-rich local client for interactive chat, performance testing, and server management, alongside dedicated scripts for monitoring and administration.

### 1.2 Recommended Architecture

For optimal performance, especially for users in regions with high latency to Google's services, the following architecture is recommended:

```
User's Local Machine (IDE, Client) → Internet → VPS (e.g., Japan) [Nginx → Gemini Adapter Service] → Google Gemini API
```

  - **Why a VPS in Japan?** It often provides a low-latency network path to Google's Asian data centers (typically 60-120ms), improving response times.
  - **Nginx**: Acts as a secure, high-performance reverse proxy, handling SSL/TLS termination and load balancing if needed.
  - **Supervisor**: Manages the application process, ensuring it runs continuously and restarts automatically if it crashes.

## 2\. Requirements

### 2.1 Server (VPS) Requirements

  - **Operating System**: Ubuntu 22.04 LTS (recommended) or other Debian-based Linux distributions.
  - **CPU**: 2 cores (minimum), 4 cores (recommended for higher loads).
  - **Memory**: 2 GB (minimum), 4 GB (recommended).
  - **Storage**: 50 GB SSD or NVMe.
  - **Dependencies**: The deployment script (`fixed_deployment.sh`) will automatically install the following:
      - Python 3.11
      - Nginx (Web Server / Reverse Proxy)
      - Supervisor (Process Manager)
      - UFW (Firewall)
      - Git, Curl, Wget

### 2.2 Application Requirements

  - **Gemini API Keys**: A minimum of 3 keys is recommended to ensure effective rotation and failover. 5-10 keys are ideal for heavy use.
  - [cite\_start]**Python Packages**: The deployment script will automatically install all necessary packages from `requirements.txt` into a virtual environment. [cite: 1]
      - [cite\_start]`fastapi`, `uvicorn`: For the web server framework. [cite: 1]
      - [cite\_start]`litellm`: To interact with the Gemini API. [cite: 1]
      - [cite\_start]`httpx`: For making HTTP requests. [cite: 1]
      - [cite\_start]`loguru`: For advanced logging. [cite: 1]
      - [cite\_start]`python-dotenv`: For managing environment variables. [cite: 1]
      - [cite\_start]`pydantic`: For data validation. [cite: 1]

## 3\. Automated Deployment

The provided deployment script automates the entire setup process.

### 3.1 Deployment Steps

**Step 1: Upload Project Files to VPS**
Use `scp` or any SFTP client to upload the entire project directory to your VPS's home folder.

```bash
# Example from your local machine:
scp -r /path/to/gemini-claude-adapter/ user@your-vps-ip:~/
```

**Step 2: Run the Deployment Script**
Connect to your VPS via SSH and execute the deployment script.

```bash
# Connect to your VPS
ssh user@your-vps-ip

# Navigate to the project directory
cd gemini-claude-adapter

# Make the script executable
chmod +x fixed_deployment.sh

# Run the script with sudo
sudo bash ./fixed_deployment.sh
```

**Step 3: Configure API Keys**
The script will create a `.env` file. You must edit this file to add your Gemini API keys.

```bash
# Edit the environment configuration file
sudo nano /home/gemini/gemini-claude-adapter/.env

# Find this line and add your keys, separated by commas
GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3,your_key_4
```

**Step 4: Start the Service**
A management script is created for easy service control. Use it to restart the service after adding your keys.

```bash
# Restart the service to apply the new configuration
sudo gemini-manage restart

# Check the service status to ensure it's running
gemini-manage status
```

If successful, the service is now live and accessible via your VPS IP address.

### 3.2 Deployment Script Breakdown

The `fixed_deployment.sh` script performs the following actions:

1.  **Checks for Root Privileges**: Ensures the script is run with `sudo`.
2.  **Installs System Dependencies**: Installs Python 3.11, Nginx, Supervisor, UFW, and other necessary tools.
3.  **Creates Application User**: Creates a dedicated, non-root user named `gemini` to run the application securely.
4.  **Sets Up Application Environment**: Copies the project files to `/home/gemini/gemini-claude-adapter`, creates a Python virtual environment, and installs all required packages.
5.  **Configures Supervisor**: Sets up a Supervisor configuration file to manage the Uvicorn process, enabling auto-start on boot and automatic restarts on failure. Logs are configured to be stored in `/home/gemini/gemini-claude-adapter/logs/`.
6.  **Configures Nginx**: Sets up Nginx as a reverse proxy to listen on port 80 and forward requests to the application. The configuration is optimized for handling streaming responses.
7.  **Configures Firewall**: Enables UFW and allows traffic on standard ports (SSH, HTTP, HTTPS).
8.  **Creates `.env` Template**: Generates the environment file for you to add your secrets.
9.  **Creates Management Script**: Creates a powerful `gemini-manage` command available system-wide for easy service management.

## 4\. Service Management

The `gemini-manage` command, installed at `/usr/local/bin/gemini-manage`, is the primary tool for controlling the adapter service on your VPS.

| Command | Description |
| :--- | :--- |
| `gemini-manage start` | Starts the Gemini Adapter service. |
| `gemini-manage stop` | Stops the Gemini Adapter service. |
| `gemini-manage restart` | Restarts the service. Use this after changing the `.env` file. |
| `gemini-manage status` | Shows the running status from Supervisor and performs a live health check. |
| `gemini-manage logs` | Tails the main application log (`app.log`) in real-time. |
| `gemini-manage error-logs` | Tails the error log (`error.log`) in real-time. |
| `gemini-manage monitor`| Launches the continuous monitoring interface from `monitoring_scripts.py`. |

## 5\. Local Client and Usage

The project includes a powerful local client (`enhanced_client.py`) for interacting with the deployed adapter.

### 5.1 Initial Client Setup

First, run the setup wizard to configure the client to connect to your VPS.

```bash
# On your local machine, run the setup wizard
python enhanced_client.py --setup
```

The wizard will prompt you for your VPS IP or domain name and test the connection. This will create a `vps_config.json` file for storing the connection details.

### 5.2 Interactive Chat

You can start an interactive chat session with support for command processing and conversation history.

```bash
# Start a standard chat session
python enhanced_client.py --chat

# Start a chat session with real-time streaming responses
python enhanced_client.py --stream
```

**In-Chat Commands:**

| Command | Description |
| :--- | :--- |
| `/quit`, `/exit`, `/q` | Exit the chat session. |
| `/clear` | Clear the current conversation history. |
| `/history` | Display the last 10 messages from the history. |
| `/model <name>` | Temporarily switch the model for subsequent requests (e.g., `/model gemini-1.5-flash`). |
| `/temp <value>` | Set the temperature for subsequent requests (e.g., `/temp 0.5`). |
| `/help` | Display the list of available commands. |
| `/stats` | Fetch and display detailed statistics from the server. |
| `/health` | Perform a health check on the server. |

### 5.3 Direct Command-Line Usage

| Command | Description |
| :--- | :--- |
| `python enhanced_client.py --message "Your prompt"` | Send a single, non-interactive message and print the reply. |
| `python enhanced_client.py --health` | Perform and display a detailed health check. |
| `python enhanced_client.py --stats` | Fetch and display detailed key statistics. |
| `python enhanced_client.py --models` | List the available models from the server. |
| `python enhanced_client.py --test` | Run a quick connection and health test. |
| `python enhanced_client.py --perf <N>` | Run a concurrency performance test with `N` simultaneous requests. |
| `python enhanced_client.py --config` | View and update the saved client configuration. |

### 5.4 Claude Code & IDE Integration

To integrate with an IDE plugin or tool that uses the OpenAI API format:

  - **API Endpoint/URL**: `http://<YOUR_VPS_IP>/v1`
  - **API Key**: Can be any non-empty string (e.g., "dummy-key"). The adapter does not validate this key.
  - **Model**: `gemini-1.5-pro` or `gemini-1.5-flash`.

## 6\. Monitoring and Maintenance

The `monitoring_scripts.py` file provides command-line tools for monitoring and administering your running instance.

| Command | Description |
| :--- | :--- |
| `python monitoring_scripts.py monitor` | Starts a continuous, real-time monitoring dashboard in your terminal. |
| `python monitoring_scripts.py --interval 60 monitor` | Starts monitoring with a custom 60-second refresh interval. |
| `python monitoring_scripts.py status` | Fetches and displays the current health and detailed key statistics once. |
| `python monitoring_scripts.py reset` | Starts an interactive wizard to reset the status of a key that is in a "cooling" or "failed" state. |
| `python monitoring_scripts.py test` | Performs a simple connection test to the server. |

## 7\. Comprehensive Testing

After deployment, you can run a full suite of tests using `test_client.py` to ensure every part of the system is functioning correctly.

```bash
# Run all standard tests
python test_client.py

# Run a quick test suite (skips longer tests)
python test_client.py --quick

# Run all tests including a 60-second performance benchmark
python test_client.py --performance --perf-duration 60
```

The test suite validates:

  - Health, stats, and models endpoints.
  - Basic and streaming chat completions.
  - Concurrent request handling.
  - Proper error handling for invalid requests.
  - A basic performance benchmark (optional).

## 8\. Configuration Details

### 8.1 Server Environment Variables (`.env`)

Configuration is managed via the `.env` file in the application directory on the server (`/home/gemini/gemini-claude-adapter/.env`).

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GEMINI_API_KEYS` | **Required.** Comma-separated list of your Google Gemini API keys. | `your_key_1,...` |
| `PROXY_URL` | Optional. The URL for an HTTP/HTTPS proxy (e.g., `http://user:pass@host:port`). | `None` |
| `PORT` | The local port the Uvicorn server will bind to. | `8000` |
| `HOST` | The local host the Uvicorn server will bind to. | `0.0.0.0` |
| `MAX_FAILURES` | Number of consecutive failures before a key is marked for cooling. | `3` |
| `COOLING_PERIOD` | The duration in seconds a key will remain in "cooling" status. | `300` (5 minutes) |
| `REQUEST_TIMEOUT` | The timeout in seconds for a single request to the Gemini API. | `45` |
| `MAX_RETRIES` | The number of times to retry a failed request using different keys. | `2` |

### 8.2 Nginx Configuration

The Nginx configuration is located at `/etc/nginx/sites-available/gemini-adapter`. It is optimized for proxying streaming responses and can be customized for advanced use cases like SSL/TLS.

## 9\. API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Root endpoint with basic service info and a list of other endpoints. |
| `/v1/chat/completions` | `POST` | The main chat endpoint. It is OpenAI-compatible and supports streaming. |
| `/v1/models` | `GET` | Returns a list of available Gemini models supported by the adapter. |
| `/health` | `GET` | Provides a detailed health check, including the status and count of all API keys. |
| `/stats` | `GET` | Returns detailed statistics for each individual API key, including request counts and failure rates. |
| `/admin/reset-key/{key_prefix}` | `POST` | Manually resets a key's status to "active" using its first 8 characters. |

## 10\. Troubleshooting

| Issue | Diagnosis & Solution |
| :--- | :--- |
| **Connection Timeout** | 1. **Check Service Status**: Run `gemini-manage status` on the VPS. If it's down, `gemini-manage restart`. \<br\> 2. **Check Firewall**: Run `sudo ufw status`. Ensure port 80 (HTTP) or 443 (HTTPS) is allowed. \<br\> 3. **Check Nginx**: Run `sudo systemctl status nginx`. If it's not running, restart it. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`. |
| **503 Service Unavailable** | This error from the `/health` endpoint or chat completions means all your API keys are currently in a "cooling" or "failed" state. \<br\> 1. **Check Key Status**: Use `python monitoring_scripts.py status`. \<br\> 2. **Reset Keys**: Use `python monitoring_scripts.py reset` to manually reactivate keys. \<br\> 3. **Add More Keys**: Edit the `.env` file to add more API keys and run `gemini-manage restart`. |
| **502 Bad Gateway** | This typically means Nginx cannot communicate with the backend Uvicorn application. \<br\> 1. **Check Service Status**: Run `gemini-manage status`. Ensure the adapter service is running. \<br\> 2. **Check Logs**: Look at `gemini-manage logs` and `gemini-manage error-logs` for application crashes or errors. |
| **High Memory Usage** | 1. **Restart Gracefully**: `gemini-manage restart` will free up memory. \<br\> 2. **Upgrade VPS**: If high usage is persistent due to heavy traffic, consider upgrading your VPS's memory. |