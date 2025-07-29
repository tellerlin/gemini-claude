# Gemini Claude Adapter

A high-performance Gemini Claude adapter designed for Claude Code and local clients, featuring multi-API key rotation, automatic failover, and streaming response support.

[🇨🇳 中文版本](README.md) | [🇺🇸 English Version](README.en.md)

## ✨ Key Features

- 🚀 **Ultra-fast Response** - Optimized request handling and key rotation algorithms
- 🔑 **Smart Key Management** - Failed keys cool down immediately, auto-switch to next available key
- 🌐 **Full Compatibility** - Compatible with Claude Code and OpenAI API formats
- ⚡ **Streaming Support** - Native streaming chat response support
- 🛡️ **Enterprise Features** - Proxy support, CORS handling, error recovery
- 📊 **Real-time Monitoring** - Detailed service status and key usage statistics

## 🎯 Design Goals

- **Fastest Response**: Optimized key rotation strategy with immediate failover
- **Strongest Compatibility**: Supports Claude Code and various clients
- **Highest Stability**: Automatic failover and key cooling mechanisms

## 🚀 Quick Start

### Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/tellerlin/gemini-claude.git
   cd gemini-claude
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   nano .env
   ```
   Add your Gemini API keys:
   ```bash
   GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...,AIzaSyGHI789...
   ```

5. **Start development server**
   ```bash
   python main.py
   ```

The server will start at `http://localhost:8000`.

### Production Deployment

Use the automated deployment script (Ubuntu 22.04 recommended):

**Method 1: Using Git Clone (Recommended)**
```bash
# Set environment variable and run deployment
export GITHUB_REPO_URL=https://github.com/tellerlin/gemini-claude.git
sudo bash scripts/deploy.sh
```

**Method 2: Manual Upload**
```bash
# Upload project to VPS
scp -r gemini-claude/ user@your-vps-ip:~/

# SSH to VPS
ssh user@your-vps-ip

# Run deployment script
cd gemini-claude
sudo bash scripts/deploy.sh
```

## 📡 API Endpoints

### Chat Completion
```
POST /v1/chat/completions
```

Request format (OpenAI compatible):
```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "model": "gemini-2.5-pro",
  "temperature": 0.7,
  "stream": false
}
```

### Health Check
```
GET /health
```

### Statistics
```
GET /stats
```

### Available Models
```
GET /v1/models
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEYS` | Gemini API keys, comma-separated | Required |
| `PROXY_URL` | Proxy server address | Optional |
| `MAX_FAILURES` | Failure threshold | 1 |
| `COOLING_PERIOD` | Cooling time (seconds) | 300 |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | 45 |
| `MAX_RETRIES` | Retry attempts | 0 |

### API Key Format Support

Multiple formats supported:
```
AIzaSyABC123...,AIzaSyDEF456...
"AIzaSyABC123...","AIzaSyDEF456..."
'AIzaSyABC123...','AIzaSyDEF456...'
AIzaSyABC123..., "AIzaSyDEF456...", 'AIzaSyGHI789...'
```

## 🛠️ Management Commands

### Local Development
```bash
# Start development server
python main.py

# Test with client
python client/client.py http://localhost:8000
```

### Production Environment
```bash
# Service management
gemini-manage start
gemini-manage stop
gemini-manage restart
gemini-manage status

# View logs
gemini-manage logs
gemini-manage error-logs

# Update dependencies
gemini-manage update

# Backup configuration
gemini-manage backup
```

## 🎮 Client Usage

### Interactive Chat
```bash
python client/client.py http://your-vps-ip
```

### Programmatic Usage
```python
from client.client import RemoteGeminiClient, ClientConfig

config = ClientConfig(
    vps_url="http://your-vps-ip",
    timeout=120,
    retries=3,
    preferred_model="gemini-2.5-pro"
)

client = RemoteGeminiClient(config)

# Send message
response = await client.chat_completion([
    {"role": "user", "content": "Hello!"}
])
```

## 🔒 Security Features

- 🔐 Secure API key storage and transmission
- 🛡️ CORS protection
- 🚫 Request rate limiting
- 📝 Detailed access logs
- 🔒 Firewall configuration

## 📊 Monitoring & Logging

- Real-time key status monitoring
- Detailed request logging
- Error tracking and analysis
- Performance metrics

## 🐛 Troubleshooting

### Common Issues

1. **Invalid API Keys**
   - Check key format in `.env` file
   - Ensure keys are valid and not expired

2. **Connection Timeout**
   - Check network connectivity
   - Consider using proxy
   - Adjust `REQUEST_TIMEOUT` value

3. **Service Unavailable**
   - Check service status: `gemini-manage status`
   - View error logs: `gemini-manage error-logs`

### Log Locations

- Application logs: `/home/gemini/gemini-claude/logs/app.log`
- Error logs: `/home/gemini/gemini-claude/logs/error.log`
- Deployment logs: `/tmp/gemini_deployment.log`

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

## 📞 Support

For questions, please check the documentation or submit an Issue.

---

**[🇨🇳 切换到中文版本](README.md)** | **[🇺🇸 English Version](README.en.md)**