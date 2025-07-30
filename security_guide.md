# Gemini Claude Adapter - Security Configuration Guide

## 🔐 Security Overview

The enhanced Gemini Claude Adapter now includes comprehensive API key authentication to secure your service. This guide covers setup, configuration, and best practices.

## 🚀 Quick Setup

### 1. Environment Configuration

Create your `.env` file with the following security settings:

```bash
# Required: Your Gemini API keys
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...

# Required for security: Client API keys
ADAPTER_API_KEYS=client-key-12345,client-key-67890,client-key-abcdef

# Optional: Separate admin keys for management endpoints
ADMIN_API_KEYS=admin-key-secure-123,admin-key-secure-456

# Other settings...
PORT=8000
HOST=0.0.0.0
```

### 2. Generate Secure API Keys

Use these methods to generate strong API keys:

```bash
# Method 1: Using openssl
openssl rand -hex 32

# Method 2: Using Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Method 3: Using uuidgen (shorter but still secure)
uuidgen | tr -d '-'
```

### 3. Start the Service

```bash
python main.py
```

## 🔑 Authentication Methods

The service supports two authentication methods:

### Method 1: X-API-Key Header
```bash
curl -H "X-API-Key: your-client-key" http://localhost:8000/v1/chat/completions
```

### Method 2: Bearer Token
```bash
curl -H "Authorization: Bearer your-client-key" http://localhost:8000/v1/chat/completions
```

## 📋 Endpoint Security Levels

### Public Endpoints (No Authentication Required)
- `GET /` - Service information
- `GET /health` - Health check for monitoring
- `OPTIONS /v1/chat/completions` - CORS preflight

### Protected Endpoints (Client API Key Required)
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List available models
- `GET /stats` - Service statistics

### Admin Endpoints (Admin API Key Required)
- `POST /admin/reset-key/{prefix}` - Reset Gemini API key
- `GET /admin/security-status` - Security configuration status

## 🛡️ Security Features

### 1. Flexible Key Management
- Support for multiple client keys
- Separate admin keys for management operations
- Automatic key validation and sanitization

### 2. Multiple Authentication Methods
- X-API-Key header (recommended for APIs)
- Authorization Bearer token (OAuth2 compatible)
- Fallback to client keys for admin operations if no admin keys set

### 3. Security Logging
- All authentication attempts are logged
- Failed authentication attempts are tracked
- Client identification in logs (first 8 chars of key)

### 4. CORS Support
- Proper CORS headers for web applications
- OPTIONS endpoint for preflight requests

## 🔧 Configuration Options

### Security Modes

#### 1. Disabled Security (Development Only)
```bash
# Leave ADAPTER_API_KEYS empty or unset
ADAPTER_API_KEYS=
```
⚠️ **Warning**: Only use this for development. All endpoints will be publicly accessible.

#### 2. Client Authentication Only
```bash
# Set client keys, no admin keys
ADAPTER_API_KEYS=client-key-1,client-key-2
# ADMIN_API_KEYS= (empty or unset)
```
Client keys can access all endpoints including admin operations.

#### 3. Full Security (Recommended for Production)
```bash
# Separate client and admin keys
ADAPTER_API_KEYS=client-key-1,client-key-2
ADMIN_API_KEYS=admin-key-1,admin-key-2
```
Client keys can only access chat and stats endpoints. Admin keys required for management operations.

## 🚀 Production Deployment

### 1. Environment Security
```bash
# Set restrictive file permissions
chmod 600 .env

# Use environment variables instead of .env file in production
export ADAPTER_API_KEYS="key1,key2,key3"
export ADMIN_API_KEYS="admin1,admin2"
```

### 2. Reverse Proxy Configuration (Nginx)
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    }
    
    # Rate limiting (optional)
    location /v1/chat/completions {
        limit_req zone=api_limit burst=10 nodelay;
        proxy_pass http://127.0.0.1:8000;
        # ... other proxy settings
    }
}

# Rate limiting configuration (add to http block)
http {
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=60r/m;
}
```

### 3. Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Security: Don't include .env in image
# Use environment variables or secrets
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["python", "main.py"]
```

### 4. Docker Compose with Secrets
```yaml
version: '3.8'
services:
  gemini-adapter:
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - GEMINI_API_KEYS=${GEMINI_API_KEYS}
      - ADAPTER_API_KEYS=${ADAPTER_API_KEYS}
      - ADMIN_API_KEYS=${ADMIN_API_KEYS}
    secrets:
      - gemini_keys
      - adapter_keys
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

secrets:
  gemini_keys:
    external: true
  adapter_keys:
    external: true
```

## 🔍 Monitoring and Logging

### 1. Security Monitoring
```bash
# Monitor authentication failures
tail -f logs/gemini_adapter_*.log | grep "Invalid API key"

# Monitor successful authentications
tail -f logs/gemini_adapter_*.log | grep "Chat completion request from client"
```

### 2. Health Monitoring
```bash
# Simple health check
curl http://localhost:8000/health

# Detailed stats (requires authentication)
curl -H "X-API-Key: your-client-key" http://localhost:8000/stats
```

### 3. Admin Operations
```bash
# Check security status
curl -H "X-API-Key: your-admin-key" http://localhost:8000/admin/security-status

# Reset a failed Gemini API key
curl -X POST -H "X-API-Key: your-admin-key" http://localhost:8000/admin/reset-key/AIzaSyAB
```

## 🚨 Security Best Practices

### 1. Key Management
- **Generate strong keys**: Use cryptographically secure random generators
- **Rotate keys regularly**: Change API keys periodically
- **Limit key distribution**: Only provide keys to trusted clients
- **Monitor key usage**: Track which keys are being used and how often

### 2. Access Control
- **Use admin keys**: Separate admin keys for management operations
- **Principle of least privilege**: Give clients only the access they need
- **Regular audits**: Review key usage and access patterns

### 3. Network Security
- **Use HTTPS**: Always use SSL/TLS in production
- **Firewall rules**: Restrict access to necessary IPs only
- **Rate limiting**: Implement rate limiting to prevent abuse
- **Reverse proxy**: Use nginx/Apache for additional security layers

### 4. Monitoring
- **Log everything**: Monitor all authentication attempts
- **Set up alerts**: Alert on suspicious activity or failures
- **Regular health checks**: Monitor service health and key status
- **Backup configurations**: Keep secure backups of your configuration

## 🔧 Troubleshooting

### Common Issues

#### 1. "Invalid API key" Error
```bash
# Check if your key is in the ADAPTER_API_KEYS list
curl -H "X-API-Key: your-key" http://localhost:8000/health
```

#### 2. "Admin API key required" Error
```bash
# Make sure you're using an admin key for admin endpoints
curl -H "X-API-Key: your-admin-key" http://localhost:8000/admin/security-status
```

#### 3. Authentication Not Working
1. Check `.env` file format - no spaces around commas
2. Verify key format - no quotes in environment variables
3. Check logs for detailed error messages
4. Test with a simple health check first

#### 4. Service Won't Start
```bash
# Check environment variables
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('ADAPTER_API_KEYS:', os.getenv('ADAPTER_API_KEYS'))
print('ADMIN_API_KEYS:', os.getenv('ADMIN_API_KEYS'))
"
```

## 📞 Support

If you encounter issues:

1. Check the logs in `logs/gemini_adapter_*.log`
2. Verify your `.env` configuration
3. Test with the provided client examples
4. Ensure your keys are properly formatted

Remember: Security is only as strong as your weakest link. Follow all best practices for a secure deployment!