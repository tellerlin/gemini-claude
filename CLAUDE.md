# CLAUDE.md - Gemini Claude Adapter

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Use English to write text.

## Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI 0.104.0+
- **ASGI Server**: Uvicorn 0.24.0+
- **LLM Integration**: LiteLLM 1.17.0+
- **Data Validation**: Pydantic 2.5.0+
- **HTTP Client**: HTTPX 0.25.0+
- **Logging**: Loguru 0.7.0+
- **Environment**: python-dotenv 1.0.0+

## Project Structure

```
gemini-claude/
├── main.py                 # Development entry point
├── src/
│   └── main.py            # FastAPI application (main server)
├── client/
│   └── client.py          # Remote client for VPS connections
├── scripts/
│   └── deploy.sh          # Production deployment script
├── collection/            # Documentation and utilities
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
└── README.md             # Project documentation
```

## Core Components

### Main Application (`src/main.py`)
- **FastAPI server** providing OpenAI-compatible API endpoints
- **API key rotation** with automatic failover and cooling mechanisms
- **Streaming support** for real-time chat responses
- **Health monitoring** and statistics endpoints
- **CORS handling** for cross-origin requests

### Key Manager
- **Smart key rotation** using round-robin algorithm
- **Automatic cooling** for failed keys (configurable duration)
- **Failure tracking** with detailed statistics
- **Key reset functionality** via admin endpoints

### LiteLLM Adapter
- **Gemini API integration** through LiteLLM abstraction
- **Request timeout** and retry handling
- **Proxy support** for restricted network environments
- **Streaming and non-streaming** response handling

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys
```

### Development Server
```bash
# Start development server (recommended way)
python main.py

# Alternative: direct uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation at: http://localhost:8000/docs
```

### Testing
```bash
# Test with the included client
python client/client.py http://localhost:8000

# Health check
curl http://localhost:8000/health

# View server statistics
curl http://localhost:8000/stats
```

### Git Workflow
```bash
# Add all changes
git add .

# Commit with descriptive message
git commit -m "Description of changes made"

# Push to remote
git push origin master
```

## Code Style Guidelines

### Python Conventions
- **Follow PEP 8** for Python code style
- **Use type hints** for all function parameters and return values
- **Prefer async/await** for I/O operations
- **Use dataclasses** for structured data
- **Implement proper error handling** with specific exception types

### FastAPI Best Practices
- **Use Pydantic models** for request/response validation
- **Implement proper status codes** for different response types
- **Add meaningful docstrings** to all endpoints
- **Use dependency injection** for shared resources
- **Implement proper CORS headers** for web clients

### Error Handling
- **Use HTTPException** for API errors with appropriate status codes
- **Log all errors** with context using Loguru
- **Provide user-friendly error messages** in API responses
- **Implement graceful degradation** for service interruptions

## Configuration

### Required Environment Variables
```bash
GEMINI_API_KEYS=AIzaSyABC123...,AIzaSyDEF456...  # Required: Gemini API keys
```

### Optional Environment Variables
```bash
PROXY_URL=http://proxy:port                        # Optional: Proxy server
MAX_FAILURES=1                                    # Optional: Failures before cooling (default: 1)
COOLING_PERIOD=300                                # Optional: Cooling duration in seconds (default: 300)
REQUEST_TIMEOUT=45                               # Optional: Request timeout (default: 45)
MAX_RETRIES=0                                    # Optional: Retry attempts (default: 0)
HEALTH_CHECK_INTERVAL=60                         # Optional: Health check interval (default: 60)
PORT=8000                                        # Optional: Server port (default: 8000)
HOST=0.0.0.0                                     # Optional: Server host (default: 0.0.0.0)
```

### API Key Format Support
- Comma-separated: `AIzaSyABC123...,AIzaSyDEF456...`
- Quoted strings: `"AIzaSyABC123...","AIzaSyDEF456..."`
- Mixed formats: `AIzaSyABC123..., "AIzaSyDEF456...", 'AIzaSyGHI789...'`

## API Endpoints

### Primary Endpoints
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /stats` - Server statistics and key status

### Admin Endpoints
- `POST /admin/reset-key/{key_prefix}` - Reset a specific key's status

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation
- `GET /` - Basic service information

## Testing Guidelines

### Unit Testing
- **Test all new features** with appropriate unit tests
- **Mock external dependencies** (API calls, database connections)
- **Test error scenarios** alongside happy paths
- **Use descriptive test names** that explain what is being tested

### Integration Testing
- **Test API endpoints** with HTTP client requests
- **Verify key rotation** behavior with multiple API keys
- **Test streaming responses** for real-time functionality
- **Validate error handling** for various failure scenarios

### Performance Testing
- **Test with concurrent requests** to verify key rotation
- **Monitor memory usage** during long-running operations
- **Verify timeout handling** for slow or unresponsive APIs
- **Test proxy functionality** if applicable

## Deployment

### Production Deployment
```bash
# Use the provided deployment script
sudo bash scripts/deploy.sh

# Manual deployment steps:
# 1. Copy project to server
# 2. Set up virtual environment
# 3. Install dependencies
# 4. Configure environment variables
# 5. Set up systemd service (or use docker)
```

### Service Management
```bash
# The deployment script sets up these commands:
gemini-manage start      # Start the service
gemini-manage stop       # Stop the service
gemini-manage restart    # Restart the service
gemini-manage status     # Check service status
gemini-manage logs       # View application logs
gemini-manage error-logs # View error logs only
```

## Security Considerations

### API Key Security
- **Never commit API keys** to version control
- **Use environment variables** for sensitive configuration
- **Rotate keys regularly** and monitor usage
- **Implement rate limiting** in production

### Production Security
- **Use HTTPS** in production environments
- **Implement authentication** for admin endpoints
- **Configure firewall rules** appropriately
- **Monitor access logs** for suspicious activity

### CORS Configuration
- **Restrict origins** in production (currently open for development)
- **Validate request headers** for API calls
- **Implement proper authentication** for production use

## Performance Optimization

### Key Management
- **Monitor key success rates** and remove consistently failing keys
- **Adjust cooling periods** based on API rate limits
- **Use multiple keys** for better load distribution
- **Implement proper timeout** handling for slow responses

### Resource Management
- **Configure appropriate connection limits** for HTTP clients
- **Implement proper logging** levels to avoid performance impact
- **Monitor memory usage** during high-load scenarios
- **Use connection pooling** for HTTP connections

## Context7 MCP Integration

**IMPORTANT**: Actively use the Context7 MCP server to:
- **Access latest FastAPI best practices** and patterns
- **Get current LiteLLM documentation** and usage examples
- **Research optimal API key rotation** strategies
- **Stay updated with Gemini API** changes and new features
- **Learn about Python async/await** patterns and performance optimizations

## Troubleshooting

### Common Issues

**API Key Failures**
- Check key format and validity
- Verify key hasn't exceeded rate limits
- Review logs for specific error messages
- Use `/stats` endpoint to check key status

**Connection Issues**
- Verify network connectivity
- Check proxy configuration if applicable
- Ensure firewall allows port 8000
- Review server logs for startup errors

**Performance Issues**
- Monitor key rotation statistics
- Check for timeout or rate limit issues
- Review memory and CPU usage
- Consider increasing timeout values for slow networks

### Log Analysis
```bash
# View application logs
tail -f logs/gemini_adapter_*.log

# Check for specific error patterns
grep "ERROR" logs/gemini_adapter_*.log

# Monitor key rotation events
grep "cooling\|active" logs/gemini_adapter_*.log
```

## Development Workflow

### Feature Development
1. **Research requirements** using Context7 MCP for latest patterns
2. **Plan implementation** considering existing architecture
3. **Write tests** before implementing new features
4. **Implement changes** following established patterns
5. **Test thoroughly** including edge cases
6. **Update documentation** as needed
7. **Commit changes** with descriptive messages

### Bug Fixes
1. **Reproduce the issue** and identify root cause
2. **Write failing test** that demonstrates the bug
3. **Implement fix** following existing code patterns
4. **Verify fix** resolves the issue without breaking existing functionality
5. **Update tests** and documentation if needed

## Important Notes

- **Always test with multiple API keys** to verify rotation functionality
- **Monitor key statistics** regularly in production environments
- **Use Context7 MCP** to stay current with best practices and API changes
- **Follow async/await patterns** consistently throughout the codebase
- **Maintain backward compatibility** when modifying API endpoints
- **Document breaking changes** in commit messages and release notes

Remember: This adapter serves as a critical bridge between Claude Code and Gemini API. Reliability, proper error handling, and performance are key priorities.