# CLAUDE.md - Gemini Claude Adapter

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Use English to write text.

## Architecture Overview

This is a FastAPI-based adapter that provides OpenAI-compatible endpoints for Google's Gemini API. The service includes intelligent API key rotation, security authentication, and Docker deployment.

### Key Design Patterns
- **FastAPI + Pydantic** for type-safe API development
- **Dependency Injection** for clean component separation
- **Async/Await** throughout for optimal performance
- **Middleware-based** security authentication
- **Docker-first** deployment approach

## Development Guidelines

### Code Style
- Follow PEP 8 with 4-space indentation
- Use type hints for all function signatures
- Prefer async/await for I/O operations
- Use dataclasses for structured data
- Implement proper error handling with specific exception types

### Key Components Location
- **Main Application**: `src/main.py` - FastAPI app with endpoints
- **Entry Point**: `main.py` - Development server startup
- **Configuration**: `.env` file (not committed to repo)
- **Docker**: `docker-compose.yml` and `Dockerfile`

### Security Implementation
- API key authentication via middleware
- Two-tier access: Client keys and Admin keys
- Environment-based configuration
- CORS handling for web clients

### Testing Approach
- Test endpoints with HTTP client requests
- Verify key rotation behavior with multiple keys
- Test streaming responses for real-time functionality
- Validate error handling for failure scenarios

## Common Tasks

### Adding New Endpoints
1. Define Pydantic models for request/response
2. Add endpoint function with proper typing
3. Include appropriate authentication decorators
4. Add to OpenAPI documentation with docstrings
5. Update tests accordingly

### Modifying Key Management
- Key rotation logic is in the KeyManager class
- Cooling mechanism uses configurable timeouts
- Statistics tracking is automatic
- Admin endpoints require special authentication

### Docker Development
- Use `docker-compose up -d` for development
- Mount volumes for live code changes
- Use `docker-compose logs -f` for debugging
- Rebuild with `--build` flag when dependencies change

## Important Notes

- Always test with multiple API keys to verify rotation
- Monitor key statistics in production environments
- Use Context7 MCP for latest FastAPI best practices
- Follow async/await patterns consistently
- Maintain backward compatibility for API endpoints

## Environment Variables

The application expects these environment variables:
- `GEMINI_API_KEYS` (required): Comma-separated Gemini API keys
- `ADAPTER_API_KEYS` (required): Client authentication keys
- `ADMIN_API_KEYS` (optional): Admin authentication keys
- `PROXY_URL` (optional): HTTP proxy for Gemini API calls
- `MAX_FAILURES`, `COOLING_PERIOD`, etc. (optional): Tuning parameters

## Deployment

This project uses Docker-only deployment. No traditional deployment scripts or client code are included. Users should:
1. Configure environment variables in `.env`
2. Use `docker-compose up -d` to start
3. Monitor with `docker-compose logs -f`
4. Update with `git pull && docker-compose up -d --build`