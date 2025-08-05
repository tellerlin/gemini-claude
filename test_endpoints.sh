#!/bin/bash

# ==============================================================================
# Gemini Claude Adapter v2.1.0 - Functional Test Script
#
# Instructions:
# 1. Replace the placeholder values for CLIENT_KEY and ADMIN_KEY below.
# 2. Make the script executable: chmod +x test_endpoints.sh
# 3. Run the script: ./test_endpoints.sh
# ==============================================================================

# --- CONFIGURATION ---
# ‼️ IMPORTANT: Replace these with your actual keys from the .env file.
export CLIENT_KEY="your-client-key"
export ADMIN_KEY="your-admin-key"

# The base URL of your running service
BASE_URL="http://localhost:8000"

# --- SCRIPT LOGIC (No need to edit below) ---

# Function to print a formatted header
print_header() {
    echo ""
    echo "============================================================"
    echo "▶️  Testing: $1"
    echo "============================================================"
}

# Check if keys are placeholders
if [ "$CLIENT_KEY" = "your-client-key" ] || [ "$ADMIN_KEY" = "your-admin-key" ]; then
    echo "❌ ERROR: Please edit this script and replace the placeholder API keys."
    exit 1
fi

# --- 1. Service Status and Public Endpoints ---
print_header "Service Status (No Authentication)"
echo "--> Checking Docker container status..."
docker-compose ps
echo -e "\n--> Pinging root endpoint..."
curl -s ${BASE_URL}/ | jq
echo -e "\n--> Pinging health endpoint..."
curl -s ${BASE_URL}/health | jq

# --- 2. Core API Functionality (Client Key required) ---
print_header "Core API Functionality (Client Key)"
echo "--> Listing available models..."
curl -s "${BASE_URL}/v1/models" -H "Authorization: Bearer $CLIENT_KEY" | jq
echo -e "\n--> Simple Q&A test (claude-3-5-sonnet)..."
curl -s -X POST "${BASE_URL}/v1/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello! In simple terms, what is machine learning?"}]
  }' | jq
echo -e "\n--> Multi-turn conversation test..."
curl -s -X POST "${BASE_URL}/v1/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "What are the main benefits of Python programming?"},
      {"role": "assistant", "content": "Python has many benefits, like its simple syntax, versatility, and a huge ecosystem of libraries."},
      {"role": "user", "content": "Which benefit is most important for data science?"}
    ]
  }' | jq

# --- 3. Monitoring & Stats Endpoints (Client Key required) ---
print_header "Monitoring & Stats Endpoints (Client Key)"
echo "--> Getting key usage stats..."
curl -s "${BASE_URL}/stats" -H "Authorization: Bearer $CLIENT_KEY" | jq
echo -e "\n--> Getting performance metrics..."
curl -s "${BASE_URL}/metrics" -H "Authorization: Bearer $CLIENT_KEY" | jq
echo -e "\n--> Getting detailed health status..."
curl -s "${BASE_URL}/health/detailed" -H "Authorization: Bearer $CLIENT_KEY" | jq
echo -e "\n--> Getting cache stats..."
curl -s "${BASE_URL}/cache/stats" -H "Authorization: Bearer $CLIENT_KEY" | jq

# --- 4. Admin Endpoints (Admin Key required) ---
print_header "Admin Endpoints (Admin Key)"
echo "--> Checking security status..."
curl -s "${BASE_URL}/admin/security-status" -H "Authorization: Bearer $ADMIN_KEY" | jq
echo -e "\n--> Clearing cache..."
curl -s -X POST "${BASE_URL}/cache/clear" -H "Authorization: Bearer $ADMIN_KEY" | jq
echo -e "\n--> Verifying cache is cleared..."
curl -s "${BASE_URL}/cache/stats" -H "Authorization: Bearer $CLIENT_KEY" | jq
echo -e "\n--> Viewing recent errors (should be empty or have old errors)..."
curl -s "${BASE_URL}/errors/recent" -H "Authorization: Bearer $ADMIN_KEY" | jq

echo ""
print_header "All tests completed!"
echo "If you see JSON output for each test without errors, your deployment is successful."
echo "============================================================"
echo ""
