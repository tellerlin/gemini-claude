#!/bin/bash
# quick-fix.sh - Quick fix for the current server issues

echo "🔧 Quick Fix for Gemini Adapter Service Issues"
echo "============================================"

APP_DIR="/home/gemini/gemini-claude"
APP_USER="gemini"

# 1. Check if files exist
echo "1. Checking application files..."
if [ -d "$APP_DIR" ]; then
    echo "✅ Application directory exists"
    
    if [ -f "$APP_DIR/src/main.py" ]; then
        echo "✅ Main application file exists"
    else
        echo "❌ Main application file missing"
        exit 1
    fi
    
    if [ -d "$APP_DIR/venv" ]; then
        echo "✅ Virtual environment exists"
    else
        echo "❌ Virtual environment missing"
        exit 1
    fi
    
    if [ -f "$APP_DIR/.env" ]; then
        echo "✅ Environment file exists"
    else
        echo "❌ Environment file missing"
        exit 1
    fi
else
    echo "❌ Application directory missing"
    exit 1
fi

# 2. Check Python environment
echo ""
echo "2. Checking Python environment..."
cd "$APP_DIR"
if sudo -u "$APP_USER" bash -c "source venv/bin/activate && python --version"; then
    echo "✅ Python environment is working"
else
    echo "❌ Python environment issue"
    exit 1
fi

# 3. Check dependencies
echo ""
echo "3. Checking dependencies..."
if sudo -u "$APP_USER" bash -c "source venv/bin/activate && python -c 'import fastapi, uvicorn, litellm'"; then
    echo "✅ All dependencies are installed"
else
    echo "❌ Missing dependencies, installing..."
    sudo -u "$APP_USER" bash -c "source venv/bin/activate && pip install -r requirements.txt"
fi

# 4. Check API keys
echo ""
echo "4. Checking API keys..."
if grep -q "GEMINI_API_KEYS=" "$APP_DIR/.env"; then
    keys=$(grep "GEMINI_API_KEYS=" "$APP_DIR/.env" | cut -d'=' -f2)
    if [ "$keys" != "your_key_1,your_key_2,your_key_3" ] && [ -n "$keys" ]; then
        echo "✅ API keys are configured"
    else
        echo "⚠️  API keys are not properly configured"
        echo "   Please edit $APP_DIR/.env and add your Gemini API keys"
    fi
else
    echo "❌ API keys configuration missing"
fi

# 5. Kill existing processes
echo ""
echo "5. Cleaning up existing processes..."
pkill -f "uvicorn src.main:app" || echo "No existing processes found"

# 6. Try to start the service
echo ""
echo "6. Starting the service..."
cd "$APP_DIR"
if sudo -u "$APP_USER" bash -c "source venv/bin/activate && nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &"; then
    echo "✅ Service start command executed"
    sleep 3
    
    # Check if process is running
    if pgrep -f "uvicorn src.main:app" > /dev/null; then
        echo "✅ Service is running"
        
        # Test health endpoint
        echo "7. Testing service health..."
        if curl -f -s http://localhost:8000/health > /dev/null; then
            echo "✅ Health check passed"
            echo "🌐 Service is available at: http://$(curl -s ifconfig.me 2>/dev/null || echo 'your-server-ip')"
        else
            echo "❌ Health check failed"
            echo "   Checking logs..."
            if [ -f "$APP_DIR/logs/app.log" ]; then
                echo "   Last 10 lines of app.log:"
                tail -10 "$APP_DIR/logs/app.log"
            fi
        fi
    else
        echo "❌ Service failed to start"
        echo "   Checking logs..."
        if [ -f "$APP_DIR/logs/app.log" ]; then
            echo "   Last 10 lines of app.log:"
            tail -10 "$APP_DIR/logs/app.log"
        fi
    fi
else
    echo "❌ Failed to execute start command"
fi

# 8. Fix management script
echo ""
echo "8. Fixing management script..."
if [ -f "/usr/local/bin/gemini-manage" ]; then
    echo "✅ Management script exists"
    echo "   Testing fixed version..."
    
    # Create a simple test
    echo "   Testing: gemini-manage status"
    /usr/local/bin/gemini-manage status
else
    echo "❌ Management script missing"
    echo "   Please re-run deployment script to create it"
fi

echo ""
echo "🎯 Summary:"
echo "=========="
echo "If the service is now running, you can:"
echo "1. Test it manually: curl http://localhost:8000/health"
echo "2. Check logs: tail -f $APP_DIR/logs/app.log"
echo "3. Use management script: gemini-manage status"
echo ""
echo "If it's still not working, check the logs above for errors."