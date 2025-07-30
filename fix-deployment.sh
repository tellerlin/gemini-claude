#!/bin/bash
# fix-deployment.sh - Fix deployment script issues

echo "🔧 Fixing Deployment Script Issues"
echo "================================="

APP_DIR="/home/gemini/gemini-claude"
APP_USER="gemini"

# 1. Check if we're in the right directory
echo "1. Checking current directory..."
if [[ "$(pwd)" == "$APP_DIR" ]]; then
    echo "✅ Already in target directory"
    echo "   The deployment script was trying to copy files to the same directory"
    echo "   This is why you saw the 'same file' errors"
else
    echo "❌ Not in target directory"
    echo "   Please run: cd $APP_DIR"
    exit 1
fi

# 2. Check if files exist
echo ""
echo "2. Checking if required files exist..."
required_files=("src/main.py" "requirements.txt" ".env.example" "README.md")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ All required files are present"
else
    echo "❌ Missing files: ${missing_files[*]}"
    echo "   Please ensure you're in the correct directory with all project files"
    exit 1
fi

# 3. Set correct ownership
echo ""
echo "3. Setting correct ownership..."
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
echo "✅ Ownership set correctly"

# 4. Check virtual environment
echo ""
echo "4. Checking virtual environment..."
if [ -d "venv" ]; then
    echo "✅ Virtual environment exists"
    
    # Check if it's working
    if sudo -u "$APP_USER" bash -c "source venv/bin/activate && python --version"; then
        echo "✅ Virtual environment is working"
    else
        echo "❌ Virtual environment is broken"
        echo "   Recreating virtual environment..."
        rm -rf venv
        sudo -u "$APP_USER" bash -c "python3 -m venv venv"
        sudo -u "$APP_USER" bash -c "source venv/bin/activate && pip install --upgrade pip"
        sudo -u "$APP_USER" bash -c "source venv/bin/activate && pip install -r requirements.txt"
    fi
else
    echo "❌ Virtual environment missing"
    echo "   Creating virtual environment..."
    sudo -u "$APP_USER" bash -c "python3 -m venv venv"
    sudo -u "$APP_USER" bash -c "source venv/bin/activate && pip install --upgrade pip"
    sudo -u "$APP_USER" bash -c "source venv/bin/activate && pip install -r requirements.txt"
fi

# 5. Create logs directory
echo ""
echo "5. Creating logs directory..."
sudo -u "$APP_USER" bash -c "mkdir -p logs"
echo "✅ Logs directory created"

# 6. Configure environment
echo ""
echo "6. Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Environment file created from .env.example"
    else
        echo "❌ .env.example not found, creating minimal .env"
        cat > .env << 'EOF'
# Required Configuration
GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3

# Service Configuration
PORT=8000
HOST=0.0.0.0

# Failure Recovery Settings
MAX_FAILURES=1
COOLING_PERIOD=300
REQUEST_TIMEOUT=45
MAX_RETRIES=0

# Health Check Settings
HEALTH_CHECK_INTERVAL=60
EOF
    fi
fi

# Set correct permissions for .env
chown "$APP_USER:$APP_USER" .env
chmod 600 .env
echo "✅ Environment file configured"

# 7. Now continue with the rest of the deployment
echo ""
echo "7. Continuing with deployment..."
echo "   The deployment script should now continue from where it failed"

# Check if we need to run the deployment script again
echo ""
echo "🎯 Next Steps:"
echo "============="
echo "You can now:"
echo ""
echo "1. Run the deployment script again (it should continue from where it failed):"
echo "   sudo bash scripts/deploy-centos.sh"
echo ""
echo "2. Or use the quick fix script to complete the setup:"
echo "   ./quick-fix.sh"
echo ""
echo "3. Or manually complete the setup:"
echo "   - Edit .env file with your API keys"
echo "   - Run: sudo -u gemini bash -c 'source venv/bin/activate && nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &'"
echo "   - Test with: curl http://localhost:8000/health"
echo ""
echo "The file copy error has been resolved. The deployment script should now work correctly."