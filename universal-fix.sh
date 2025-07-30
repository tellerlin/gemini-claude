#!/bin/bash
# universal-fix.sh - Universal fix script for Gemini Claude Adapter deployment issues

echo "🔧 Gemini Claude Adapter Universal Fix Script"
echo "============================================="

# Configuration
APP_DIR="/home/gemini/gemini-claude"
APP_USER="gemini"
CURRENT_USER=$(whoami)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root for system-wide fixes
if [[ $CURRENT_USER != "root" ]]; then
    log_warning "Not running as root. Some system-wide fixes may not work."
    log_info "For full functionality, run with: sudo bash $0"
    echo ""
fi

# Function to ask yes/no questions
ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# 1. Check current directory and files
echo "1. Checking environment..."
echo "========================"

if [[ "$(pwd)" == "$APP_DIR" ]]; then
    log_success "Already in target directory: $APP_DIR"
else
    log_info "Current directory: $(pwd)"
    log_info "Target directory: $APP_DIR"
    
    if [[ -d "$APP_DIR" ]]; then
        if ask_yes_no "Switch to application directory?"; then
            cd "$APP_DIR" || log_error "Cannot switch to $APP_DIR"
        fi
    else
        log_warning "Target directory doesn't exist. Will create it if needed."
    fi
fi

# Check required files
required_files=("src/main.py" "requirements.txt" ".env.example")
missing_files=()

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -eq 0 ]]; then
    log_success "All required files are present"
else
    log_warning "Missing files: ${missing_files[*]}"
    if ask_yes_no "Try to find project files in parent directories?"; then
        # Look for files in parent directories
        search_dir=$(pwd)
        while [[ "$search_dir" != "/" ]]; do
            found_files=0
            for file in "${missing_files[@]}"; do
                if [[ -f "$search_dir/$file" ]]; then
                    ((found_files++))
                fi
            done
            
            if [[ $found_files -gt 0 ]]; then
                log_info "Found some files in: $search_dir"
                if ask_yes_no "Copy missing files from here?"; then
                    for file in "${missing_files[@]}"; do
                        if [[ -f "$search_dir/$file" ]]; then
                            cp "$search_dir/$file" ./
                            log_success "Copied: $file"
                        fi
                    done
                    break
                fi
            fi
            search_dir=$(dirname "$search_dir")
        done
    fi
fi

# 2. Check and fix user permissions
echo ""
echo "2. Checking user and permissions..."
echo "================================="

if [[ $CURRENT_USER == "root" ]]; then
    # Create application user if it doesn't exist
    if ! id "$APP_USER" &>/dev/null; then
        log_info "Creating application user: $APP_USER"
        useradd -m -s /bin/bash "$APP_USER" || log_warning "Failed to create user $APP_USER"
    else
        log_success "Application user $APP_USER exists"
    fi
    
    # Set up application directory
    mkdir -p "$APP_DIR"
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    log_success "Set correct ownership for $APP_DIR"
else
    log_warning "Running as non-root user. Some permission fixes skipped."
fi

# 3. Check and fix Python environment
echo ""
echo "3. Checking Python environment..."
echo "=============================="

# Find Python command
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    log_error "No Python 3 found on system"
fi

log_info "Using Python: $($PYTHON_CMD --version)"

# Check virtual environment
if [[ ! -d "venv" ]]; then
    log_info "Creating virtual environment..."
    if [[ $CURRENT_USER == "root" ]]; then
        sudo -u "$APP_USER" -H "$PYTHON_CMD" -m venv venv
    else
        "$PYTHON_CMD" -m venv venv
    fi
    log_success "Virtual environment created"
else
    log_info "Virtual environment exists"
    
    # Test if it's working
    if [[ $CURRENT_USER == "root" ]]; then
        if sudo -u "$APP_USER" -H bash -c "source venv/bin/activate && python --version"; then
            log_success "Virtual environment is working"
        else
            log_warning "Virtual environment is broken, recreating..."
            rm -rf venv
            sudo -u "$APP_USER" -H "$PYTHON_CMD" -m venv venv
        fi
    else
        if source venv/bin/activate && python --version; then
            log_success "Virtual environment is working"
        else
            log_warning "Virtual environment is broken, recreating..."
            rm -rf venv
            "$PYTHON_CMD" -m venv venv
        fi
    fi
fi

# Install/upgrade pip and packages
log_info "Installing/upgrading packages..."
if [[ $CURRENT_USER == "root" ]]; then
    sudo -u "$APP_USER" -H bash -c "source venv/bin/activate && pip install --upgrade pip"
    if [[ -f "requirements.txt" ]]; then
        sudo -u "$APP_USER" -H bash -c "source venv/bin/activate && pip install -r requirements.txt"
    fi
else
    source venv/bin/activate
    pip install --upgrade pip
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
fi

log_success "Python packages installed"

# 4. Check and fix environment configuration
echo ""
echo "4. Checking environment configuration..."
echo "===================================="

# Create .env file if missing
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        log_success "Created .env from .env.example"
    else
        log_warning "Creating minimal .env file..."
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
        log_success "Created minimal .env file"
    fi
fi

# Set correct permissions
if [[ $CURRENT_USER == "root" ]]; then
    chown "$APP_USER:$APP_USER" .env
fi
chmod 600 .env
log_success "Environment file configured"

# 5. Check and fix service management
echo ""
echo "5. Checking service management..."
echo "==============================="

# Create logs directory
mkdir -p logs
if [[ $CURRENT_USER == "root" ]]; then
    chown -R "$APP_USER:$APP_USER" logs
fi
log_success "Logs directory created"

# Check if management script exists
if [[ ! -f "manage.sh" ]]; then
    log_info "Creating management script..."
    
    cat > manage.sh << 'EOF'
#!/bin/bash

APP_DIR="/home/gemini/gemini-claude"
APP_USER="gemini"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_service_status() {
    # Try supervisorctl first, fallback to process check
    if command -v supervisorctl &> /dev/null && systemctl is-active --quiet supervisord; then
        local status=$(supervisorctl status gemini-adapter 2>/dev/null | awk '{print $2}')
        echo "$status"
    else
        # Fallback to process checking
        if pgrep -f "uvicorn src.main:app" > /dev/null; then
            echo "RUNNING"
        else
            echo "STOPPED"
        fi
    fi
}

case "$1" in
    start) 
        log_info "Starting Gemini Adapter service..."
        if command -v supervisorctl &> /dev/null && systemctl is-active --quiet supervisord; then
            if supervisorctl start gemini-adapter; then
                log_success "Service started successfully via Supervisor"
            else
                log_error "Failed to start service via Supervisor"
                exit 1
            fi
        else
            # Fallback to manual start
            cd "$APP_DIR"
            if sudo -u "$APP_USER" -H bash -c "source venv/bin/activate && nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &"; then
                log_success "Service started successfully in background"
            else
                log_error "Failed to start service"
                exit 1
            fi
        fi
        ;;
    stop) 
        log_info "Stopping Gemini Adapter service..."
        if command -v supervisorctl &> /dev/null && systemctl is-active --quiet supervisord; then
            if supervisorctl stop gemini-adapter; then
                log_success "Service stopped successfully via Supervisor"
            else
                log_error "Failed to stop service via Supervisor"
                exit 1
            fi
        else
            # Fallback to manual stop
            if pkill -f "uvicorn src.main:app"; then
                log_success "Service stopped successfully"
            else
                log_warning "No running process found"
            fi
        fi
        ;;
    restart) 
        log_info "Restarting Gemini Adapter service..."
        if command -v supervisorctl &> /dev/null && systemctl is-active --quiet supervisord; then
            if supervisorctl restart gemini-adapter; then
                log_success "Service restarted successfully via Supervisor"
            else
                log_error "Failed to restart service via Supervisor"
                exit 1
            fi
        else
            # Fallback to manual restart
            pkill -f "uvicorn src.main:app" || true
            sleep 2
            cd "$APP_DIR"
            if sudo -u "$APP_USER" -H bash -c "source venv/bin/activate && nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &"; then
                log_success "Service restarted successfully in background"
            else
                log_error "Failed to restart service"
                exit 1
            fi
        fi
        ;;
    status) 
        echo "=== Service Status ==="
        status=$(check_service_status)
        echo "Service status: $status"
        
        if command -v supervisorctl &> /dev/null && systemctl is-active --quiet supervisord; then
            echo -e "\n=== Supervisor Status ==="
            supervisorctl status gemini-adapter
        fi
        
        echo -e "\n=== Process Information ==="
        if pgrep -f "uvicorn src.main:app" > /dev/null; then
            echo "Running processes:"
            ps aux | grep '[u]vicorn' | awk '{print "  PID: " $2 ", CPU: " $3 "%, Memory: " $4 "%"}'
        else
            echo "No running processes found"
        fi
        
        echo -e "\n=== Service Health Check ==="
        if command -v curl &> /dev/null; then
            if curl -f -s http://localhost:8000/health > /dev/null; then
                echo "Health check: PASSED"
                echo "Service URL: http://localhost:8000"
            else
                echo "Health check: FAILED"
            fi
        else
            echo "Health check: curl not available"
        fi
        ;;
    logs) 
        log_info "Showing application logs (Ctrl+C to exit)..."
        tail -f logs/app.log
        ;;
    error-logs)
        log_info "Showing error logs (Ctrl+C to exit)..."
        tail -f logs/error.log
        ;;
    *) 
        echo "Gemini Claude Adapter Management Script"
        echo "Usage: $0 {start|stop|restart|status|logs|error-logs}"
        echo ""
        echo "Commands:"
        echo "  start       - Start the service"
        echo "  stop        - Stop the service"
        echo "  restart     - Restart the service"
        echo "  status      - Show service status"
        echo "  logs        - Show application logs"
        echo "  error-logs  - Show error logs"
        exit 1
        ;;
esac
EOF
    
    chmod +x manage.sh
    if [[ $CURRENT_USER == "root" ]]; then
        chown "$APP_USER:$APP_USER" manage.sh
        ln -sf "$APP_DIR/manage.sh" /usr/local/bin/gemini-manage
        log_success "Management script created and linked to 'gemini-manage'"
    else
        log_success "Management script created (run as root to create system-wide link)"
    fi
else
    log_success "Management script already exists"
fi

# 6. System-level fixes (root only)
if [[ $CURRENT_USER == "root" ]]; then
    echo ""
    echo "6. Applying system-level fixes..."
    echo "================================"
    
    # Try to configure supervisor if available
    if command -v supervisorctl &> /dev/null; then
        log_info "Configuring Supervisor..."
        
        # Enable and start supervisor
        systemctl enable supervisord || log_warning "Failed to enable supervisor"
        systemctl start supervisord || log_warning "Failed to start supervisor"
        
        # Create supervisor config
        cat > /etc/supervisord.d/gemini-adapter.conf << EOF
[program:gemini-adapter]
command=$APP_DIR/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
directory=$APP_DIR
user=$APP_USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$APP_DIR/logs/app.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
stderr_logfile=$APP_DIR/logs/error.log
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=10
environment=PATH="$APP_DIR/venv/bin",PYTHONUNBUFFERED=1
stopsignal=INT
stopwaitsecs=10
killasgroup=true
priority=999
EOF
        
        # Reload supervisor
        supervisorctl reread || log_warning "Supervisor reread failed"
        supervisorctl update || log_warning "Supervisor update failed"
        
        log_success "Supervisor configured"
    else
        log_warning "Supervisor not available, will use manual process management"
    fi
    
    # Configure firewall if available
    if command -v ufw &> /dev/null; then
        log_info "Configuring UFW firewall..."
        ufw allow http || log_warning "Failed to allow HTTP"
        ufw allow https || log_warning "Failed to allow HTTPS"
        log_success "Firewall configured"
    elif command -v firewall-cmd &> /dev/null; then
        log_info "Configuring firewalld..."
        firewall-cmd --permanent --add-service=http || log_warning "Failed to allow HTTP"
        firewall-cmd --permanent --add-service=https || log_warning "Failed to allow HTTPS"
        firewall-cmd --reload || log_warning "Failed to reload firewall"
        log_success "Firewall configured"
    fi
fi

# 7. Final diagnostics and next steps
echo ""
echo "7. Final diagnostics..."
echo "====================="

# Check if service can be started
if [[ -f "src/main.py" && -d "venv" && -f ".env" ]]; then
    log_info "Testing service startup..."
    
    # Check if .env has API keys
    if grep -q "your_key_" .env; then
        log_warning "API keys not configured in .env file"
        log_info "Please edit .env file and add your Gemini API keys"
    fi
    
    if ask_yes_no "Try to start the service now?"; then
        if [[ $CURRENT_USER == "root" ]]; then
            gemini-manage restart
        else
            # Manual start for non-root users
            source venv/bin/activate
            nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &
            sleep 3
            if pgrep -f "uvicorn src.main:app" > /dev/null; then
                log_success "Service started successfully"
            else
                log_error "Service failed to start. Check logs/app.log for details."
            fi
        fi
    fi
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo ""
echo "1. 📝 Configure your API keys:"
echo "   nano .env"
echo "   # Replace 'your_key_1,your_key_2,your_key_3' with your actual Gemini API keys"
echo ""
echo "2. 🚀 Start the service:"
echo "   gemini-manage restart  # if running as root"
echo "   # or manual start: source venv/bin/activate && python -m uvicorn src.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "3. ✅ Verify everything works:"
echo "   gemini-manage status"
echo "   curl http://localhost:8000/health"
echo ""
echo "4. 📚 Management commands:"
echo "   gemini-manage {start|stop|restart|status|logs|error-logs}"
echo ""
echo "🎉 Universal fix completed! Most common issues should now be resolved."