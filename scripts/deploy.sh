#!/bin/bash
# deploy.sh - Automated deployment script for Gemini Claude Adapter

set -euo pipefail # Exit on error, undefined variables, and pipe failures

# --- Configuration ---
readonly APP_USER="gemini"
readonly APP_DIR="/home/$APP_USER/gemini-claude"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/tmp/gemini_deployment.log"

# --- Helper Functions ---
log_info() { 
    echo -e "\033[0;34m[INFO]\033[0m $1" | tee -a "$LOG_FILE"
}

log_success() { 
    echo -e "\033[0;32m[SUCCESS]\033[0m $1" | tee -a "$LOG_FILE"
}

log_warning() { 
    echo -e "\033[1;33m[WARNING]\033[0m $1" | tee -a "$LOG_FILE"
}

log_error() { 
    echo -e "\033[0;31m[ERROR]\033[0m $1" | tee -a "$LOG_FILE"
    exit 1
}

# --- Validation Functions ---
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root. Please use 'sudo bash $0'."
    fi
}

check_system() {
    if ! command -v apt-get &> /dev/null; then
        log_error "This script currently only supports Debian-based systems (like Ubuntu)."
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "src/main.py" ]]; then
        log_error "Could not find main application file. Please run this script from the project's root directory."
    fi
}

# --- Installation Functions ---
install_dependencies() {
    log_info "Updating system and installing dependencies..."
    
    # Update package list
    apt-get update || log_error "Failed to update package list"
    
    # Install essential packages
    apt-get install -y software-properties-common curl wget gnupg2 lsb-release || \
        log_error "Failed to install essential packages"
    
    # Install Python 3.11 if not available
    if ! command -v python3.11 &> /dev/null; then
        log_info "Installing Python 3.11..."
        add-apt-repository ppa:deadsnakes/ppa -y || log_error "Failed to add Python PPA"
        apt-get update || log_error "Failed to update after adding PPA"
    fi
    
    # Install all required packages
    local packages=(
        python3.11
        python3.11-venv
        python3.11-dev
        python3-pip
        git
        nginx
        supervisor
        ufw
        htop
        unzip
    )
    
    for package in "${packages[@]}"; do
        log_info "Installing $package..."
        apt-get install -y "$package" || log_error "Failed to install $package"
    done
    
    # Ensure python3.11 is available as python3
    if ! command -v python3 &> /dev/null || [[ "$(python3 --version)" != *"3.11"* ]]; then
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 || \
            log_warning "Failed to set python3.11 as default python3"
    fi
    
    log_success "Dependencies installed successfully."
}

setup_app_user() {
    log_info "Setting up application user..."
    
    if ! id "$APP_USER" &>/dev/null; then
        useradd -m -s /bin/bash "$APP_USER" || log_error "Failed to create user $APP_USER"
        log_info "Created application user '$APP_USER'."
    else
        log_info "Application user '$APP_USER' already exists."
    fi
    
    # Ensure home directory exists with correct permissions
    mkdir -p "/home/$APP_USER"
    chown "$APP_USER:$APP_USER" "/home/$APP_USER"
    chmod 755 "/home/$APP_USER"
}

setup_app_environment() {
    log_info "Setting up application environment..."
    
    # Create application directory
    mkdir -p "$APP_DIR"
    
    # Check if git is available and repository URL is provided
    if command -v git &> /dev/null && [[ -n "${GITHUB_REPO_URL:-}" ]]; then
        log_info "Cloning from GitHub repository: $GITHUB_REPO_URL"
        sudo -u "$APP_USER" -H git clone "$GITHUB_REPO_URL" "$APP_DIR" || log_error "Failed to clone repository"
        log_success "Repository cloned to $APP_DIR."
    else
        # Copy application files (fallback method)
        log_info "Copying application files..."
        # SCRIPT_DIR is project-root/scripts, we need to copy from parent directory
        cp -r "$(dirname "$SCRIPT_DIR")"/* "$APP_DIR/" || log_error "Failed to copy application files"
        log_success "Application files copied to $APP_DIR."
    fi
    
    # Set correct ownership
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    
    # Create Python virtual environment
    log_info "Creating Python virtual environment..."
    sudo -u "$APP_USER" -H python3.11 -m venv "$APP_DIR/venv" || \
        log_error "Failed to create virtual environment"
    
    # Upgrade pip
    sudo -u "$APP_USER" -H "$APP_DIR/venv/bin/pip" install --upgrade pip || \
        log_error "Failed to upgrade pip"
    
    # Install Python packages
    log_info "Installing Python packages..."
    sudo -u "$APP_USER" -H "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" || \
        log_error "Failed to install Python packages"
    
    # Create logs directory
    sudo -u "$APP_USER" -H mkdir -p "$APP_DIR/logs"
    
    log_success "Python environment created successfully."
}

create_env_file() {
    log_info "Creating .env template..."
    local env_file="$APP_DIR/.env"

    if [[ ! -f "$env_file" ]]; then
        cat > "$env_file" << 'EOF'
# --- Required Configuration ---
# Add your Gemini API keys here, separated by commas
# Get your keys from: https://makersuite.google.com/app/apikey
GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3

# --- Optional Proxy Configuration ---
# Uncomment and set if you need to use a proxy
# PROXY_URL=http://username:password@proxy-server:port

# --- Service Configuration ---
# Port for the internal service (should match nginx config)
PORT=8000
HOST=0.0.0.0

# --- Failure Recovery Settings ---
# Number of consecutive failures before a key is put in cooling
MAX_FAILURES=3
# Cooldown duration in seconds (5 minutes = 300)
COOLING_PERIOD=300
# Request timeout in seconds (higher for better reliability)
REQUEST_TIMEOUT=45
# Number of retries with different keys if a request fails
MAX_RETRIES=2

# --- Health Check Settings ---
# How often to check key status (in seconds)
HEALTH_CHECK_INTERVAL=60
EOF
        chown "$APP_USER:$APP_USER" "$env_file"
        chmod 600 "$env_file"  # Secure permissions for sensitive data
        log_success ".env template created."
    else
        log_info ".env file already exists, skipping creation."
    fi
}

configure_supervisor() {
    log_info "Configuring Supervisor..."
    
    local supervisor_conf="/etc/supervisor/conf.d/gemini-adapter.conf"
    
    cat > "$supervisor_conf" << EOF
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

    # Test supervisor configuration
    if ! supervisorctl reread 2>/dev/null; then
        log_error "Failed to read supervisor configuration"
    fi
    
    if ! supervisorctl update 2>/dev/null; then
        log_error "Failed to update supervisor"
    fi
    
    log_success "Supervisor configured successfully."
}

configure_nginx() {
    log_info "Configuring Nginx..."
    
    # Remove default site
    rm -f /etc/nginx/sites-enabled/default
    
    # Create rate limiting configuration
    local rate_limit_conf="/etc/nginx/conf.d/rate-limit.conf"
    cat > "$rate_limit_conf" << 'EOF'
# Rate limiting configuration
limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
EOF

    local nginx_conf="/etc/nginx/sites-available/gemini-adapter"
    
    cat > "$nginx_conf" << 'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # Rate limiting
    limit_req zone=api burst=60 nodelay;

    client_max_body_size 10M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Settings for streaming responses
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Handle WebSocket upgrades if needed
        # proxy_set_header Upgrade $http_upgrade;
        # proxy_set_header Connection "upgrade";
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    # Custom error pages
    error_page 502 503 504 /50x.html;
    location = /50x.html {
        root /var/www/html;
        internal;
    }
}
EOF

    # Create custom error page
    mkdir -p /var/www/html
    cat > /var/www/html/50x.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Service Temporarily Unavailable</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        h1 { color: #e74c3c; }
        p { color: #666; }
    </style>
</head>
<body>
    <h1>Service Temporarily Unavailable</h1>
    <p>The Gemini Claude Adapter is temporarily unavailable. Please try again in a few moments.</p>
    <p>If the problem persists, please contact your administrator.</p>
</body>
</html>
EOF

    # Enable the site
    ln -sf "$nginx_conf" /etc/nginx/sites-enabled/
    
    # Test nginx configuration
    if ! nginx -t; then
        log_error "Nginx configuration test failed"
    fi
    
    # Enable and restart nginx
    systemctl enable nginx || log_error "Failed to enable nginx"
    systemctl restart nginx || log_error "Failed to restart nginx"
    
    log_success "Nginx configured successfully."
}

configure_firewall() {
    log_info "Configuring firewall (UFW)..."
    
    # Reset UFW to default settings
    ufw --force reset > /dev/null 2>&1
    
    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow essential services
    ufw allow ssh comment 'SSH access'
    ufw allow http comment 'HTTP access'
    ufw allow https comment 'HTTPS access'
    
    # Enable UFW
    ufw --force enable
    
    log_success "Firewall configured successfully."
}

create_management_script() {
    log_info "Creating management script..."
    
    local manage_script="$APP_DIR/manage.sh"
    
    cat > "$manage_script" << 'EOF'
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
    local status=$(supervisorctl status gemini-adapter 2>/dev/null | awk '{print $2}')
    echo "$status"
}

case "$1" in
    start) 
        log_info "Starting Gemini Adapter service..."
        if supervisorctl start gemini-adapter; then
            log_success "Service started successfully"
        else
            log_error "Failed to start service"
            exit 1
        fi
        ;;
    stop) 
        log_info "Stopping Gemini Adapter service..."
        if supervisorctl stop gemini-adapter; then
            log_success "Service stopped successfully"
        else
            log_error "Failed to stop service"
            exit 1
        fi
        ;;
    restart) 
        log_info "Restarting Gemini Adapter service..."
        if supervisorctl restart gemini-adapter; then
            log_success "Service restarted successfully"
        else
            log_error "Failed to restart service"
            exit 1
        fi
        ;;
    status) 
        echo "=== Supervisor Status ==="
        supervisorctl status gemini-adapter
        
        echo -e "\n=== Service Health Check ==="
        if command -v python3 &> /dev/null; then
            timeout 10 python3 -c "
import httpx
import json
try:
    response = httpx.get('http://localhost/health', timeout=5)
    data = response.json()
    print(f'HTTP {response.status_code}')
    print(f'Status: {data.get(\"status\", \"unknown\")}')
    print(f'Active Keys: {data.get(\"active_keys\", 0)}/{data.get(\"total_keys\", 0)}')
except Exception as e:
    print(f'Health check failed: {e}')
" 2>/dev/null || log_warning "Health check timeout or failed"
        fi
        
        echo -e "\n=== System Resources ==="
        echo "Memory usage:"
        ps aux | grep '[u]vicorn' | awk '{print "  CPU: " $3 "%, Memory: " $4 "%, PID: " $2}'
        echo "Disk usage:"
        df -h "$APP_DIR" | tail -1 | awk '{print "  " $1 ": " $3 "/" $2 " (" $5 " used)"}'
        ;;
    logs) 
        log_info "Showing application logs (Ctrl+C to exit)..."
        tail -f "$APP_DIR/logs/app.log"
        ;;
    error-logs)
        log_info "Showing error logs (Ctrl+C to exit)..."
        tail -f "$APP_DIR/logs/error.log"
        ;;
    update)
        log_info "Updating Python packages..."
        sudo -u "$APP_USER" -H "$APP_DIR/venv/bin/pip" install --upgrade -r "$APP_DIR/requirements.txt"
        log_success "Packages updated. Restart the service to apply changes."
        ;;
    backup)
        local backup_dir="/tmp/gemini-adapter-backup-$(date +%Y%m%d-%H%M%S)"
        log_info "Creating backup at $backup_dir..."
        mkdir -p "$backup_dir"
        cp -r "$APP_DIR"/{*.py,*.txt,.env,logs} "$backup_dir/" 2>/dev/null || true
        tar -czf "$backup_dir.tar.gz" -C /tmp "$(basename "$backup_dir")"
        rm -rf "$backup_dir"
        log_success "Backup created: $backup_dir.tar.gz"
        ;;
    *) 
        echo "Gemini Claude Adapter Management Script"
        echo "Usage: $0 {start|stop|restart|status|logs|error-logs|update|backup}"
        echo ""
        echo "Commands:"
        echo "  start       - Start the service"
        echo "  stop        - Stop the service"
        echo "  restart     - Restart the service (use after config changes)"
        echo "  status      - Show detailed service status and health"
        echo "  logs        - Show and follow application logs"
        echo "  error-logs  - Show and follow error logs"
        echo "  update      - Update Python packages"
        echo "  backup      - Create a backup of configuration and logs"
        echo ""
        echo "Examples:"
        echo "  $0 restart    # After changing .env file"
        echo "  $0 status     # Check if everything is working"
        echo "  $0 logs       # Debug issues"
        exit 1
        ;;
esac
EOF
    
    chmod +x "$manage_script"
    chown "$APP_USER:$APP_USER" "$manage_script"
    
    # Create system-wide symlink
    ln -sf "$manage_script" /usr/local/bin/gemini-manage
    
    log_success "Management script created and linked to 'gemini-manage' command."
}

perform_system_optimizations() {
    log_info "Applying system optimizations..."
    
    # Increase file descriptor limits
    cat >> /etc/security/limits.conf << 'EOF'
# Gemini Adapter optimizations
gemini soft nofile 32768
gemini hard nofile 32768
EOF

    # Configure systemd service limits
    mkdir -p /etc/systemd/system/supervisor.service.d
    cat > /etc/systemd/system/supervisor.service.d/override.conf << 'EOF'
[Service]
LimitNOFILE=65536
EOF

    # Reload systemd
    systemctl daemon-reload
    
    # Configure kernel parameters for better networking
    cat >> /etc/sysctl.conf << 'EOF'
# Gemini Adapter networking optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 5000
EOF
    
    # Apply sysctl changes
    sysctl -p > /dev/null 2>&1 || log_warning "Some sysctl parameters could not be applied"
    
    log_success "System optimizations applied."
}

run_post_installation_tests() {
    log_info "Running post-installation tests..."
    
    # Test if the service can start
    if ! supervisorctl start gemini-adapter 2>/dev/null; then
        log_warning "Service failed to start automatically. This is expected if API keys are not configured yet."
    else
        sleep 5
        # Test basic connectivity
        if curl -f -s http://localhost:8000/ > /dev/null; then
            log_success "Service is responding to HTTP requests"
        else
            log_warning "Service is not responding. Check logs after configuring API keys."
        fi
        supervisorctl stop gemini-adapter 2>/dev/null
    fi
    
    # Test nginx configuration
    if curl -f -s http://localhost/ > /dev/null; then
        log_success "Nginx is properly configured and running"
    else
        log_warning "Nginx may not be properly configured"
    fi
}

cleanup_installation() {
    log_info "Cleaning up installation files..."
    
    # Clean up package cache
    apt-get autoremove -y > /dev/null 2>&1
    apt-get autoclean > /dev/null 2>&1
    
    # Set final permissions
    chmod -R 755 "$APP_DIR"
    chmod 600 "$APP_DIR/.env"
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    
    log_success "Installation cleanup completed."
}

# --- Main Installation Flow ---
main() {
    local start_time=$(date +%s)
    
    log_info "Starting Gemini Claude Adapter deployment..."
    echo "Deployment log: $LOG_FILE"
    echo "Start time: $(date)"
    echo ""
    
    # Pre-installation checks
    check_root
    check_system
    
    # Installation steps
    install_dependencies
    setup_app_user
    setup_app_environment
    create_env_file
    configure_supervisor
    configure_nginx
    configure_firewall
    create_management_script
    perform_system_optimizations
    run_post_installation_tests
    cleanup_installation
    
    # Get server IP
    local server_ip
    server_ip=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "Unable to detect")
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Final success message
    echo ""
    log_success "🎉 Gemini Claude Adapter deployment completed successfully!"
    echo ""
    echo "⏱️  Total deployment time: ${duration} seconds"
    echo ""
    log_info "⚠️  IMPORTANT: Complete these final steps:"
    echo ""
    echo "1. 📝 Configure your API keys:"
    echo "   sudo nano $APP_DIR/.env"
    echo "   # Add your Gemini API keys to GEMINI_API_KEYS="
    echo ""
    echo "2. 🚀 Start the service:"
    echo "   gemini-manage restart"
    echo ""
    echo "3. ✅ Verify everything works:"
    echo "   gemini-manage status"
    echo ""
    echo "🌐 Your adapter will be available at:"
    echo "   http://$server_ip"
    echo "   Health check: http://$server_ip/health"
    echo ""
    echo "📚 Management commands:"
    echo "   gemini-manage {start|stop|restart|status|logs}"
    echo ""
    echo "📋 Next steps:"
    echo "   - Configure your IDE/Claude Code to use: http://$server_ip/v1"
    echo "   - Use any non-empty string as API key in your client"
    echo ""
    echo "💡 Alternative deployment method (using git clone):"
    echo "   GITHUB_REPO_URL=https://github.com/tellerlin/gemini-claude.git sudo bash scripts/deploy.sh"
    echo ""
    log_info "Installation complete! Check the deployment log at: $LOG_FILE"
}

# Error handling
trap 'log_error "Deployment failed. Check $LOG_FILE for details."' ERR

# Run main installation
main "$@"