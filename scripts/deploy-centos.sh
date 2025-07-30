#!/bin/bash
# deploy-centos.sh - Automated deployment script for Gemini Claude Adapter (CentOS/RHEL)

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
    if ! command -v yum &> /dev/null && ! command -v dnf &> /dev/null; then
        log_error "This script currently only supports CentOS/RHEL-based systems."
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "src/main.py" ]]; then
        log_error "Could not find main application file. Please run this script from the project's root directory."
    fi
}

# --- Installation Functions ---
install_dependencies() {
    log_info "Updating system and installing dependencies..."
    
    # Determine package manager
    if command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
    else
        PKG_MANAGER="yum"
    fi
    
    # Update package list
    $PKG_MANAGER update -y || log_error "Failed to update package list"
    
    # Install EPEL repository
    $PKG_MANAGER install -y epel-release || log_error "Failed to install EPEL repository"
    
    # Install essential packages (excluding nginx for now)
    local packages=(
        python3
        python3-devel
        python3-pip
        git
        supervisor
        policycoreutils-python-utils
        htop
        unzip
        wget
        curl
    )
    
    for package in "${packages[@]}"; do
        log_info "Installing $package..."
        $PKG_MANAGER install -y "$package" || log_error "Failed to install $package"
    done
    
    # Install nginx from EPEL
    log_info "Installing nginx from EPEL..."
    if ! $PKG_MANAGER install -y nginx; then
        log_warning "Failed to install nginx from EPEL, trying alternative method..."
        
        # Check and remove exclude filters that might block nginx
        log_info "Checking for repository exclude filters..."
        if command -v dnf &> /dev/null; then
            # Try to install with disableexcludes
            if ! dnf install -y nginx --disableexcludes=all; then
                log_warning "Failed to install with disableexcludes, trying PowerTools..."
                
                # Try to enable PowerTools repository (for CentOS Stream 8)
                dnf config-manager --enable powertools || log_warning "Failed to enable PowerTools"
                
                # Try installing nginx again
                if ! dnf install -y nginx --disableexcludes=all; then
                    log_warning "Still failed to install nginx, adding official nginx repository..."
                    
                    # Add official nginx repository
                    cat > /etc/yum.repos.d/nginx.repo << 'EOF'
[nginx-stable]
name=nginx stable repo
baseurl=http://nginx.org/packages/centos/$releasever/$basearch/
gpgcheck=0
enabled=1
exclude=
EOF
                    
                    # Update package cache and try again
                    dnf update -y
                    dnf install -y nginx --disableexcludes=all || log_error "Failed to install nginx. Please check repository configuration."
                fi
            fi
        else
            # For yum systems
            if ! yum install -y nginx --disableexcludes=all; then
                log_warning "Failed to install with disableexcludes, trying PowerTools..."
                
                # Try to enable PowerTools repository
                yum-config-manager --enable powertools || log_warning "Failed to enable PowerTools"
                
                # Try installing nginx again
                if ! yum install -y nginx --disableexcludes=all; then
                    log_warning "Still failed to install nginx, adding official nginx repository..."
                    
                    # Add official nginx repository
                    cat > /etc/yum.repos.d/nginx.repo << 'EOF'
[nginx-stable]
name=nginx stable repo
baseurl=http://nginx.org/packages/centos/$releasever/$basearch/
gpgcheck=0
enabled=1
exclude=
EOF
                    
                    # Update package cache and try again
                    yum update -y
                    yum install -y nginx --disableexcludes=all || log_error "Failed to install nginx. Please check repository configuration."
                fi
            fi
        fi
    fi
    
    # Install Python 3.11 if not available
    if ! python3.11 --version &> /dev/null 2>&1; then
        log_info "Installing Python 3.11..."
        
        # Try different methods to install Python 3.11
        if $PKG_MANAGER install -y python3.11 python3.11-devel python3.11-pip &> /dev/null 2>&1; then
            log_success "Python 3.11 installed from EPEL"
        else
            log_warning "Python 3.11 not available in EPEL, trying alternative methods..."
            
            # Try from SCL (Software Collections)
            if $PKG_MANAGER install -y centos-release-scl &> /dev/null 2>&1; then
                $PKG_MANAGER update -y
                if $PKG_MANAGER install -y rh-python311 &> /dev/null 2>&1; then
                    log_success "Python 3.11 installed from SCL"
                    # Create symlinks for easier access
                    ln -sf /opt/rh/rh-python311/root/bin/python3.11 /usr/bin/python3.11 || log_warning "Failed to create python3.11 symlink"
                    ln -sf /opt/rh/rh-python311/root/bin/pip3.11 /usr/bin/pip3.11 || log_warning "Failed to create pip3.11 symlink"
                else
                    log_warning "Failed to install Python 3.11 from SCL, will use system Python 3.6"
                fi
            else
                log_warning "Cannot install SCL, will use system Python 3.6"
            fi
        fi
    fi
    
    # Check if we have at least Python 3.6
    if ! python3 --version &> /dev/null 2>&1; then
        log_error "No Python 3 found on system"
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
        # Check if we're running from the target directory
        if [[ "$(pwd)" == "$APP_DIR" ]]; then
            log_info "Already in target directory, skipping file copy"
            log_success "Application files are already in place."
        else
            # SCRIPT_DIR is project-root/scripts, we need to copy from parent directory
            local source_dir="$(dirname "$SCRIPT_DIR")"
            if [[ -d "$source_dir" && "$source_dir" != "$APP_DIR" ]]; then
                log_info "Copying application files from $source_dir to $APP_DIR..."
                # Copy all files except the target directory itself
                cp -r "$source_dir"/* "$APP_DIR/" || log_error "Failed to copy application files"
                log_success "Application files copied to $APP_DIR."
            else
                log_info "Source directory is same as target directory, skipping file copy"
                log_success "Application files are already in place."
            fi
        fi
    fi
    
    # Set correct ownership
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    
    # Create Python virtual environment
    log_info "Creating Python virtual environment..."
    
    # Determine Python command (prefer 3.11, fallback to 3.x)
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        log_info "Using Python 3.11 for virtual environment"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        log_info "Using system Python 3 for virtual environment"
    else
        log_error "No Python 3 found on system"
    fi
    
    # Show Python version
    log_info "Python version: $($PYTHON_CMD --version)"
    
    sudo -u "$APP_USER" -H "$PYTHON_CMD" -m venv "$APP_DIR/venv" || \
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
    log_info "Setting up environment configuration..."
    local env_file="$APP_DIR/.env"
    local env_example="$APP_DIR/.env.example"

    # Copy .env.example to .env if .env doesn't exist
    if [[ ! -f "$env_file" ]]; then
        if [[ -f "$env_example" ]]; then
            log_info "Copying .env.example to .env..."
            cp "$env_example" "$env_file"
            log_success ".env file created from .env.example."
        else
            log_warning ".env.example not found, creating minimal .env template..."
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
MAX_FAILURES=1
# Cooldown duration in seconds (5 minutes = 300)
COOLING_PERIOD=300
# Request timeout in seconds (higher for better reliability)
REQUEST_TIMEOUT=45
# Number of retries with different keys if a request fails
MAX_RETRIES=0

# --- Health Check Settings ---
# How often to check key status (in seconds)
HEALTH_CHECK_INTERVAL=60
EOF
            log_success ".env template created."
        fi
    else
        log_info ".env file already exists, skipping creation."
    fi
    
    # Set secure permissions
    chown "$APP_USER:$APP_USER" "$env_file"
    chmod 600 "$env_file"  # Secure permissions for sensitive data
}

configure_supervisor() {
    log_info "Configuring Supervisor..."
    
    # Enable and start supervisor
    systemctl enable supervisord || log_warning "Failed to enable supervisor"
    systemctl start supervisord || log_warning "Failed to start supervisor"
    
    # Wait for supervisor to start
    sleep 5
    
    # Check if supervisor is running
    if ! systemctl is-active --quiet supervisord; then
        log_warning "Supervisor failed to start, will use manual management"
        return 0
    fi
    
    local supervisor_conf="/etc/supervisord.d/gemini-adapter.conf"
    
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

    # Wait a moment for supervisor to detect the new config
    sleep 2
    
    # Test supervisor configuration
    if command -v supervisorctl &> /dev/null; then
        if supervisorctl reread && supervisorctl update; then
            log_success "Supervisor configured successfully."
        else
            log_warning "Supervisor configuration failed, will use manual management"
        fi
    else
        log_warning "supervisorctl not available, will use manual management"
    fi
}

configure_nginx() {
    log_info "Configuring Nginx..."
    
    # Enable and start nginx
    systemctl enable nginx || log_error "Failed to enable nginx"
    systemctl start nginx || log_error "Failed to start nginx"
    
    # Remove default site
    rm -f /etc/nginx/conf.d/default.conf
    
    # Create rate limiting configuration
    local rate_limit_conf="/etc/nginx/conf.d/rate-limit.conf"
    cat > "$rate_limit_conf" << 'EOF'
# Rate limiting configuration
limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
EOF

    local nginx_conf="/etc/nginx/conf.d/gemini-adapter.conf"
    
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
        root /usr/share/nginx/html;
        internal;
    }
}
EOF

    # Create custom error page
    mkdir -p /usr/share/nginx/html
    cat > /usr/share/nginx/html/50x.html << 'EOF'
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

    # Test nginx configuration
    nginx -t || log_error "Nginx configuration test failed"
    
    # Restart nginx
    systemctl restart nginx || log_error "Failed to restart nginx"
    
    log_success "Nginx configured successfully."
}

configure_firewall() {
    log_info "Configuring firewall (firewalld)..."
    
    # Enable and start firewalld
    systemctl enable firewalld || log_error "Failed to enable firewalld"
    systemctl start firewalld || log_error "Failed to start firewalld"
    
    # Allow essential services
    firewall-cmd --permanent --add-service=ssh || log_error "Failed to allow SSH"
    firewall-cmd --permanent --add-service=http || log_error "Failed to allow HTTP"
    firewall-cmd --permanent --add-service=https || log_error "Failed to allow HTTPS"
    
    # Reload firewall
    firewall-cmd --reload || log_error "Failed to reload firewall"
    
    log_success "Firewall configured successfully."
}

configure_selinux() {
    log_info "Configuring SELinux..."
    
    # Check if SELinux is enabled
    if getenforce | grep -q "Enforcing"; then
        # Allow nginx to connect to network
        setsebool -P httpd_can_network_connect 1 || log_warning "Failed to set httpd_can_network_connect"
        
        # Allow nginx to connect to port 8000
        semanage port -a -t http_port_t -p tcp 8000 || log_warning "Failed to add port 8000 to SELinux"
        
        log_success "SELinux configured successfully."
    else
        log_info "SELinux is not in enforcing mode, skipping SELinux configuration."
    fi
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
        
        echo -e "\n=== System Resources ==="
        echo "Disk usage:"
        df -h "$APP_DIR" | tail -1 | awk '{print "  " $1 ": " $3 "/" $2 " (" $5 " used)"}' || echo "  Disk info unavailable"
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
    mkdir -p /etc/systemd/system/supervisord.service.d
    cat > /etc/systemd/system/supervisord.service.d/override.conf << 'EOF'
[Service]
LimitNOFILE=32768
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
    if command -v dnf &> /dev/null; then
        dnf autoremove -y > /dev/null 2>&1
        dnf clean all > /dev/null 2>&1
    else
        yum autoremove -y > /dev/null 2>&1
        yum clean all > /dev/null 2>&1
    fi
    
    # Set final permissions
    chmod -R 755 "$APP_DIR"
    chmod 600 "$APP_DIR/.env"
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    
    log_success "Installation cleanup completed."
}

# --- Main Installation Flow ---
main() {
    local start_time=$(date +%s)
    
    log_info "Starting Gemini Claude Adapter deployment for CentOS/RHEL..."
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
    configure_selinux
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
    echo "   GITHUB_REPO_URL=https://github.com/tellerlin/gemini-claude.git sudo bash scripts/deploy-centos.sh"
    echo ""
    log_info "Installation complete! Check the deployment log at: $LOG_FILE"
}

# Error handling
trap 'log_error "Deployment failed. Check $LOG_FILE for details."' ERR

# Run main installation
main "$@"