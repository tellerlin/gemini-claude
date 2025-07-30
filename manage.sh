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
