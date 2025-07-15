#!/bin/sh
# UltraThink Crypto Trading System Manager
# POSIX-compliant management script with bulletproof error handling
# 
# This script provides comprehensive management of the UltraThink trading system
# with robust error handling, automatic recovery, and cross-platform compatibility.

set -u  # Exit on undefined variables
IFS=' 	
'  # Set secure IFS

# ============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# ============================================================================

# Script metadata
readonly SCRIPT_NAME="ultrathink_manager"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"

# System paths and files
readonly VENV_DIR="$PROJECT_ROOT/venv_ultrathink"
readonly PYTHON_MAIN="$PROJECT_ROOT/ultrathink_main.py"
readonly CONFIG_DIR="$PROJECT_ROOT/configs"
readonly LOGS_DIR="$PROJECT_ROOT/logs"
readonly CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
readonly BACKUPS_DIR="$PROJECT_ROOT/backups"
readonly TEMP_DIR="$PROJECT_ROOT/temp"
readonly LOCK_DIR="$PROJECT_ROOT/.locks"

# Log files
readonly MAIN_LOG="$LOGS_DIR/ultrathink_manager.log"
readonly ERROR_LOG="$LOGS_DIR/ultrathink_errors.log"
readonly ANALYSIS_LOG="$LOGS_DIR/ultrathink_analysis.log"

# Lock files
readonly MAIN_LOCK="$LOCK_DIR/ultrathink_main.lock"
readonly BACKUP_LOCK="$LOCK_DIR/backup.lock"
readonly ANALYSIS_LOCK="$LOCK_DIR/analysis.lock"

# Configuration
readonly MAX_LOG_SIZE=10485760  # 10MB
readonly MAX_LOG_FILES=5
readonly BACKUP_RETENTION_DAYS=30
readonly HEALTH_CHECK_INTERVAL=300  # 5 minutes
readonly DEFAULT_TIMEOUT=1800  # 30 minutes

# Exit codes
readonly EXIT_SUCCESS=0
readonly EXIT_GENERAL_ERROR=1
readonly EXIT_MISUSE=2
readonly EXIT_CANNOT_EXECUTE=126
readonly EXIT_COMMAND_NOT_FOUND=127
readonly EXIT_INVALID_ARGUMENT=128
readonly EXIT_SIGNAL_BASE=128

# Colors for output (only if terminal supports it)
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    readonly COLOR_RED="$(tput setaf 1 2>/dev/null || true)"
    readonly COLOR_GREEN="$(tput setaf 2 2>/dev/null || true)"
    readonly COLOR_YELLOW="$(tput setaf 3 2>/dev/null || true)"
    readonly COLOR_BLUE="$(tput setaf 4 2>/dev/null || true)"
    readonly COLOR_MAGENTA="$(tput setaf 5 2>/dev/null || true)"
    readonly COLOR_CYAN="$(tput setaf 6 2>/dev/null || true)"
    readonly COLOR_RESET="$(tput sgr0 2>/dev/null || true)"
else
    readonly COLOR_RED=""
    readonly COLOR_GREEN=""
    readonly COLOR_YELLOW=""
    readonly COLOR_BLUE=""
    readonly COLOR_MAGENTA=""
    readonly COLOR_CYAN=""
    readonly COLOR_RESET=""
fi

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Print message with timestamp and level
log_message() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    # Print to appropriate streams
    case "$level" in
        ERROR)
            printf '%s [%s] %s: %s\n' "$timestamp" "$$" "$level" "$message" >&2
            printf '%s [%s] %s: %s\n' "$timestamp" "$$" "$level" "$message" >> "$ERROR_LOG" 2>/dev/null || true
            ;;
        WARN)
            printf '%s [%s] %s: %s\n' "$timestamp" "$$" "$level" "$message" >&2
            ;;
        INFO|DEBUG)
            printf '%s [%s] %s: %s\n' "$timestamp" "$$" "$level" "$message"
            ;;
    esac
    
    # Always log to main log file
    printf '%s [%s] %s: %s\n' "$timestamp" "$$" "$level" "$message" >> "$MAIN_LOG" 2>/dev/null || true
}

# Colored output functions
print_error() {
    printf '%s[ERROR]%s %s\n' "$COLOR_RED" "$COLOR_RESET" "$*" >&2
    log_message "ERROR" "$@"
}

print_warning() {
    printf '%s[WARN]%s %s\n' "$COLOR_YELLOW" "$COLOR_RESET" "$*" >&2
    log_message "WARN" "$@"
}

print_info() {
    printf '%s[INFO]%s %s\n' "$COLOR_BLUE" "$COLOR_RESET" "$*"
    log_message "INFO" "$@"
}

print_success() {
    printf '%s[SUCCESS]%s %s\n' "$COLOR_GREEN" "$COLOR_RESET" "$*"
    log_message "INFO" "SUCCESS: $*"
}

print_debug() {
    [ "${DEBUG:-0}" = "1" ] && {
        printf '%s[DEBUG]%s %s\n' "$COLOR_MAGENTA" "$COLOR_RESET" "$*" >&2
        log_message "DEBUG" "$@"
    }
}

# Safe file operations
safe_copy() {
    local src="$1"
    local dest="$2"
    local temp_dest
    
    [ -f "$src" ] || {
        print_error "Source file does not exist: $src"
        return 1
    }
    
    temp_dest="${dest}.tmp.$$"
    
    # Copy to temporary file first
    if cp "$src" "$temp_dest" 2>/dev/null; then
        # Atomic move
        if mv "$temp_dest" "$dest" 2>/dev/null; then
            return 0
        else
            rm -f "$temp_dest" 2>/dev/null || true
            return 1
        fi
    else
        rm -f "$temp_dest" 2>/dev/null || true
        return 1
    fi
}

# Safe directory creation
safe_mkdir() {
    local dir="$1"
    local mode="${2:-755}"
    
    [ -d "$dir" ] && return 0
    
    if mkdir -p "$dir" 2>/dev/null; then
        chmod "$mode" "$dir" 2>/dev/null || true
        return 0
    else
        return 1
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate path safety (prevent directory traversal)
validate_path() {
    local path="$1"
    
    case "$path" in
        */../*|*/./*|*/.|*/..) 
            print_error "Unsafe path detected: $path"
            return 1
            ;;
        /*)
            # Absolute path is OK if within project
            case "$path" in
                "$PROJECT_ROOT"/*) return 0 ;;
                *) 
                    print_error "Path outside project root: $path"
                    return 1
                    ;;
            esac
            ;;
        *)
            # Relative path is OK
            return 0
            ;;
    esac
}

# Check available disk space
check_disk_space() {
    local required_mb="${1:-1024}"  # Default 1GB
    local available_mb
    
    if command_exists df; then
        # Get available space in MB
        available_mb="$(df -P "$PROJECT_ROOT" | awk 'NR==2 {print int($4/1024)}')"
        
        if [ "$available_mb" -lt "$required_mb" ]; then
            print_error "Insufficient disk space. Required: ${required_mb}MB, Available: ${available_mb}MB"
            return 1
        fi
    else
        print_warning "Cannot check disk space: df command not available"
    fi
    
    return 0
}

# ============================================================================
# LOCK MANAGEMENT
# ============================================================================

# Acquire exclusive lock
acquire_lock() {
    local lock_file="$1"
    local timeout="${2:-30}"
    local wait_time=0
    
    safe_mkdir "$LOCK_DIR" || {
        print_error "Cannot create lock directory: $LOCK_DIR"
        return 1
    }
    
    while [ "$wait_time" -lt "$timeout" ]; do
        if mkdir "$lock_file" 2>/dev/null; then
            # Store PID in lock file
            echo "$$" > "$lock_file/pid" 2>/dev/null || true
            echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$lock_file/timestamp" 2>/dev/null || true
            print_debug "Acquired lock: $lock_file"
            return 0
        fi
        
        # Check if lock is stale
        if [ -d "$lock_file" ] && [ -f "$lock_file/pid" ]; then
            local lock_pid
            lock_pid="$(cat "$lock_file/pid" 2>/dev/null || echo "")"
            
            if [ -n "$lock_pid" ] && ! kill -0 "$lock_pid" 2>/dev/null; then
                print_warning "Removing stale lock: $lock_file (PID: $lock_pid)"
                release_lock "$lock_file"
                continue
            fi
        fi
        
        sleep 1
        wait_time=$((wait_time + 1))
    done
    
    print_error "Failed to acquire lock after ${timeout} seconds: $lock_file"
    return 1
}

# Release lock
release_lock() {
    local lock_file="$1"
    
    if [ -d "$lock_file" ]; then
        rm -rf "$lock_file" 2>/dev/null || {
            print_warning "Failed to remove lock: $lock_file"
            return 1
        }
        print_debug "Released lock: $lock_file"
    fi
    
    return 0
}

# ============================================================================
# SIGNAL HANDLING AND CLEANUP
# ============================================================================

# Global cleanup function
cleanup() {
    local exit_code="$?"
    
    print_debug "Performing cleanup (exit code: $exit_code)"
    
    # Release any locks held by this process
    if [ -d "$LOCK_DIR" ]; then
        find "$LOCK_DIR" -type d -name "*.lock" 2>/dev/null | while read -r lock_file; do
            if [ -f "$lock_file/pid" ]; then
                local lock_pid
                lock_pid="$(cat "$lock_file/pid" 2>/dev/null || echo "")"
                if [ "$lock_pid" = "$$" ]; then
                    release_lock "$lock_file"
                fi
            fi
        done
    fi
    
    # Clean up temporary files
    find "$TEMP_DIR" -name "*.tmp.$$" -type f -delete 2>/dev/null || true
    
    # Kill child processes
    local child_pids
    child_pids="$(jobs -p 2>/dev/null || true)"
    if [ -n "$child_pids" ]; then
        print_debug "Terminating child processes: $child_pids"
        # shellcheck disable=SC2086
        kill $child_pids 2>/dev/null || true
        sleep 2
        # shellcheck disable=SC2086
        kill -9 $child_pids 2>/dev/null || true
    fi
    
    exit "$exit_code"
}

# Signal handlers
handle_signal() {
    local signal="$1"
    print_warning "Received signal: $signal"
    cleanup
    exit $((EXIT_SIGNAL_BASE + signal))
}

# Set up signal handlers
trap 'handle_signal 1' HUP
trap 'handle_signal 2' INT
trap 'handle_signal 3' QUIT
trap 'handle_signal 15' TERM
trap 'cleanup' EXIT

# ============================================================================
# SYSTEM VALIDATION AND SETUP
# ============================================================================

# Validate system requirements
validate_system() {
    print_info "Validating system requirements..."
    
    # Check essential commands
    local required_commands="python3 pip curl tar gzip"
    local missing_commands=""
    
    for cmd in $required_commands; do
        if ! command_exists "$cmd"; then
            missing_commands="$missing_commands $cmd"
        fi
    done
    
    if [ -n "$missing_commands" ]; then
        print_error "Missing required commands:$missing_commands"
        return 1
    fi
    
    # Check Python version
    local python_version
    python_version="$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")"
    
    case "$python_version" in
        3.[8-9]|3.1[0-9]|[4-9].*)
            print_debug "Python version OK: $python_version"
            ;;
        *)
            print_error "Python 3.8+ required, found: $python_version"
            return 1
            ;;
    esac
    
    # Check disk space
    check_disk_space 2048 || return 1  # 2GB minimum
    
    # Check write permissions
    for dir in "$PROJECT_ROOT" "$CONFIG_DIR" "$LOGS_DIR"; do
        if ! [ -w "$dir" ] 2>/dev/null; then
            print_error "No write permission for: $dir"
            return 1
        fi
    done
    
    print_success "System validation passed"
    return 0
}

# Initialize directory structure
init_directories() {
    print_info "Initializing directory structure..."
    
    local dirs="$LOGS_DIR $CHECKPOINTS_DIR $BACKUPS_DIR $TEMP_DIR $LOCK_DIR $CONFIG_DIR"
    
    for dir in $dirs; do
        safe_mkdir "$dir" || {
            print_error "Failed to create directory: $dir"
            return 1
        }
    done
    
    # Set up log rotation
    setup_log_rotation
    
    print_success "Directory structure initialized"
    return 0
}

# Set up log rotation
setup_log_rotation() {
    local log_files="$MAIN_LOG $ERROR_LOG $ANALYSIS_LOG"
    
    for log_file in $log_files; do
        if [ -f "$log_file" ] && [ "$(wc -c < "$log_file" 2>/dev/null || echo 0)" -gt "$MAX_LOG_SIZE" ]; then
            print_debug "Rotating log file: $log_file"
            
            # Rotate existing logs
            local i="$MAX_LOG_FILES"
            while [ "$i" -gt 1 ]; do
                local old_log="${log_file}.$(($i - 1))"
                local new_log="${log_file}.$i"
                
                if [ -f "$old_log" ]; then
                    mv "$old_log" "$new_log" 2>/dev/null || true
                fi
                
                i=$((i - 1))
            done
            
            # Move current log to .1
            mv "$log_file" "${log_file}.1" 2>/dev/null || true
            touch "$log_file" 2>/dev/null || true
        fi
    done
}

# ============================================================================
# VIRTUAL ENVIRONMENT MANAGEMENT
# ============================================================================

# Create or update virtual environment
setup_venv() {
    print_info "Setting up virtual environment..."
    
    acquire_lock "$LOCK_DIR/venv_setup.lock" 60 || return 1
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment..."
        
        if ! python3 -m venv "$VENV_DIR"; then
            release_lock "$LOCK_DIR/venv_setup.lock"
            print_error "Failed to create virtual environment"
            return 1
        fi
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    . "$VENV_DIR/bin/activate" || {
        release_lock "$LOCK_DIR/venv_setup.lock"
        print_error "Failed to activate virtual environment"
        return 1
    }
    
    # Upgrade pip
    print_info "Upgrading pip..."
    python -m pip install --upgrade pip >/dev/null 2>&1 || {
        print_warning "Failed to upgrade pip"
    }
    
    # Install requirements
    local requirements_file="$PROJECT_ROOT/requirements.txt"
    if [ -f "$requirements_file" ]; then
        print_info "Installing Python packages..."
        
        if ! python -m pip install -r "$requirements_file" >/dev/null 2>&1; then
            release_lock "$LOCK_DIR/venv_setup.lock"
            print_error "Failed to install requirements"
            return 1
        fi
    else
        print_warning "Requirements file not found: $requirements_file"
    fi
    
    release_lock "$LOCK_DIR/venv_setup.lock"
    print_success "Virtual environment setup complete"
    return 0
}

# Activate virtual environment
activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found. Run 'setup' first."
        return 1
    fi
    
    # shellcheck source=/dev/null
    . "$VENV_DIR/bin/activate" || {
        print_error "Failed to activate virtual environment"
        return 1
    }
    
    return 0
}

# ============================================================================
# BACKUP AND RESTORE OPERATIONS
# ============================================================================

# Create system backup
create_backup() {
    local backup_name="${1:-$(date '+%Y%m%d_%H%M%S')}"
    local backup_file="$BACKUPS_DIR/ultrathink_backup_${backup_name}.tar.gz"
    
    print_info "Creating backup: $backup_name"
    
    acquire_lock "$BACKUP_LOCK" 300 || return 1
    
    # Check disk space
    check_disk_space 1024 || {
        release_lock "$BACKUP_LOCK"
        return 1
    }
    
    # Create temporary directory for backup
    local temp_backup_dir="$TEMP_DIR/backup_$$"
    safe_mkdir "$temp_backup_dir" || {
        release_lock "$BACKUP_LOCK"
        return 1
    }
    
    # Copy essential files
    print_info "Copying files for backup..."
    
    local backup_items="ultrathink ultrathink_main.py utils features data models strategies configs"
    
    for item in $backup_items; do
        local src_path="$PROJECT_ROOT/$item"
        if [ -e "$src_path" ]; then
            if [ -d "$src_path" ]; then
                cp -r "$src_path" "$temp_backup_dir/" 2>/dev/null || {
                    print_warning "Failed to copy directory: $item"
                }
            else
                cp "$src_path" "$temp_backup_dir/" 2>/dev/null || {
                    print_warning "Failed to copy file: $item"
                }
            fi
        fi
    done
    
    # Create backup metadata
    cat > "$temp_backup_dir/backup_metadata.json" << EOF
{
    "backup_name": "$backup_name",
    "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "hostname": "$(hostname 2>/dev/null || echo unknown)",
    "script_version": "$SCRIPT_VERSION",
    "created_by": "$(whoami 2>/dev/null || echo unknown)"
}
EOF
    
    # Create compressed archive
    print_info "Creating compressed archive..."
    
    if (cd "$TEMP_DIR" && tar -czf "$backup_file" "backup_$$" 2>/dev/null); then
        rm -rf "$temp_backup_dir"
        release_lock "$BACKUP_LOCK"
        
        print_success "Backup created: $backup_file"
        
        # Clean old backups
        cleanup_old_backups
        
        return 0
    else
        rm -rf "$temp_backup_dir"
        release_lock "$BACKUP_LOCK"
        print_error "Failed to create backup archive"
        return 1
    fi
}

# Restore from backup
restore_backup() {
    local backup_file="$1"
    
    [ -f "$backup_file" ] || {
        print_error "Backup file not found: $backup_file"
        return 1
    }
    
    print_warning "This will overwrite existing files. Continue? (y/N)"
    read -r response
    case "$response" in
        [Yy]|[Yy][Ee][Ss])
            ;;
        *)
            print_info "Restore cancelled"
            return 0
            ;;
    esac
    
    print_info "Restoring from backup: $backup_file"
    
    acquire_lock "$BACKUP_LOCK" 300 || return 1
    
    # Create emergency checkpoint before restore
    create_backup "emergency_before_restore" || {
        print_warning "Failed to create emergency backup"
    }
    
    # Extract backup to temporary location
    local temp_restore_dir="$TEMP_DIR/restore_$$"
    safe_mkdir "$temp_restore_dir" || {
        release_lock "$BACKUP_LOCK"
        return 1
    }
    
    if tar -xzf "$backup_file" -C "$temp_restore_dir" 2>/dev/null; then
        # Find the backup directory (should be backup_*)
        local backup_content_dir
        backup_content_dir="$(find "$temp_restore_dir" -maxdepth 1 -type d -name "backup_*" | head -n 1)"
        
        if [ -n "$backup_content_dir" ] && [ -d "$backup_content_dir" ]; then
            # Copy files back
            print_info "Restoring files..."
            
            find "$backup_content_dir" -mindepth 1 -maxdepth 1 ! -name "backup_metadata.json" | while read -r item; do
                local item_name
                item_name="$(basename "$item")"
                local dest_path="$PROJECT_ROOT/$item_name"
                
                if [ -d "$item" ]; then
                    rm -rf "$dest_path" 2>/dev/null || true
                    cp -r "$item" "$dest_path" 2>/dev/null || {
                        print_warning "Failed to restore directory: $item_name"
                    }
                else
                    cp "$item" "$dest_path" 2>/dev/null || {
                        print_warning "Failed to restore file: $item_name"
                    }
                fi
            done
            
            print_success "Restore completed successfully"
        else
            print_error "Invalid backup format"
            rm -rf "$temp_restore_dir"
            release_lock "$BACKUP_LOCK"
            return 1
        fi
    else
        print_error "Failed to extract backup"
        rm -rf "$temp_restore_dir"
        release_lock "$BACKUP_LOCK"
        return 1
    fi
    
    rm -rf "$temp_restore_dir"
    release_lock "$BACKUP_LOCK"
    return 0
}

# Clean up old backups
cleanup_old_backups() {
    print_debug "Cleaning up old backups..."
    
    if [ -d "$BACKUPS_DIR" ]; then
        # Find backups older than retention period
        find "$BACKUPS_DIR" -name "ultrathink_backup_*.tar.gz" -type f -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
    fi
}

# ============================================================================
# ANALYSIS OPERATIONS
# ============================================================================

# Run UltraThink analysis
run_analysis() {
    local symbols="$1"
    local mode="${2:-portfolio}"
    local output_dir="${3:-}"
    
    print_info "Running UltraThink analysis..."
    print_info "Symbols: $symbols"
    print_info "Mode: $mode"
    
    acquire_lock "$ANALYSIS_LOCK" 600 || return 1  # 10 minute timeout
    
    # Activate virtual environment
    activate_venv || {
        release_lock "$ANALYSIS_LOCK"
        return 1
    }
    
    # Validate main script exists
    if [ ! -f "$PYTHON_MAIN" ]; then
        release_lock "$ANALYSIS_LOCK"
        print_error "Main script not found: $PYTHON_MAIN"
        return 1
    fi
    
    # Prepare command arguments
    local cmd_args=""
    
    case "$mode" in
        single)
            cmd_args="--single-symbol $symbols"
            ;;
        portfolio)
            cmd_args="--portfolio-mode --symbols $symbols"
            ;;
        individual)
            cmd_args="--symbols $symbols"
            ;;
        *)
            release_lock "$ANALYSIS_LOCK"
            print_error "Invalid analysis mode: $mode"
            return 1
            ;;
    esac
    
    # Add output directory if specified
    if [ -n "$output_dir" ]; then
        validate_path "$output_dir" || {
            release_lock "$ANALYSIS_LOCK"
            return 1
        }
        safe_mkdir "$output_dir" || {
            release_lock "$ANALYSIS_LOCK"
            return 1
        }
        cmd_args="$cmd_args --output $output_dir"
    fi
    
    # Run analysis with timeout
    print_info "Executing analysis command..."
    
    local analysis_start_time
    analysis_start_time="$(date '+%s')"
    
    # Use timeout if available, otherwise rely on signal handling
    if command_exists timeout; then
        timeout "$DEFAULT_TIMEOUT" python "$PYTHON_MAIN" $cmd_args >> "$ANALYSIS_LOG" 2>&1
        local exit_code="$?"
    else
        python "$PYTHON_MAIN" $cmd_args >> "$ANALYSIS_LOG" 2>&1 &
        local python_pid="$!"
        
        local elapsed=0
        while [ "$elapsed" -lt "$DEFAULT_TIMEOUT" ]; do
            if ! kill -0 "$python_pid" 2>/dev/null; then
                wait "$python_pid"
                local exit_code="$?"
                break
            fi
            sleep 5
            elapsed=$((elapsed + 5))
        done
        
        # Kill if timeout exceeded
        if [ "$elapsed" -ge "$DEFAULT_TIMEOUT" ]; then
            kill "$python_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$python_pid" 2>/dev/null || true
            local exit_code=124  # timeout exit code
        fi
    fi
    
    local analysis_end_time
    analysis_end_time="$(date '+%s')"
    local duration=$((analysis_end_time - analysis_start_time))
    
    release_lock "$ANALYSIS_LOCK"
    
    if [ "$exit_code" -eq 0 ]; then
        print_success "Analysis completed successfully in ${duration}s"
        return 0
    elif [ "$exit_code" -eq 124 ]; then
        print_error "Analysis timed out after ${DEFAULT_TIMEOUT}s"
        return 1
    else
        print_error "Analysis failed with exit code: $exit_code"
        return 1
    fi
}

# ============================================================================
# HEALTH MONITORING
# ============================================================================

# Perform system health check
health_check() {
    print_info "Performing system health check..."
    
    local issues=0
    
    # Check disk space
    if ! check_disk_space 1024; then
        issues=$((issues + 1))
    fi
    
    # Check virtual environment
    if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
        print_warning "Virtual environment missing or corrupted"
        issues=$((issues + 1))
    fi
    
    # Check essential files
    local essential_files="$PYTHON_MAIN ultrathink/__init__.py"
    for file in $essential_files; do
        if [ ! -f "$PROJECT_ROOT/$file" ]; then
            print_warning "Essential file missing: $file"
            issues=$((issues + 1))
        fi
    done
    
    # Check for stale locks
    if [ -d "$LOCK_DIR" ]; then
        find "$LOCK_DIR" -name "*.lock" -type d 2>/dev/null | while read -r lock_file; do
            if [ -f "$lock_file/pid" ]; then
                local lock_pid
                lock_pid="$(cat "$lock_file/pid" 2>/dev/null || echo "")"
                
                if [ -n "$lock_pid" ] && ! kill -0 "$lock_pid" 2>/dev/null; then
                    print_warning "Found stale lock: $lock_file"
                    release_lock "$lock_file"
                fi
            fi
        done
    fi
    
    # Check log file sizes
    local large_logs
    large_logs="$(find "$LOGS_DIR" -name "*.log" -size +50M 2>/dev/null || true)"
    if [ -n "$large_logs" ]; then
        print_warning "Large log files detected, consider rotation"
        issues=$((issues + 1))
    fi
    
    if [ "$issues" -eq 0 ]; then
        print_success "System health check passed"
        return 0
    else
        print_warning "System health check found $issues issues"
        return 1
    fi
}

# Monitor system in background
monitor_system() {
    print_info "Starting system monitoring (PID: $$)"
    
    while true; do
        sleep "$HEALTH_CHECK_INTERVAL"
        
        # Perform health check
        if ! health_check >/dev/null 2>&1; then
            print_warning "Health check failed during monitoring"
        fi
        
        # Rotate logs if needed
        setup_log_rotation
        
        # Clean up old temporary files
        find "$TEMP_DIR" -name "*.tmp.*" -type f -mtime +1 -delete 2>/dev/null || true
    done
}

# ============================================================================
# MAINTENANCE OPERATIONS
# ============================================================================

# Clean up system
cleanup_system() {
    print_info "Performing system cleanup..."
    
    # Clean temporary files
    if [ -d "$TEMP_DIR" ]; then
        find "$TEMP_DIR" -type f -name "*.tmp.*" -delete 2>/dev/null || true
        find "$TEMP_DIR" -type d -empty -delete 2>/dev/null || true
    fi
    
    # Clean old log files
    if [ -d "$LOGS_DIR" ]; then
        find "$LOGS_DIR" -name "*.log.[0-9]*" -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Clean old checkpoints
    if [ -d "$CHECKPOINTS_DIR" ]; then
        find "$CHECKPOINTS_DIR" -name "checkpoint_*" -type d -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Clean old backups
    cleanup_old_backups
    
    # Remove stale locks
    if [ -d "$LOCK_DIR" ]; then
        find "$LOCK_DIR" -name "*.lock" -type d 2>/dev/null | while read -r lock_file; do
            if [ -f "$lock_file/pid" ]; then
                local lock_pid
                lock_pid="$(cat "$lock_file/pid" 2>/dev/null || echo "")"
                
                if [ -n "$lock_pid" ] && ! kill -0 "$lock_pid" 2>/dev/null; then
                    release_lock "$lock_file"
                fi
            fi
        done
    fi
    
    print_success "System cleanup completed"
}

# Update system
update_system() {
    print_info "Updating UltraThink system..."
    
    # Create backup before update
    create_backup "before_update_$(date '+%Y%m%d_%H%M%S')" || {
        print_warning "Failed to create pre-update backup"
    }
    
    # Update virtual environment
    if [ -d "$VENV_DIR" ]; then
        activate_venv || return 1
        
        print_info "Updating Python packages..."
        python -m pip install --upgrade pip >/dev/null 2>&1
        
        local requirements_file="$PROJECT_ROOT/requirements.txt"
        if [ -f "$requirements_file" ]; then
            python -m pip install --upgrade -r "$requirements_file" >/dev/null 2>&1 || {
                print_warning "Failed to update some packages"
            }
        fi
    fi
    
    print_success "System update completed"
}

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

# Display help information
show_help() {
    cat << EOF
${COLOR_CYAN}UltraThink Crypto Trading System Manager${COLOR_RESET}
Version: $SCRIPT_VERSION

${COLOR_YELLOW}USAGE:${COLOR_RESET}
    $0 <command> [options]

${COLOR_YELLOW}COMMANDS:${COLOR_RESET}

${COLOR_GREEN}System Management:${COLOR_RESET}
    setup                   Initialize system and virtual environment
    health                  Perform system health check
    monitor                 Start background system monitoring
    cleanup                 Clean up temporary files and old data
    update                  Update system and dependencies
    
${COLOR_GREEN}Analysis Operations:${COLOR_RESET}
    analyze <symbols>       Run UltraThink analysis
        --mode <mode>           Analysis mode: single, portfolio, individual
        --output <dir>          Output directory for results
        
    Examples:
        $0 analyze "BTC-USD ETH-USD" --mode portfolio
        $0 analyze "BTC-USD" --mode single --output /tmp/results

${COLOR_GREEN}Backup & Restore:${COLOR_RESET}
    backup [name]           Create system backup
    restore <file>          Restore from backup file
    list-backups            List available backups

${COLOR_GREEN}Utilities:${COLOR_RESET}
    status                  Show system status
    logs [type]             Show logs (main, error, analysis)
    version                 Show version information
    help                    Show this help message

${COLOR_YELLOW}ENVIRONMENT VARIABLES:${COLOR_RESET}
    DEBUG=1                 Enable debug output
    TIMEOUT=<seconds>       Set analysis timeout (default: 1800)

${COLOR_YELLOW}EXAMPLES:${COLOR_RESET}
    # Initial setup
    $0 setup
    
    # Run portfolio analysis
    $0 analyze "BTC-USD ETH-USD SOL-USD" --mode portfolio
    
    # Create backup and monitor system
    $0 backup daily_backup
    $0 monitor &
    
    # Health check and cleanup
    $0 health
    $0 cleanup

EOF
}

# Show system status
show_status() {
    print_info "UltraThink System Status"
    echo "================================"
    
    # Basic system info
    printf "Script Version: %s\n" "$SCRIPT_VERSION"
    printf "Project Root: %s\n" "$PROJECT_ROOT"
    printf "Hostname: %s\n" "$(hostname 2>/dev/null || echo "unknown")"
    printf "Timestamp: %s\n" "$(date)"
    echo
    
    # Virtual environment status
    if [ -d "$VENV_DIR" ]; then
        printf "%sVirtual Environment:%s Present\n" "$COLOR_GREEN" "$COLOR_RESET"
        if [ -f "$VENV_DIR/bin/python" ]; then
            printf "Python Version: %s\n" "$("$VENV_DIR/bin/python" --version 2>&1)"
        fi
    else
        printf "%sVirtual Environment:%s Missing\n" "$COLOR_RED" "$COLOR_RESET"
    fi
    echo
    
    # Disk space
    if command_exists df; then
        printf "Disk Usage:\n"
        df -h "$PROJECT_ROOT" | awk 'NR==2 {printf "  Available: %s (%s used)\n", $4, $5}'
    fi
    echo
    
    # Active locks
    if [ -d "$LOCK_DIR" ]; then
        local active_locks
        active_locks="$(find "$LOCK_DIR" -name "*.lock" -type d 2>/dev/null | wc -l)"
        printf "Active Locks: %s\n" "$active_locks"
    fi
    
    # Recent backups
    if [ -d "$BACKUPS_DIR" ]; then
        local backup_count
        backup_count="$(find "$BACKUPS_DIR" -name "ultrathink_backup_*.tar.gz" 2>/dev/null | wc -l)"
        printf "Available Backups: %s\n" "$backup_count"
        
        if [ "$backup_count" -gt 0 ]; then
            printf "Latest Backup: %s\n" "$(find "$BACKUPS_DIR" -name "ultrathink_backup_*.tar.gz" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- | xargs basename 2>/dev/null || echo "unknown")"
        fi
    fi
    echo
}

# Show logs
show_logs() {
    local log_type="${1:-main}"
    local lines="${2:-50}"
    
    case "$log_type" in
        main)
            if [ -f "$MAIN_LOG" ]; then
                tail -n "$lines" "$MAIN_LOG"
            else
                print_warning "Main log file not found"
            fi
            ;;
        error)
            if [ -f "$ERROR_LOG" ]; then
                tail -n "$lines" "$ERROR_LOG"
            else
                print_warning "Error log file not found"
            fi
            ;;
        analysis)
            if [ -f "$ANALYSIS_LOG" ]; then
                tail -n "$lines" "$ANALYSIS_LOG"
            else
                print_warning "Analysis log file not found"
            fi
            ;;
        *)
            print_error "Invalid log type: $log_type (use: main, error, analysis)"
            return 1
            ;;
    esac
}

# List available backups
list_backups() {
    print_info "Available backups:"
    
    if [ ! -d "$BACKUPS_DIR" ]; then
        print_warning "Backup directory does not exist"
        return 1
    fi
    
    local backup_files
    backup_files="$(find "$BACKUPS_DIR" -name "ultrathink_backup_*.tar.gz" -type f 2>/dev/null)"
    
    if [ -z "$backup_files" ]; then
        print_info "No backups found"
        return 0
    fi
    
    printf "%-30s %-15s %s\n" "Backup Name" "Size" "Date"
    printf "%-30s %-15s %s\n" "----------" "----" "----"
    
    echo "$backup_files" | while read -r backup_file; do
        local backup_name
        backup_name="$(basename "$backup_file" .tar.gz | sed 's/ultrathink_backup_//')"
        
        local file_size
        if command_exists du; then
            file_size="$(du -h "$backup_file" 2>/dev/null | cut -f1)"
        else
            file_size="unknown"
        fi
        
        local file_date
        if command_exists stat; then
            file_date="$(stat -c %y "$backup_file" 2>/dev/null | cut -d' ' -f1)"
        else
            file_date="unknown"
        fi
        
        printf "%-30s %-15s %s\n" "$backup_name" "$file_size" "$file_date"
    done
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Validate command line arguments
validate_args() {
    case "$1" in
        setup|health|monitor|cleanup|update|status|version|help|list-backups)
            return 0
            ;;
        analyze|backup|restore|logs)
            if [ $# -lt 2 ]; then
                print_error "Command '$1' requires additional arguments"
                return 1
            fi
            return 0
            ;;
        "")
            print_error "No command specified"
            return 1
            ;;
        *)
            print_error "Unknown command: $1"
            return 1
            ;;
    esac
}

# Main function
main() {
    # Initialize logging
    init_directories
    
    print_info "UltraThink Manager v$SCRIPT_VERSION starting..."
    
    # Validate arguments
    validate_args "$@" || {
        echo "Use '$0 help' for usage information"
        return $EXIT_INVALID_ARGUMENT
    }
    
    local command="$1"
    shift
    
    case "$command" in
        setup)
            validate_system && setup_venv
            ;;
        health)
            health_check
            ;;
        monitor)
            monitor_system
            ;;
        cleanup)
            cleanup_system
            ;;
        update)
            update_system
            ;;
        analyze)
            local symbols="$1"
            local mode="portfolio"
            local output_dir=""
            
            # Parse additional arguments
            shift
            while [ $# -gt 0 ]; do
                case "$1" in
                    --mode)
                        mode="$2"
                        shift 2
                        ;;
                    --output)
                        output_dir="$2"
                        shift 2
                        ;;
                    *)
                        print_error "Unknown option: $1"
                        return $EXIT_INVALID_ARGUMENT
                        ;;
                esac
            done
            
            run_analysis "$symbols" "$mode" "$output_dir"
            ;;
        backup)
            local backup_name="${1:-}"
            create_backup "$backup_name"
            ;;
        restore)
            local backup_file="$1"
            restore_backup "$backup_file"
            ;;
        status)
            show_status
            ;;
        logs)
            local log_type="${1:-main}"
            local lines="${2:-50}"
            show_logs "$log_type" "$lines"
            ;;
        list-backups)
            list_backups
            ;;
        version)
            echo "UltraThink Manager v$SCRIPT_VERSION"
            ;;
        help)
            show_help
            ;;
        *)
            print_error "This should not happen - command validation failed"
            return $EXIT_GENERAL_ERROR
            ;;
    esac
}

# Execute main function with all arguments
main "$@"