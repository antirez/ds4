#!/bin/bash

# ==============================================================================
# DwarfStar DGX Spark Service Setup Script
# ==============================================================================

# --- CONFIGURATION (Edit these if necessary) ---
PROJECT_DIR="$(pwd)"
BINARY_PATH="${PROJECT_DIR}/ds4-server"
MODEL_PATH="${PROJECT_DIR}/ds4flash.gguf"
KV_CACHE_DIR="/var/lib/ds4/kvcache"
SERVICE_NAME="ds4-server"
SERVICE_USER="ds4"

# Server Runtime Options
CTX_SIZE="100000"
KV_DISK_SPACE="8192"
HOST_ADDR="0.0.0.0"
PORT="8000"
API_KEY=""            # Set to a secret key to enable API key authentication

set -e

echo "🚀 Starting DwarfStar Service Setup..."

# 1. Create system user if not exists
if id "$SERVICE_USER" &>/dev/null; then
    echo "ℹ️ User $SERVICE_USER already exists."
else
    echo "👤 Creating system user $SERVICE_USER..."
    sudo useradd -r -s /usr/sbin/nologin $SERVICE_USER
fi

# 2. Setup KV Cache directory
echo "📂 Setting up KV cache directory at $KV_CACHE_DIR..."
sudo mkdir -p $KV_CACHE_DIR
sudo chown -R $SERVICE_USER:$SERVICE_USER $KV_CACHE_DIR
sudo chmod 750 $KV_CACHE_DIR

# 3. Ensure permissions for the project directory
# Note: The ds4 user needs read access to the model and execute access to the binary
echo "🔑 Setting permissions for project files..."
sudo chown -R $USER:$USER $PROJECT_DIR # Ensure current user owns it
sudo chmod -R o+rx $PROJECT_DIR # Allow others (including ds4 user) to read/execute

# 4. Create Systemd Service File
echo "🛠️ Creating systemd service unit..."
cat <<EOF | sudo tee /etc/systemd/system/${SERVICE_NAME}.service
[Unit]
Description=DwarfStar DeepSeek V4 Inference Server
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${BINARY_PATH} \\
    -m ${MODEL_PATH} \\
    --host ${HOST_ADDR} \\
    --port ${PORT} \\
    --ctx ${CTX_SIZE} \\
    --kv-disk-dir ${KV_CACHE_DIR} \\
    --kv-disk-space-mb ${KV_DISK_SPACE}
    ${API_KEY:+--api-key ${API_KEY}}
Restart=always
RestartSec=5
StandardOutput=append:/var/log/ds4-server.log
StandardError=append:/var/log/ds4-server.log

[Install]
WantedBy=multi-user.target
EOF

# 5. Create log file and set permissions
sudo touch /var/log/ds4-server.log
sudo chown $SERVICE_USER:$SERVICE_USER /var/log/ds4-server.log

# 6. Enable and Start Service
echo "⚙️ Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

echo "----------------------------------------------------------------"
echo "✅ Setup Complete!"
echo "----------------------------------------------------------------"
echo "Service Status:  sudo systemctl status ${SERVICE_NAME}"
echo "View Logs:       sudo tail -f /var/log/ds4-server.log"
echo "Test API:        curl http://localhost:${PORT}/v1/models"
echo "----------------------------------------------------------------"
