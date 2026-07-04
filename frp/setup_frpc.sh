#!/usr/bin/env bash
set -euo pipefail

FRP_VER="0.65.0"
ARCH="linux_amd64"

echo "=== Installing frpc v${FRP_VER} ==="

cd /tmp
if [ ! -f "frp_${FRP_VER}_${ARCH}.tar.gz" ]; then
    curl -LO "https://github.com/fatedier/frp/releases/download/v${FRP_VER}/frp_${FRP_VER}_${ARCH}.tar.gz"
fi
tar xzf "frp_${FRP_VER}_${ARCH}.tar.gz"

sudo install -D "frp_${FRP_VER}_${ARCH}/frpc" /usr/local/bin/frpc

sudo mkdir -p /etc/frp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/frpc.toml" ]; then
    sudo cp "$SCRIPT_DIR/frpc.toml" /etc/frp/frpc.toml
else
    sudo tee /etc/frp/frpc.toml << 'EOF'
serverAddr = "<ALIYUN_PUBLIC_IP>"
serverPort = 7000

auth.method = "token"
auth.token = "YOUR_FRP_TOKEN_HERE"

transport.tls.enable = true

[[proxies]]
name = "llama-proxy"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8000
remotePort = 25555
EOF
fi

sudo tee /etc/systemd/system/frpc.service << 'EOF'
[Unit]
Description=frpc service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.toml
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable frpc
sudo systemctl restart frpc

echo "=== frpc installed and started ==="
sudo systemctl status frpc --no-pager -l
