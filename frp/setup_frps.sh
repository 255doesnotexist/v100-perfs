#!/usr/bin/env bash
# ─── 在阿里云上安装并启动 frps ────────────────────────────────
# 用法: scp 这个脚本到阿里云, 然后 ssh 执行
#   scp setup_frps.sh root@<ALIYUN_PUBLIC_IP>:/tmp/
#   ssh root@<ALIYUN_PUBLIC_IP> 'bash /tmp/setup_frps.sh'
set -euo pipefail

FRP_VER="0.65.0"
ARCH="linux_amd64"

echo "=== Installing frps v${FRP_VER} ==="

# Download
cd /tmp
if [ ! -f "frp_${FRP_VER}_${ARCH}.tar.gz" ]; then
    curl -LO "https://github.com/fatedier/frp/releases/download/v${FRP_VER}/frp_${FRP_VER}_${ARCH}.tar.gz"
fi
tar xzf "frp_${FRP_VER}_${ARCH}.tar.gz"

# Install binary
install -D "frp_${FRP_VER}_${ARCH}/frps" /usr/local/bin/frps

# Config dir
mkdir -p /etc/frp

# Write config if not exists
if [ ! -f /etc/frp/frps.toml ]; then
    cat > /etc/frp/frps.toml << 'EOF'
bindPort = 7000

auth.method = "token"
auth.token = "YOUR_FRP_TOKEN_HERE"

transport.tls.force = true

allowPorts = [
  { start = 25555, end = 25555 },
]
EOF
    echo "Created /etc/frp/frps.toml"
else
    echo "/etc/frp/frps.toml already exists, skipping"
fi

# systemd service
cat > /etc/systemd/system/frps.service << 'EOF'
[Unit]
Description=frps service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frps -c /etc/frp/frps.toml
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable frps
systemctl restart frps

echo "=== frps installed and started ==="
systemctl status frps --no-pager -l

echo ""
echo "=== Firewall: ensure ports 7000 + 25555 are open ==="
echo "Aliyun security group needs TCP 7000 and 25555 inbound."
echo ""
echo "Also configure /etc/frp/frps.toml if you changed the token."
