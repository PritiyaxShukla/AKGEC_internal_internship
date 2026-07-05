#!/usr/bin/env bash
#
# One-shot provisioning script for an Ubuntu EC2 instance.
#
# Architecture:
#   Internet -> Nginx (port 80) -> Gunicorn (127.0.0.1:8000) -> Flask app
#
# It installs dependencies, trains the model, runs the app under gunicorn as a
# systemd service, and puts Nginx in front as a reverse proxy on port 80.
#
# Usage (run as the 'ubuntu' user on the instance):
#     bash deploy/setup_ec2.sh
#
set -euo pipefail

APP_DIR="$HOME/AKGEC_internal_internship"
REPO="https://github.com/PritiyaxShukla/AKGEC_internal_internship.git"

echo ">>> [1/6] Installing system packages (python, git, nginx)..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git nginx

echo ">>> [2/6] Cloning / updating the repository..."
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull
else
    git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo ">>> [3/6] Creating virtualenv and installing dependencies..."
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo ">>> [4/6] Training the model (creates model/loan_model.pkl)..."
./.venv/bin/python train.py

echo ">>> [5/6] Installing and starting the gunicorn systemd service..."
sudo cp deploy/loanapp.service /etc/systemd/system/loanapp.service
sudo systemctl daemon-reload
sudo systemctl enable --now loanapp
sleep 2
sudo systemctl --no-pager status loanapp || true

echo ">>> [6/6] Configuring Nginx reverse proxy (port 80 -> gunicorn 8000)..."
sudo cp deploy/nginx-loanapp.conf /etc/nginx/sites-available/loanapp
sudo ln -sf /etc/nginx/sites-available/loanapp /etc/nginx/sites-enabled/loanapp
sudo rm -f /etc/nginx/sites-enabled/default   # remove Nginx's default welcome site
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# Best-effort public IP (IMDSv2, falls back to a placeholder).
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
PUBIP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<EC2_PUBLIC_IP>")

echo ""
echo ">>> Done. With security-group port 80 open, the app is live at:"
echo ">>>     http://$PUBIP"
