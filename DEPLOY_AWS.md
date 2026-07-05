# Deploying to AWS EC2 (Ubuntu, free tier)

## Architecture

```
User
  |
  v
Internet
  |
  v
AWS EC2 (Ubuntu)
  |
  +-- Nginx  (port 80, reverse proxy)
  |
  v
Gunicorn  (127.0.0.1:8000)
  |
  v
Flask App
  |
  +-- HTML templates
  +-- CSS (static)
  +-- loan_model.pkl  (created on the server by train.py)
  +-- Python (app.py, predict_utils.py)

Codebase is pulled from GitHub:
https://github.com/PritiyaxShukla/AKGEC_internal_internship
```

Cost: a `t3.micro` / `t2.micro` instance is **free-tier eligible** for 12 months
(750 hours/month). **Terminate the instance when done** to avoid charges.

---

## Step 1 - Launch an EC2 instance (AWS Console)

1. Go to **EC2 -> Instances -> Launch instances**.
2. **Name:** `loan-predictor`
3. **AMI:** **Ubuntu Server 24.04 LTS** (look for "Free tier eligible").
4. **Instance type:** `t3.micro` (or `t2.micro`).
5. **Key pair:** Create a new key pair (e.g. `loan-key`), type **RSA**, format **.pem**,
   and download it. Keep this file safe.
6. **Network settings -> Edit -> Security group.** Add these inbound rules:
   | Type | Port | Source | Why |
   |---|---|---|---|
   | SSH | 22 | My IP | so you can connect |
   | HTTP | 80 | Anywhere (0.0.0.0/0) | so the web app is reachable |
7. Leave storage at the default 8 GB.
8. Click **Launch instance**, then copy the **Public IPv4 address**.

---

## Step 2 - Connect and deploy

From the folder holding your `.pem` file (PowerShell or Git Bash):

```bash
chmod 400 loan-key.pem
ssh -i loan-key.pem ubuntu@<PUBLIC_IP>
```

On the instance, run this single command. It clones the repo and runs the setup
script (installs Python + Nginx, trains the model, starts gunicorn + Nginx):

```bash
sudo apt-get update -y && sudo apt-get install -y git && \
git clone https://github.com/PritiyaxShukla/AKGEC_internal_internship.git && \
cd AKGEC_internal_internship && \
bash deploy/setup_ec2.sh
```

---

## Step 3 - Open the app

```
http://<PUBLIC_IP>
```

(Port 80 - no port number needed, because Nginx is in front.)

---

## Managing the app (on the instance)

```bash
sudo systemctl status loanapp        # gunicorn app service
sudo systemctl restart loanapp       # restart after code changes
sudo journalctl -u loanapp -f        # live app logs
sudo systemctl status nginx          # nginx status
sudo nginx -t                        # test nginx config
```

Deploy code updates later:

```bash
cd ~/AKGEC_internal_internship && git pull && sudo systemctl restart loanapp
```

---

## Cleaning up (avoid charges)

**EC2 -> Instances**, select the instance, **Instance state -> Terminate instance**.
Optionally delete the key pair and security group afterward.
