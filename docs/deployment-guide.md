# Deployment Guide

> Complete guide for deploying Glitch Core to production

## 🎯 Overview

This guide covers deploying Glitch Core to production environments, including cloud platforms, monitoring, and scaling considerations.

## 🚀 Quick Deployment

### Local Production

```bash
# Build and start production stack
make build
make up

# Verify deployment
make health
curl http://localhost:8000/health
```

### Docker Compose Production

```bash
# Create production environment file
cp .env.example .env.prod

# Edit production settings
vim .env.prod

# Deploy with production config
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## ☁️ Cloud Deployment

### Azure VM Deployment

#### 1. VM Setup

```bash
# Create Azure VM
az vm create \
  --resource-group glitch-core-rg \
  --name glitch-core-vm \
  --image Ubuntu2204 \
  --size Standard_D2s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Open required ports
az vm open-port \
  --resource-group glitch-core-rg \
  --name glitch-core-vm \
  --port 80,443,8000
```

#### 2. Server Configuration

```bash
# SSH into VM
ssh azureuser@<vm-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx
sudo apt install nginx -y
```

#### 3. Application Deployment

```bash
# Clone repository
git clone https://github.com/your-org/glitch-core.git
cd glitch-core

# Create production environment
cp .env.example .env.prod
vim .env.prod

# Build and start
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
curl http://localhost:8000/health
```

#### 4. Nginx Configuration

```nginx
# /etc/nginx/sites-available/glitch-core
server {
    listen 80;
    server_name api.cognitive-drift.app;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/glitch-core /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 5. SSL Certificate

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d api.cognitive-drift.app

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### AWS EC2 Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --count 1 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx

# Configure security group
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0
```

#### 2. Application Deployment

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@<ec2-ip>

# Install Docker
sudo apt update
sudo apt install docker.io -y
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone https://github.com/your-org/glitch-core.git
cd glitch-core
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Google Cloud Platform

#### 1. Compute Engine Setup

```bash
# Create VM instance
gcloud compute instances create glitch-core-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=http-server,https-server

# Configure firewall
gcloud compute firewall-rules create allow-http \
  --allow tcp:80 \
  --target-tags=http-server \
  --source-ranges=0.0.0.0/0

gcloud compute firewall-rules create allow-https \
  --allow tcp:443 \
  --target-tags=https-server \
  --source-ranges=0.0.0.0/0
```

#### 2. Application Deployment

```bash
# SSH into VM
gcloud compute ssh glitch-core-vm --zone=us-central1-a

# Install Docker and deploy
# (Same as Azure/AWS steps)
```

## 🔧 Production Configuration

### Environment Variables

```bash
# .env.prod
ENV=production
LOG_LEVEL=INFO
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b

# Security
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=api.cognitive-drift.app,localhost

# Performance
WORKER_PROCESSES=4
MAX_CONNECTIONS=1000
TIMEOUT=30

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    deploy:
      resources:
        limits:
          memory: 2G
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        limits:
          memory: 6G
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

volumes:
  qdrant_data:
  redis_data:
  ollama_data:
  prometheus_data:
```

### Nginx Production Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream glitch_core {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name api.cognitive-drift.app;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.cognitive-drift.app;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://glitch_core;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://glitch_core;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 86400;
        }

        # Health check
        location /health {
            proxy_pass http://glitch_core;
            access_log off;
        }

        # Metrics (internal only)
        location /metrics {
            allow 127.0.0.1;
            deny all;
            proxy_pass http://glitch_core;
        }
    }
}
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert.rules"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'glitch-core-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Glitch Core Metrics",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Experiments",
        "type": "stat",
        "targets": [
          {
            "expr": "glitch_core_active_experiments",
            "legendFormat": "Experiments"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{container=\"api\"}",
            "legendFormat": "API Memory"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# monitoring/alert.rules
groups:
  - name: glitch-core-alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API response time is high"
          description: "95th percentile response time is {{ $value }}s"

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{container="api"} > 3e9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "API container is using {{ $value }} bytes"

      - alert: ServiceDown
        expr: up{job="glitch-core-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API service is down"
          description: "Glitch Core API is not responding"
```

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
# Generate self-signed certificate for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=api.cognitive-drift.app"

# For production, use Let's Encrypt
sudo certbot --nginx -d api.cognitive-drift.app
```

### Firewall Configuration

```bash
# UFW firewall setup
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -P INPUT DROP
```

### Security Headers

```nginx
# Additional security headers
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()";
```

## 📈 Scaling Considerations

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  api:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    environment:
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
```

### Load Balancer Configuration

```nginx
# nginx/load-balancer.conf
upstream glitch_core {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    server_name api.cognitive-drift.app;
    
    location / {
        proxy_pass http://glitch_core;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Database Scaling

```yaml
# Qdrant cluster configuration
services:
  qdrant1:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    volumes:
      - qdrant1_data:/qdrant/storage

  qdrant2:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    volumes:
      - qdrant2_data:/qdrant/storage

  qdrant3:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    volumes:
      - qdrant3_data:/qdrant/storage
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: your-registry/glitch-core:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Deploy to server
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.KEY }}
          script: |
            cd /opt/glitch-core
            docker-compose pull
            docker-compose up -d
            docker system prune -f
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

echo "Deploying Glitch Core..."

# Pull latest changes
git pull origin main

# Build new image
docker-compose build

# Stop current services
docker-compose down

# Start with new image
docker-compose up -d

# Wait for health check
echo "Waiting for services to be healthy..."
for i in {1..30}; do
  if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "Deployment successful!"
    exit 0
  fi
  sleep 2
done

echo "Deployment failed - services not healthy"
exit 1
```

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Domain DNS configured
- [ ] Firewall rules set
- [ ] Monitoring configured
- [ ] Backup strategy defined

### Deployment

- [ ] Docker images built
- [ ] Services started
- [ ] Health checks passing
- [ ] SSL certificate installed
- [ ] Nginx configured
- [ ] Monitoring active

### Post-Deployment

- [ ] Load testing completed
- [ ] Monitoring alerts configured
- [ ] Backup tested
- [ ] Documentation updated
- [ ] Team notified

## 🚨 Emergency Procedures

### Rollback

```bash
# Quick rollback to previous version
docker-compose down
docker tag glitch-core:previous glitch-core:latest
docker-compose up -d
```

### Disaster Recovery

```bash
# Restore from backup
docker-compose down
docker volume rm glitch_core_qdrant_data glitch_core_redis_data
docker volume create glitch_core_qdrant_data
docker volume create glitch_core_redis_data
# Restore backup data
docker-compose up -d
```

---

**For more information, see the [Developer Guide](developer-guide.md) and [Troubleshooting Guide](troubleshooting.md).** 