# PhiLLM Infrastructure Deployment Guide

## 🚀 Cost-Optimized AWS Deployment

### Architecture Overview
This deployment uses a **simplified, cost-optimized architecture** that saves ~$80/month:
- **Single AZ with public subnets** - No NAT Gateway needed (~$45/month savings)
- **Direct service access** - No Application Load Balancer (~$20/month savings)
- **EFS Burst Mode** - Cost-effective storage (~$15/month savings)
- **Containerized Redis** - Self-managed with persistent EFS storage

### Prerequisites
- AWS CLI configured with appropriate permissions
- OpenTofu >= 1.0 installed (or Terraform >= 1.0)
- Docker (for building and pushing images)
- An S3 bucket for Terraform state (optional but recommended)

## Step 1: Prepare Your Environment

### 1.1 Clone and Navigate
```bash
cd /Users/ap/Documents/code/phiLLM/infrastructure/terraform
```

### 1.2 Configure Terraform Backend (Optional but Recommended)
```bash
# Create an S3 bucket for Terraform state
aws s3 mb s3://your-phillm-terraform-state-bucket

# Edit main.tf and uncomment/configure the backend:
# backend "s3" {
#   bucket = "your-phillm-terraform-state-bucket"
#   key    = "phillm/terraform.tfstate"
#   region = "us-east-1"
# }
```

### 1.3 Install Dependencies
```bash
# Update Python dependencies (includes new OpenTelemetry AWS detector)
cd /Users/ap/Documents/code/phiLLM
uv sync --all-extras
```

## Step 2: Configure Secrets

### 2.1 Set Required Variables
Create a `terraform.tfvars` file:

```bash
cat > terraform.tfvars << 'EOF'
# Basic Configuration
aws_region = "us-east-1"
environment = "production"

# Application Resources
app_count = 1
fargate_cpu = "512"
fargate_memory = "1024"

# Redis Configuration
redis_cpu = "256"
redis_memory = "512"
redis_password = "your-secure-redis-password-here"

# Required Secrets (set these to your actual values)
openai_api_key = "sk-your-openai-api-key"
slack_bot_token = "xoxb-your-slack-bot-token"
slack_signing_secret = "your-slack-signing-secret"
slack_app_token = "xapp-your-slack-app-token"
honeycomb_api_key = "your-honeycomb-api-key"
EOF
```

### 2.2 Secure Your Variables File
```bash
# Make sure terraform.tfvars is not committed to git
echo "*.tfvars" >> .gitignore
chmod 600 terraform.tfvars
```

## Step 3: Deploy Infrastructure

### 3.1 Initialize Terraform
```bash
terraform init
```

### 3.2 Plan Deployment
```bash
terraform plan
```

### 3.3 Deploy Infrastructure
```bash
terraform apply
```

**⚠️ This will create AWS resources that incur costs!**

## Step 4: Build and Deploy Application

### 4.1 Get ECR Repository URL
```bash
ECR_REPO=$(terraform output -raw ecr_repository_url)
echo "ECR Repository: $ECR_REPO"
```

### 4.2 Build and Push Docker Image
```bash
# Navigate to project root
cd /Users/ap/Documents/code/phiLLM

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO

# Build image
docker build -t phillm .

# Tag and push
docker tag phillm:latest $ECR_REPO:latest
docker push $ECR_REPO:latest
```

### 4.3 Update ECS Service
```bash
# Force new deployment to pull latest image
aws ecs update-service \
  --cluster phillm-cluster \
  --service phillm \
  --force-new-deployment
```

## Step 5: Configure Secrets in AWS

### 5.1 Update SSM Parameters (if needed)
```bash
# Redis password (if you need to change it)
aws ssm put-parameter \
  --name "/phillm/redis_password" \
  --value "your-secure-redis-password" \
  --type "SecureString" \
  --overwrite

# OpenAI API Key
aws ssm put-parameter \
  --name "/phillm/openai_api_key" \
  --value "sk-your-actual-openai-key" \
  --type "SecureString" \
  --overwrite

# Slack Bot Token
aws ssm put-parameter \
  --name "/phillm/slack_bot_token" \
  --value "xoxb-your-actual-bot-token" \
  --type "SecureString" \
  --overwrite

# Continue for other secrets...
```

## Step 6: Verify Deployment

### 6.1 Check Infrastructure Status
```bash
# Check ECS services
aws ecs describe-services \
  --cluster phillm-cluster \
  --services phillm phillm-redis

# Get application public IP
aws ecs describe-tasks \
  --cluster phillm-cluster \
  --tasks $(aws ecs list-tasks --cluster phillm-cluster --service-name phillm --query 'taskArns[0]' --output text) \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | \
  xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text
```

### 6.2 Check Application Health
```bash
# Get the public IP from previous step and test health endpoint
PUBLIC_IP="<ip-from-above-command>"
curl http://$PUBLIC_IP:3000/health

# Check application logs
aws logs tail /ecs/phillm --follow

# Check Redis logs
aws logs tail /ecs/phillm-redis --follow
```

### 6.3 Test Redis Connectivity
```bash
# Get Redis connection details
REDIS_ENDPOINT=$(terraform output -raw redis_service_discovery_endpoint)
echo "Redis endpoint: $REDIS_ENDPOINT"

# Test from within the VPC (you'll need to exec into a container)
# aws ecs execute-command --cluster phillm-cluster --task <task-id> --container phillm --interactive --command "/bin/sh"
# Then inside container: redis-cli -h redis.phillm.local -p 6379 -a <password> ping
```

## Step 7: Environment-Specific Configuration

### 7.1 Development Environment
```bash
# Use smaller resources for dev
redis_cpu = "128"
redis_memory = "256"
fargate_cpu = "256"
fargate_memory = "512"
log_retention_days = 3
```

### 7.2 Production Environment
```bash
# Use larger resources for production
redis_cpu = "512"
redis_memory = "1024"
fargate_cpu = "1024"
fargate_memory = "2048"
log_retention_days = 30
app_count = 1  # Still single instance as requested
```

## 🔧 Troubleshooting

### Common Issues

#### 1. ECS Tasks Not Starting
```bash
# Check task definition
aws ecs describe-task-definition --task-definition phillm

# Check service events
aws ecs describe-services --cluster phillm-cluster --services phillm

# Check task logs
aws logs describe-log-streams --log-group-name /ecs/phillm
```

#### 2. Redis Connection Issues
```bash
# Check Redis service health
aws ecs describe-services --cluster phillm-cluster --services phillm-redis

# Check security groups
aws ec2 describe-security-groups --filters "Name=group-name,Values=phillm-*"

# Test service discovery
nslookup redis.phillm.local
```

#### 3. Application Not Accessible
```bash
# Check if tasks have public IPs
aws ecs describe-tasks \
  --cluster phillm-cluster \
  --tasks $(aws ecs list-tasks --cluster phillm-cluster --service-name phillm --query 'taskArns[0]' --output text) \
  --query 'tasks[0].attachments[0].details'

# Check security group rules (should allow port 3000 from 0.0.0.0/0)
aws ec2 describe-security-groups --filters "Name=group-name,Values=phillm-ecs-tasks"
```

#### 4. SSM Parameter Issues
```bash
# List all parameters
aws ssm describe-parameters --filters "Key=Name,Values=/phillm/"

# Check parameter value (be careful with secrets!)
aws ssm get-parameter --name "/phillm/redis_url" --with-decryption
```

## 🔄 Updates and Maintenance

### Update Application Code
```bash
# Build and push new image
docker build -t phillm .
docker tag phillm:latest $ECR_REPO:latest
docker push $ECR_REPO:latest

# Force deployment
aws ecs update-service --cluster phillm-cluster --service phillm --force-new-deployment
```

### Update Infrastructure
```bash
# Make changes to Terraform files
terraform plan
terraform apply
```

### Scale Resources
```bash
# Update variables in terraform.tfvars
redis_cpu = "512"      # Double Redis CPU
redis_memory = "1024"  # Double Redis memory

# Apply changes
terraform apply
```

## 🧹 Cleanup

### Destroy Infrastructure
```bash
# ⚠️ This will delete everything!
terraform destroy
```

### Cleanup Docker Images
```bash
# Remove local images
docker rmi phillm:latest
docker rmi $ECR_REPO:latest
```

## 📊 Monitoring and Observability

### Key Metrics to Monitor
- **ECS Service Health**: Task count, CPU/memory utilization
- **Redis Performance**: Connection count, memory usage, command stats
- **Network Performance**: Direct public IP access, security group rules
- **EFS Performance**: IOPS, throughput, storage utilization
- **Cost Optimization**: No NAT Gateway, ALB, or provisioned EFS throughput charges

### Useful CloudWatch Dashboards
```bash
# Create a custom dashboard with key metrics
aws cloudwatch put-dashboard --dashboard-name "PhiLLM-Overview" --dashboard-body file://dashboard.json
```

### Honeycomb Integration
With OpenTelemetry AWS resource detectors, you should see:
- ECS cluster and service information
- Task and container metadata  
- AWS region and availability zone data
- Redis instrumentation traces
- Cost-optimized architecture telemetry

### Cost Monitoring
This simplified architecture saves approximately $80/month:
- NAT Gateway: ~$45/month saved
- Application Load Balancer: ~$20/month saved  
- EFS Provisioned Throughput: ~$15/month saved
- Single AZ deployment: Reduced data transfer costs

## 🎯 Next Steps

1. **Set up CI/CD**: Consider GitHub Actions for automated deployments
2. **Add Monitoring**: Set up CloudWatch alarms for critical metrics
3. **Backup Strategy**: Configure EFS backup policies
4. **Security Hardening**: Review IAM policies and security groups
5. **Performance Tuning**: Monitor and adjust resource allocations based on usage

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review CloudWatch logs for both services
3. Verify all SSM parameters are set correctly
4. Ensure Docker image is built and pushed successfully

The infrastructure is now ready to run Redis OSS 8.x with full persistence, monitoring, and AWS integration! 🚀