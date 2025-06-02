# AWS Deployment Guide

This guide explains how to deploy PhiLLM to AWS using Terraform and GitHub Actions.

## Architecture Overview

The deployment uses the following AWS services:

- **ECS Fargate**: Container orchestration for the application and Redis
- **ECR**: Container registry for Docker images
- **VPC**: Simplified single-AZ network with public subnets
- **EFS**: Persistent storage for Redis data
- **Service Discovery**: Internal DNS for service communication
- **SSM Parameter Store**: Secure storage for secrets
- **CloudWatch**: Logging and monitoring
- **IAM**: Roles and policies for secure access

### Cost-Optimized Design

This infrastructure is designed for cost efficiency (~$80/month savings):
- **No NAT Gateway**: Services run in public subnets with direct internet access
- **No Application Load Balancer**: Direct service access via public IPs
- **EFS Burst Mode**: Cost-effective file storage for Redis persistence
- **Single AZ**: Simplified deployment for development/staging environments

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** installed locally (>= 1.0)
3. **AWS CLI** configured with your credentials
4. **Docker** installed for local testing
5. **GitHub repository** for CI/CD

## Deployment Steps

### 1. Infrastructure Setup

#### Configure Terraform Backend (Recommended)

Create an S3 bucket for Terraform state:

```bash
aws s3 mb s3://your-terraform-state-bucket
aws s3api put-bucket-versioning \
  --bucket your-terraform-state-bucket \
  --versioning-configuration Status=Enabled
```

Update `infrastructure/terraform/main.tf` with your bucket details:

```terraform
backend "s3" {
  bucket = "your-terraform-state-bucket"
  key    = "phillm/terraform.tfstate"
  region = "us-east-1"
}
```

#### Deploy Infrastructure

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply infrastructure
terraform apply
```

#### Set Secrets in SSM Parameter Store

After infrastructure is deployed, set your actual secrets:

```bash
# OpenAI API Key
aws ssm put-parameter \
  --name "/phillm/openai_api_key" \
  --value "sk-your-actual-openai-key" \
  --type "SecureString" \
  --overwrite

# Slack Bot Token
aws ssm put-parameter \
  --name "/phillm/slack_bot_token" \
  --value "xoxb-your-actual-slack-token" \
  --type "SecureString" \
  --overwrite

# Slack Signing Secret
aws ssm put-parameter \
  --name "/phillm/slack_signing_secret" \
  --value "your-actual-signing-secret" \
  --type "SecureString" \
  --overwrite

# Slack App Token
aws ssm put-parameter \
  --name "/phillm/slack_app_token" \
  --value "xapp-your-actual-app-token" \
  --type "SecureString" \
  --overwrite

# Honeycomb API Key
aws ssm put-parameter \
  --name "/phillm/honeycomb_api_key" \
  --value "your-actual-honeycomb-key" \
  --type "SecureString" \
  --overwrite

# Redis URL (uses internal service discovery)
aws ssm put-parameter \
  --name "/phillm/redis_url" \
  --value "redis://:your-password@redis.phillm.local:6379" \
  --type "SecureString" \
  --overwrite

# Redis Password (for the containerized Redis)
aws ssm put-parameter \
  --name "/phillm/redis_password" \
  --value "your-secure-redis-password" \
  --type "SecureString" \
  --overwrite
```

### 2. GitHub Actions Setup

#### Configure Repository Secrets

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

```
AWS_REGION=us-east-1
AWS_ROLE_ARN=<github_actions_role_arn from terraform output>
ECR_REPOSITORY=phillm
ECS_CLUSTER=phillm-cluster
ECS_SERVICE=phillm
ECS_TASK_DEFINITION=phillm
SLACK_WEBHOOK_URL=<optional: for deployment notifications>
```

#### Update GitHub OIDC Configuration

In `infrastructure/terraform/iam.tf`, update the GitHub repository reference:

```terraform
StringLike = {
  "token.actions.githubusercontent.com:sub" = "repo:YOUR_GITHUB_USERNAME/phillm:*"
}
```

Then run `terraform apply` again.

### 3. Initial Deployment

#### Push Docker Image

```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push initial image
docker build -t phillm .
docker tag phillm:latest <ecr-repository-url>:latest
docker push <ecr-repository-url>:latest
```

#### Deploy via GitHub Actions

Push to the `main` branch to trigger automatic deployment:

```bash
git add .
git commit -m "Initial deployment setup"
git push origin main
```

## Redis Setup

The application includes a **containerized Redis instance** running on ECS Fargate with persistent EFS storage.

### Included Redis Configuration

- **Redis 8.0.2 Alpine**: Lightweight, official Redis image
- **EFS Persistence**: Data persists across container restarts
- **Service Discovery**: Accessible at `redis.phillm.local:6379`
- **Security**: Password-protected with configurable authentication
- **Backup Strategy**: Automatic saves (900s/1change, 300s/10changes, 60s/10000changes)

### Redis Connection

The Redis URL is automatically configured as: `redis://:password@redis.phillm.local:6379`

Update the password in SSM Parameter Store:

```bash
aws ssm put-parameter \
  --name "/phillm/redis_password" \
  --value "your-secure-redis-password" \
  --type "SecureString" \
  --overwrite
```

### Alternative: External Redis

For production, consider managed Redis services:
- **AWS ElastiCache**: Fully managed, high availability
- **Redis Cloud**: Global, multi-cloud Redis service
- **Upstash**: Serverless Redis with pay-per-request pricing

Update the Redis URL in SSM Parameter Store to use external services.

## Monitoring and Troubleshooting

### CloudWatch Logs

View application logs:

```bash
aws logs tail /ecs/phillm --follow
```

### ECS Service Status

Check service health:

```bash
aws ecs describe-services \
  --cluster phillm-cluster \
  --services phillm
```

### Service Health Checks

Get the public IP and verify the health endpoint:

```bash
# Get service public IP
aws ecs describe-tasks \
  --cluster phillm-cluster \
  --tasks $(aws ecs list-tasks --cluster phillm-cluster --service-name phillm --query 'taskArns[0]' --output text) \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | \
  xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text

# Test health endpoint
curl http://<public-ip>:3000/health
```

### Common Issues

1. **Service fails to start**: Check CloudWatch logs for errors
2. **Health checks failing**: Ensure `/health` endpoint is accessible
3. **Image pull errors**: Verify ECR permissions and image exists
4. **Secret access errors**: Check IAM roles and SSM parameter names

## Scaling

### Horizontal Scaling

Update the ECS service desired count:

```bash
aws ecs update-service \
  --cluster phillm-cluster \
  --service phillm \
  --desired-count 3
```

### Vertical Scaling

Update CPU/memory in the task definition via Terraform:

```terraform
variable "fargate_cpu" {
  default = "1024"  # 1 vCPU
}

variable "fargate_memory" {
  default = "2048"  # 2 GB
}
```

Then run `terraform apply`.

## Cost Optimization

This infrastructure is already optimized for cost (~$80/month savings):

### Built-in Optimizations
1. **No NAT Gateway** (~$45/month savings) - Direct internet access via public subnets
2. **No Application Load Balancer** (~$20/month savings) - Direct service access
3. **EFS Burst Mode** (~$15/month savings) - Pay only for storage used
4. **Container Insights Disabled** - Can be enabled if monitoring needed
5. **Short Log Retention** - 7 days to minimize CloudWatch costs
6. **Single AZ** - Reduced complexity and data transfer costs

### Additional Optimizations
1. **Right-size containers** - Start with small CPU/memory and scale up
2. **Monitor usage** - Use CloudWatch metrics to optimize resource allocation
3. **Reserved Capacity** - Consider Savings Plans for predictable workloads
4. **Spot Instances** - For development environments (requires Terraform changes)

## Security Best Practices

1. **Rotate secrets regularly** in SSM Parameter Store
2. **Use least privilege IAM policies**
3. **Enable VPC Flow Logs** for network monitoring
4. **Set up AWS Config** for compliance monitoring
5. **Enable GuardDuty** for threat detection
6. **Use ACM certificates** for HTTPS (add to ALB listener)

## Backup and Disaster Recovery

1. **Database backups**: Ensure Redis persistence is configured
2. **Infrastructure as Code**: Keep Terraform state backed up
3. **Multi-region deployment**: Consider for high availability
4. **Automated testing**: Ensure deployments are validated

## Next Steps

1. Set up custom domain with Route 53 and ACM
2. Configure HTTPS with SSL certificates
3. Set up monitoring dashboards in CloudWatch
4. Configure log aggregation and alerting
5. Implement blue-green deployments for zero-downtime updates