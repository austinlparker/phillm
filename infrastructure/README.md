# AWS Deployment Guide

This guide explains how to deploy PhiLLM to AWS using Terraform and GitHub Actions.

## Architecture Overview

The deployment uses the following AWS services:

- **ECS Fargate**: Container orchestration for the application
- **ECR**: Container registry for Docker images
- **Application Load Balancer**: HTTP load balancing and health checks
- **VPC**: Isolated network with public/private subnets
- **SSM Parameter Store**: Secure storage for secrets
- **CloudWatch**: Logging and monitoring
- **IAM**: Roles and policies for secure access

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

# Redis URL (if using external Redis)
aws ssm put-parameter \
  --name "/phillm/redis_url" \
  --value "redis://your-redis-host:6379" \
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

The application requires Redis for vector storage. You have several options:

### Option 1: AWS ElastiCache (Recommended for Production)

Add to your Terraform configuration:

```terraform
resource "aws_elasticache_subnet_group" "phillm" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_cluster" "phillm" {
  cluster_id           = var.project_name
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.phillm.name
  security_group_ids   = [aws_security_group.redis.id]
}
```

### Option 2: External Redis Provider

Use a managed Redis service like:
- **Redis Cloud**
- **Upstash**
- **Digital Ocean Managed Redis**

Update the Redis URL in SSM Parameter Store accordingly.

### Option 3: Self-hosted Redis Container

Add a Redis container to your ECS task definition (not recommended for production).

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

### Load Balancer Health Checks

Verify the health endpoint is responding:

```bash
curl http://<load-balancer-dns>/health
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

1. **Use appropriate instance sizes** - Start with small and scale up
2. **Enable container insights selectively** - Can increase costs
3. **Set CloudWatch log retention** - Default is 7 days
4. **Consider spot instances** for non-critical environments
5. **Use Reserved Instances** for predictable workloads

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