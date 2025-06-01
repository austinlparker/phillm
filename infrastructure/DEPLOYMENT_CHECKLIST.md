# 🚀 PhiLLM Deployment Checklist

## ✅ Pre-Deployment Checklist

### 📋 Prerequisites
- [ ] AWS CLI installed and configured
- [ ] Terraform >= 1.0 installed  
- [ ] Docker installed and running
- [ ] Required API keys and tokens ready:
  - [ ] OpenAI API key
  - [ ] Slack Bot Token
  - [ ] Slack Signing Secret
  - [ ] Slack App Token
  - [ ] Honeycomb API key

### 🔧 Configuration
- [ ] Create `terraform.tfvars` with all required variables
- [ ] Set secure Redis password
- [ ] Review resource sizing (CPU/memory)
- [ ] Configure Terraform backend S3 bucket (optional)

## ✅ Deployment Steps

### 🏗️ Infrastructure
- [ ] `terraform init`
- [ ] `terraform plan` (review changes)
- [ ] `terraform apply` (deploy infrastructure)
- [ ] Verify all resources created successfully

### 🐳 Application
- [ ] Get ECR repository URL from Terraform output
- [ ] Build Docker image: `docker build -t phillm .`
- [ ] Push to ECR
- [ ] Force ECS deployment to pull new image

### 🔐 Secrets
- [ ] Update SSM parameters with real values (not REPLACE_ME)
- [ ] Verify all secrets are accessible by ECS tasks

## ✅ Post-Deployment Verification

### 🔍 Health Checks
- [ ] ECS services running (both phillm and phillm-redis)
- [ ] Load balancer health check passing
- [ ] Application responds to `/health` endpoint
- [ ] Redis container healthy and accessible

### 📊 Monitoring
- [ ] CloudWatch logs streaming for both services
- [ ] Honeycomb receiving telemetry data
- [ ] AWS resource attributes visible in traces

### 🧪 Functionality Tests
- [ ] Slack bot responds to messages
- [ ] Vector storage working (check Redis keys)
- [ ] AI completions generating responses
- [ ] Memory system storing conversations

## 🎯 Quick Verification Commands

```bash
# Check everything is running
terraform output
aws ecs describe-services --cluster phillm-cluster --services phillm phillm-redis

# Test application
curl $(terraform output -raw load_balancer_url)/health

# Check logs
aws logs tail /ecs/phillm --follow
aws logs tail /ecs/phillm-redis --follow

# Verify Redis
aws ecs execute-command --cluster phillm-cluster --task <task-id> --container phillm --interactive --command "/bin/sh"
# Then: redis-cli -h redis.phillm.local -p 6379 -a <password> ping
```

## 🚨 Common Issues & Solutions

### ❌ ECS Tasks Failing to Start
**Check**: Task definition, security groups, SSM parameter access
```bash
aws ecs describe-services --cluster phillm-cluster --services phillm
aws logs describe-log-streams --log-group-name /ecs/phillm
```

### ❌ Redis Connection Failed
**Check**: Service discovery, security groups, Redis password
```bash
aws ecs describe-services --cluster phillm-cluster --services phillm-redis
aws servicediscovery list-services
```

### ❌ Load Balancer 503 Errors
**Check**: Target group health, application startup time
```bash
aws elbv2 describe-target-health --target-group-arn $(terraform output -raw target_group_arn)
```

### ❌ SSM Parameter Access Denied
**Check**: IAM roles and policies
```bash
aws iam get-role-policy --role-name phillm-ecs-task-role --policy-name phillm-ecs-task-policy
```

## 🔄 Update Workflow

### Code Changes
1. [ ] Make code changes
2. [ ] Build new Docker image
3. [ ] Push to ECR
4. [ ] Force ECS deployment

### Infrastructure Changes
1. [ ] Modify Terraform files
2. [ ] `terraform plan`
3. [ ] `terraform apply`

## 🧹 Cleanup Instructions

### Temporary Cleanup (keep infrastructure)
```bash
# Scale down to 0 (saves costs)
aws ecs update-service --cluster phillm-cluster --service phillm --desired-count 0
aws ecs update-service --cluster phillm-cluster --service phillm-redis --desired-count 0
```

### Full Cleanup
```bash
# ⚠️ DESTROYS EVERYTHING
terraform destroy
```

## 📈 Expected Costs (Rough Estimates)

### Development Configuration:
- **ECS Fargate**: ~$15-30/month (2 small tasks)
- **EFS**: ~$3-5/month (minimal storage)
- **Load Balancer**: ~$16/month
- **CloudWatch**: ~$1-3/month
- **Total**: ~$35-55/month

### Production Configuration:
- **ECS Fargate**: ~$30-60/month (larger tasks)
- **EFS**: ~$5-15/month (more storage/throughput)
- **Load Balancer**: ~$16/month
- **CloudWatch**: ~$5-10/month
- **Total**: ~$55-100/month

*Costs vary by region and actual usage*

## 🎉 Success Indicators

You'll know everything is working when:

✅ **Infrastructure**: All Terraform resources created successfully  
✅ **Application**: ECS tasks running and healthy  
✅ **Redis**: Container running with persistent storage  
✅ **Networking**: Load balancer routing traffic correctly  
✅ **Slack**: Bot responding to messages  
✅ **AI**: Generating contextual responses  
✅ **Storage**: Vector embeddings persisting in Redis  
✅ **Monitoring**: Telemetry flowing to Honeycomb  

## 📞 Need Help?

If you run into issues:
1. Check the troubleshooting section in DEPLOYMENT_GUIDE.md
2. Review CloudWatch logs for both services
3. Verify Terraform outputs match expected values
4. Ensure all SSM parameters have real values (not REPLACE_ME)

---

**Ready to deploy? Start with the [Deployment Guide](DEPLOYMENT_GUIDE.md)! 🚀**