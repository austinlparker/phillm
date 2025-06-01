# Redis Infrastructure Configuration Changes

## Summary

Updated the infrastructure configuration to use Redis OSS 8.x running as an ECS Fargate container with EFS persistence, since AWS ElastiCache doesn't support Redis 8.0 yet.

## Changes Made

### 1. Terraform Provider Version
- **Updated**: AWS Provider from `~> 5.0` to `~> 5.99` (latest stable as of Dec 2024)
- **Note**: Version 6.0 is in beta but not recommended for production yet

### 2. ECS-Based Redis Configuration

#### Added New Resources:
- **Security Group**: `aws_security_group.redis` - Allows ECS tasks to connect to Redis on port 6379
- **Security Group**: `aws_security_group.efs` - Allows Redis container to access EFS storage
- **EFS File System**: `aws_efs_file_system.redis` - Persistent storage for Redis data
- **EFS Mount Targets**: `aws_efs_mount_target.redis` - EFS access points in private subnets
- **ECS Task Definition**: `aws_ecs_task_definition.redis` - Redis container configuration
- **ECS Service**: `aws_ecs_service.redis` - Redis service with service discovery
- **Service Discovery**: `aws_service_discovery_*` - DNS-based service discovery

#### Redis Specifications:
- **Container Image**: `redis:8-alpine` (Redis OSS 8.x, latest patch version)
- **CPU/Memory**: 256 CPU units / 512 MiB memory (configurable)
- **Persistence**: EFS-backed persistent storage
- **Authentication**: Password-based authentication
- **High Availability**: Single instance (as requested)

### 3. Application Deployment
- **Instance Count**: Changed from 2 to 1 instance (`app_count = 1`)
- **Reason**: Application is designed to run as a single instance

### 4. New Variables Added
```hcl
variable "redis_cpu" {
  description = "CPU units for Redis container (1 vCPU = 1024 CPU units)"
  type        = string
  default     = "256"
}

variable "redis_memory" {
  description = "Memory for Redis container (in MiB)"
  type        = string
  default     = "512"
}

variable "redis_password" {
  description = "Redis password for authentication"
  type        = string
  sensitive   = true
  default     = ""
}
```

### 5. SSM Parameters Updated
- **Redis URL**: Now uses service discovery endpoint: `redis://:${password}@redis.phillm.local:6379`
- **Redis Password**: New SSM parameter for Redis authentication

### 6. New Outputs Added
- `redis_service_discovery_endpoint` - Service discovery DNS name
- `redis_port` - Redis port (6379)
- `redis_password_ssm_parameter` - SSM parameter name for password
- `redis_url_ssm_parameter` - SSM parameter name for Redis URL
- `redis_efs_id` - EFS file system ID for persistence
- `redis_service_name` - ECS service name
- `redis_task_definition_family` - ECS task definition family

### 7. OpenTelemetry Enhancements

#### Added AWS Resource Detection:
- **New Dependency**: `opentelemetry-resourcedetector-aws>=1.21.0`
- **Resource Detectors**: ECS, EC2, EKS, Lambda, Elastic Beanstalk
- **Automatic Detection**: Detects AWS environment and adds appropriate resource attributes

#### AWS Resource Detectors Added:
```python
aws_detectors = [
    AwsEcsResourceDetector(),      # For ECS Fargate
    AwsEc2ResourceDetector(),      # For EC2 instances  
    AwsEksResourceDetector(),      # For EKS
    BeanstalkResourceDetector(),   # For Elastic Beanstalk
    AwsLambdaResourceDetector(),   # For Lambda
]
```

## Security Features

### Redis Security:
- **Encryption at Rest**: EFS encryption enabled
- **Authentication**: Password-based authentication
- **Network Security**: Security groups restrict access to ECS tasks only
- **Private Network**: Redis deployed in private subnets only
- **Container Isolation**: Redis runs in its own ECS task

### Access Control:
- **Security Group Rules**: Only application ECS tasks can connect to Redis
- **EFS Security**: Separate security group for EFS access
- **SSM Parameters**: All secrets stored securely in Parameter Store
- **IAM Integration**: ECS tasks retrieve credentials via IAM roles

### Data Persistence:
- **EFS Storage**: Persistent data storage across container restarts
- **Backup Strategy**: Redis RDB snapshots with configurable intervals
- **AOF Logging**: Append-only file for data durability

## Deployment Notes

### Before Deploying:
1. **Set Password**: Update the `redis_password` variable with a secure password
2. **Review Resources**: Consider adjusting CPU/memory for production workloads
3. **EFS Performance**: Consider performance mode settings for high IOPS

### After Deployment:
1. **Verify Connectivity**: Application should connect via `redis.phillm.local:6379`
2. **Monitor Logs**: CloudWatch logs available for Redis container
3. **Check Health**: ECS health checks monitor Redis availability
4. **Check Telemetry**: AWS resource attributes should appear in Honeycomb

### Redis Configuration:
```redis
# Configured Redis settings:
requirepass <password>         # Password authentication
appendonly yes                 # AOF persistence enabled
appendfsync everysec          # AOF sync every second
dir /data                     # Data directory (mounted from EFS)
save 900 1                    # Save snapshot if 1 key changed in 900 sec
save 300 10                   # Save snapshot if 10 keys changed in 300 sec
save 60 10000                 # Save snapshot if 10000 keys changed in 60 sec
```

## Cost Optimization

### Current Configuration:
- **Fargate**: 256 CPU / 512 MiB memory - Minimal cost for development
- **EFS**: Provisioned throughput mode (10 MiB/s)
- **Storage**: Only pay for data stored and throughput used

### Scaling Options:
- **CPU/Memory**: Increase for better performance
- **EFS Throughput**: Adjust based on I/O requirements
- **Multi-AZ**: EFS automatically provides multi-AZ durability

## Redis OSS Version Notes

- **Requested**: Redis OSS 8.x or better
- **Deployed**: `redis:8-alpine` - Latest Redis 8.x Docker image
- **Automatic Updates**: Using tag `8-alpine` gets latest 8.x patch versions
- **Future Proof**: Will automatically get Redis 8.1, 8.2, etc. when available

## Service Discovery

The Redis service is accessible via DNS at `redis.phillm.local:6379` from any ECS task in the same VPC.

## Verification Commands

```bash
# Check Terraform configuration
terraform validate
terraform plan

# Verify ECS services after deployment
aws ecs list-services --cluster phillm-cluster
aws ecs describe-services --cluster phillm-cluster --services phillm-redis

# Check service discovery
aws servicediscovery list-services

# Test Redis connectivity from application container
redis-cli -h redis.phillm.local -p 6379 -a <password> ping

# Check EFS mount
aws efs describe-file-systems --file-system-id <efs-id>

# Monitor Redis logs
aws logs tail /ecs/phillm-redis --follow
```