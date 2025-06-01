output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.phillm.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.phillm.name
}

output "ecs_task_definition_family" {
  description = "Family of the ECS task definition"
  value       = aws_ecs_task_definition.phillm.family
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_url" {
  description = "URL of the load balancer"
  value       = "http://${aws_lb.main.dns_name}"
}

output "github_actions_role_arn" {
  description = "ARN of the GitHub Actions IAM role"
  value       = aws_iam_role.github_actions_role.arn
}

output "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.phillm.name
}

# Redis ECS outputs
output "redis_service_discovery_endpoint" {
  description = "Service discovery endpoint for Redis"
  value       = "redis.${var.project_name}.local"
}

output "redis_port" {
  description = "Port of the Redis service"
  value       = 6379
}

output "redis_password_ssm_parameter" {
  description = "SSM parameter name for Redis password"
  value       = aws_ssm_parameter.redis_password.name
}

output "redis_url_ssm_parameter" {
  description = "SSM parameter name for Redis URL"
  value       = aws_ssm_parameter.redis_url.name
}

output "redis_efs_id" {
  description = "EFS file system ID for Redis data persistence"
  value       = aws_efs_file_system.redis.id
}

output "redis_service_name" {
  description = "Name of the Redis ECS service"
  value       = aws_ecs_service.redis.name
}

output "redis_task_definition_family" {
  description = "Family of the Redis ECS task definition"
  value       = aws_ecs_task_definition.redis.family
}