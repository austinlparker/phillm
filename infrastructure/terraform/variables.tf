variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "phillm"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "fargate_cpu" {
  description = "Fargate instance CPU units to provision (1 vCPU = 1024 CPU units)"
  type        = string
  default     = "512"
}

variable "fargate_memory" {
  description = "Fargate instance memory to provision (in MiB)"
  type        = string
  default     = "1024"
}

variable "app_count" {
  description = "Number of app instances to run"
  type        = number
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

# Secrets - these should be set via environment variables or Terraform Cloud
variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "max_response_tokens" {
  description = "Maximum tokens for AI responses"
  type        = string
  default     = "3000"
}

variable "slack_bot_token" {
  description = "Slack bot token"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_signing_secret" {
  description = "Slack signing secret"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_app_token" {
  description = "Slack app token"
  type        = string
  sensitive   = true
  default     = ""
}

variable "honeycomb_api_key" {
  description = "Honeycomb API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "redis_url" {
  description = "Redis connection URL"
  type        = string
  sensitive   = true
  default     = ""
}

# Redis ECS configuration
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

variable "target_user_id" {
  description = "Target user ID for PhiLLM scraping"
  type        = string
  default     = ""
}

variable "scrape_channels" {
  description = "Comma-separated list of channels for PhiLLM scraping"
  type        = string
  default     = ""
}

variable "style_similarity_threshold" {
  description = "Similarity threshold for style transfer examples (0.0-1.0)"
  type        = string
  default     = "0.3"
}

variable "conversation_distance_threshold" {
  description = "Distance threshold for conversation context retrieval (0.0-1.0)"
  type        = string
  default     = "0.8"
}