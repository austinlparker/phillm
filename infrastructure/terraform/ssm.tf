# SSM Parameters for storing secrets securely
resource "aws_ssm_parameter" "openai_api_key" {
  name  = "/${var.project_name}/openai_api_key"
  type  = "SecureString"
  value = var.openai_api_key != "" ? var.openai_api_key : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "slack_bot_token" {
  name  = "/${var.project_name}/slack_bot_token"
  type  = "SecureString"
  value = var.slack_bot_token != "" ? var.slack_bot_token : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "slack_signing_secret" {
  name  = "/${var.project_name}/slack_signing_secret"
  type  = "SecureString"
  value = var.slack_signing_secret != "" ? var.slack_signing_secret : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "slack_app_token" {
  name  = "/${var.project_name}/slack_app_token"
  type  = "SecureString"
  value = var.slack_app_token != "" ? var.slack_app_token : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "honeycomb_api_key" {
  name  = "/${var.project_name}/honeycomb_api_key"
  type  = "SecureString"
  value = var.honeycomb_api_key != "" ? var.honeycomb_api_key : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "redis_url" {
  name  = "/${var.project_name}/redis_url"
  type  = "SecureString"
  value = var.redis_url != "" ? var.redis_url : "redis://:${aws_ssm_parameter.redis_password.value}@redis.${var.project_name}.local:6379"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  depends_on = [aws_service_discovery_service.redis, aws_ssm_parameter.redis_password]

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "redis_password" {
  name  = "/${var.project_name}/redis_password"
  type  = "SecureString"
  value = var.redis_password != "" ? var.redis_password : "REPLACE_ME_WITH_SECURE_PASSWORD"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "target_user_id" {
  name  = "/${var.project_name}/target_user_id"
  type  = "String"
  value = var.target_user_id != "" ? var.target_user_id : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "scrape_channels" {
  name  = "/${var.project_name}/scrape_channels"
  type  = "String"
  value = var.scrape_channels != "" ? var.scrape_channels : "REPLACE_ME"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}