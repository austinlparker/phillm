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
  value = var.redis_url != "" ? var.redis_url : "redis://localhost:6379"

  tags = {
    Environment = var.environment
    Application = var.project_name
  }

  lifecycle {
    ignore_changes = [value]
  }
}