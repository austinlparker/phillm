terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.99"
    }
  }

  backend "s3" {
    bucket = "your-phillm-terraform-state-bucket"
    key    = "phillm/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC and Networking - SIMPLIFIED: Single AZ, public subnets only
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${var.project_name}-igw"
    Environment = var.environment
  }
}

# Public subnets - ALB requires at least 2 AZs
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 1)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.project_name}-public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${var.project_name}-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Security Groups
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-alb"
    Environment = var.environment
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks"
  description = "Security group for ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id] # Only ALB can reach app
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # App needs internet for APIs, image pulls
  }

  tags = {
    Name        = "${var.project_name}-ecs-tasks"
    Environment = var.environment
  }
}

resource "aws_security_group" "redis" {
  name        = "${var.project_name}-redis"
  description = "Security group for Redis ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # Redis needs egress for pulling images
  }

  tags = {
    Name        = "${var.project_name}-redis"
    Environment = var.environment
  }
}

# ECR Repository
resource "aws_ecr_repository" "phillm" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true # Allow deletion even with images present

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
  }
}

# EFS for Redis persistence - COST OPTIMIZED: Burst mode instead of provisioned
resource "aws_efs_file_system" "redis" {
  creation_token = "${var.project_name}-redis-data"
  encrypted      = true

  performance_mode = "generalPurpose"
  throughput_mode  = "bursting" # Changed from provisioned to save ~$15/month

  tags = {
    Name        = "${var.project_name}-redis-efs"
    Environment = var.environment
  }
}

resource "aws_efs_mount_target" "redis" {
  file_system_id  = aws_efs_file_system.redis.id
  subnet_id       = aws_subnet.public[0].id # Use first public subnet
  security_groups = [aws_security_group.efs.id]
}

resource "aws_security_group" "efs" {
  name        = "${var.project_name}-efs"
  description = "Security group for EFS"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.redis.id]
  }

  tags = {
    Name        = "${var.project_name}-efs"
    Environment = var.environment
  }
}

# Redis ECS Task Definition - runs in public subnet
resource "aws_ecs_task_definition" "redis" {
  family                   = "${var.project_name}-redis"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.redis_cpu
  memory                   = var.redis_memory

  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "redis"
      image = "public.ecr.aws/docker/library/redis:8.0.2-alpine"

      portMappings = [
        {
          containerPort = 6379
          hostPort      = 6379
        }
      ]

      command = [
        "redis-server",
        "--requirepass", var.redis_password,
        "--appendonly", "yes",
        "--appendfsync", "everysec",
        "--dir", "/data",
        "--save", "900 1",
        "--save", "300 10",
        "--save", "60 10000"
      ]

      mountPoints = [
        {
          sourceVolume  = "redis-data"
          containerPath = "/data"
          readOnly      = false
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.redis.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      essential = true

      healthCheck = {
        command = [
          "CMD-SHELL",
          "redis-cli --raw incr ping || exit 1"
        ]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  volume {
    name = "redis-data"

    efs_volume_configuration {
      file_system_id = aws_efs_file_system.redis.id
      root_directory = "/"
    }
  }

  tags = {
    Environment = var.environment
  }
}

# Redis ECS Service - runs in public subnet with direct internet access
resource "aws_ecs_service" "redis" {
  name            = "${var.project_name}-redis"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.redis.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.redis.id]
    subnets          = [aws_subnet.public[0].id] # Use first public subnet
    assign_public_ip = true                      # Direct internet access, no NAT Gateway needed
  }

  # Enable service discovery
  service_registries {
    registry_arn = aws_service_discovery_service.redis.arn
  }

  tags = {
    Environment = var.environment
  }
}

# Service Discovery for Redis
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${var.project_name}.local"
  description = "Private DNS namespace for ${var.project_name}"
  vpc         = aws_vpc.main.id

  tags = {
    Environment = var.environment
  }
}

resource "aws_service_discovery_service" "redis" {
  name = "redis"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 60
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  tags = {
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "phillm" {
  name        = "${var.project_name}-tg"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }

  tags = {
    Environment = var.environment
  }
}

resource "aws_lb_listener" "phillm" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.phillm.arn
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled" # Disabled to save costs - can enable if needed
  }

  tags = {
    Environment = var.environment
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "phillm" {
  family                   = var.project_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory

  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = var.project_name
      image = "${aws_ecr_repository.phillm.repository_url}:latest"

      portMappings = [
        {
          containerPort = 3000
          hostPort      = 3000
        }
      ]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
          value = "true"
        }
      ]

      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = aws_ssm_parameter.openai_api_key.arn
        },
        {
          name      = "SLACK_BOT_TOKEN"
          valueFrom = aws_ssm_parameter.slack_bot_token.arn
        },
        {
          name      = "SLACK_SIGNING_SECRET"
          valueFrom = aws_ssm_parameter.slack_signing_secret.arn
        },
        {
          name      = "SLACK_APP_TOKEN"
          valueFrom = aws_ssm_parameter.slack_app_token.arn
        },
        {
          name      = "HONEYCOMB_API_KEY"
          valueFrom = aws_ssm_parameter.honeycomb_api_key.arn
        },
        {
          name      = "REDIS_URL"
          valueFrom = aws_ssm_parameter.redis_url.arn
        },
        {
          name      = "TARGET_USER_ID"
          valueFrom = aws_ssm_parameter.target_user_id.arn
        },
        {
          name      = "SCRAPE_CHANNELS"
          valueFrom = aws_ssm_parameter.scrape_channels.arn
        },
        {
          name      = "MAX_RESPONSE_TOKENS"
          valueFrom = aws_ssm_parameter.max_response_tokens.arn
        },
        {
          name      = "STYLE_SIMILARITY_THRESHOLD"
          valueFrom = aws_ssm_parameter.style_similarity_threshold.arn
        },
        {
          name      = "CONVERSATION_DISTANCE_THRESHOLD"
          valueFrom = aws_ssm_parameter.conversation_distance_threshold.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.phillm.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      essential = true
    }
  ])

  tags = {
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "phillm" {
  name            = var.project_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.phillm.arn
  desired_count   = var.app_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.public[*].id # Can use multiple subnets for better availability
    assign_public_ip = true                    # Still need public IP for outbound internet access
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.phillm.arn
    container_name   = var.project_name
    container_port   = 3000
  }

  depends_on = [aws_lb_listener.phillm]

  tags = {
    Environment = var.environment
  }
}

# CloudWatch Log Groups - with shorter retention to save costs
resource "aws_cloudwatch_log_group" "phillm" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 7 # Reduced from 14+ days to save costs

  tags = {
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/ecs/${var.project_name}-redis"
  retention_in_days = 7 # Reduced from 14+ days to save costs

  tags = {
    Environment = var.environment
  }
}