# TerraformとAWSプロバイダの設定
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# 1. Dockerイメージを保存するためのECRリポジトリを作成
resource "aws_ecr_repository" "app_ecr_repo" {
  name = "fastapi-app-repo" # リポジトリ名
}

# 2. EC2インスタンスがECRからイメージをプルするためのIAMロールを作成
resource "aws_iam_role" "ec2_role" {
  name = "ec2-ecr-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
      },
    ],
  })
}

# IAMロールにECR読み取り権限ポリシーをアタッチ
resource "aws_iam_role_policy_attachment" "ec2_ecr_policy_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# IAMインスタンスプロファイルを作成（EC2にIAMロールを割り当てるために必要）
resource "aws_iam_instance_profile" "ec2_instance_profile" {
  name = "ec2-instance-profile"
  role = aws_iam_role.ec2_role.name
}

# 3. EC2インスタンスのセキュリティグループ（ファイアウォール）を作成
resource "aws_security_group" "app_sg" {
  name        = "app-security-group"
  description = "Allow HTTP and SSH inbound traffic"

  # HTTP (80番ポート) のインバウンド通信を全てのIPから許可
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH (22番ポート) のインバウンド通信を全てのIPから許可 (デバッグ用に便利)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # 全てのアウトバウンド通信を許可
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 4. EC2インスタンスを作成
resource "aws_instance" "app_server" {
  # 無料枠のt2.microインスタンスを使用
  instance_type = "t3.micro"
  # Amazon Linux 2の最新AMIを自動で検索して使用
  ami           = data.aws_ami.amazon_linux_2.id
  # セキュリティグループを適用
  vpc_security_group_ids = [aws_security_group.app_sg.id]
  # IAMインスタンスプロファイルをアタッチ
  iam_instance_profile = aws_iam_instance_profile.ec2_instance_profile.name

  # 起動時に実行するスクリプトをuser_dataとして指定
  # templatefile関数で、user_data.shにECRリポジトリのURLを渡す
  user_data = templatefile("${path.module}/user_data.sh", {
    ecr_repo_url = aws_ecr_repository.app_ecr_repo.repository_url
  })

  tags = {
    Name = "FastAPI-App-Server"
  }
}

# 最新のAmazon Linux 2のAMI IDを検索するためのデータソース
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}