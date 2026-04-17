# パッケージをアップデート
yum update -y

# Dockerをインストールし、サービスを開始・有効化
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user
chkconfig docker on

# AWS CLI v2をインストール（ECRログインに必要）
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# ECRにログイン
# Terraformから渡されたリージョンとリポジトリURLを使用
AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${ecr_repo_url}

# 最新のDockerイメージをプルしてコンテナを実行
# :latestタグのイメージをプル
docker pull ${ecr_repo_url}:latest

# コンテナを実行。ホストの80番ポートをコンテナの80番ポートにマッピング
docker run -d -p 80:80 ${ecr_repo_url}:latest