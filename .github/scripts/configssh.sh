#!/bin/bash

set -e

# ============================================================
# 配置
# ============================================================

PASSWORD="123456"
REMOTE_USER="root"

REMOTE_HOST="$1"
SSH_PORT="${2:-2222}"

SSH_DIR="$HOME/.ssh"
PRIVATE_KEY="$SSH_DIR/id_ed25519"
PUBLIC_KEY="$SSH_DIR/id_ed25519.pub"
SSH_CONFIG="$SSH_DIR/config"

# ============================================================
# 参数检查
# ============================================================

if [ -z "$REMOTE_HOST" ]; then
    echo "用法: $0 <服务器IP/主机名> [SSH端口]"
    echo "例如: $0 nmz9 2222"
    exit 1
fi

echo "目标服务器: $REMOTE_USER@$REMOTE_HOST:$SSH_PORT"

# ============================================================
# 1. 创建 SSH 目录
# ============================================================

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# ============================================================
# 2. 生成 SSH 密钥
# ============================================================

if [ ! -f "$PRIVATE_KEY" ]; then
    echo "生成 SSH 密钥..."

    ssh-keygen \
        -t ed25519 \
        -f "$PRIVATE_KEY" \
        -N ""
else
    echo "SSH 密钥已存在，跳过生成"
fi

# 修复 SSH 密钥权限
chmod 600 "$PRIVATE_KEY"
chmod 644 "$PUBLIC_KEY"

# ============================================================
# 3. 配置 ~/.ssh/config
# ============================================================

echo "配置 SSH config..."

# 如果已经存在相同 Host，先删除旧配置
if [ -f "$SSH_CONFIG" ]; then
    sed -i "/^Host $REMOTE_HOST$/,/^$/d" "$SSH_CONFIG"
fi

cat >> "$SSH_CONFIG" <<EOF

Host $REMOTE_HOST
    HostName $REMOTE_HOST
    Port $SSH_PORT
    User $REMOTE_USER
    IdentityFile $PRIVATE_KEY
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

EOF

chmod 600 "$SSH_CONFIG"

# ============================================================
# 4. 检查远程服务器
# ============================================================

echo "检查远程服务器..."

sshpass -p "$PASSWORD" ssh \
    -p "$SSH_PORT" \
    -o PubkeyAuthentication=no \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "$REMOTE_USER@$REMOTE_HOST" \
    "mkdir -p ~/.ssh && chmod 700 ~/.ssh"

# ============================================================
# 5. 复制公钥
# ============================================================

echo "复制 SSH 公钥..."

sshpass -p "$PASSWORD" ssh-copy-id \
    -i "$PUBLIC_KEY" \
    -p "$SSH_PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "$REMOTE_USER@$REMOTE_HOST"

# ============================================================
# 6. 修复服务器端 SSH 权限
# ============================================================

echo "修复远程 SSH 权限..."

sshpass -p "$PASSWORD" ssh \
    -p "$SSH_PORT" \
    -o PubkeyAuthentication=no \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "$REMOTE_USER@$REMOTE_HOST" \
    "chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

# ============================================================
# 7. 测试免密登录
# ============================================================

echo "测试 SSH 免密登录..."

if ssh \
    -o PreferredAuthentications=publickey \
    -o PasswordAuthentication=no \
    "$REMOTE_HOST" \
    "echo 'SSH免密登录配置成功'"
then

    echo "========================================"
    echo "免密登录配置成功"
    echo "========================================"
    echo "SSH 配置:"
    echo "  Host: $REMOTE_HOST"
    echo "  HostName: $REMOTE_HOST"
    echo "  Port: $SSH_PORT"
    echo "  User: $REMOTE_USER"
    echo
    echo "以后直接执行:"
    echo
    echo "  ssh $REMOTE_HOST"
    echo

else

    echo "========================================"
    echo "免密登录配置失败"
    echo "========================================"
    exit 1
fi