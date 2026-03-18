#!/bin/bash
# Virtual Skin Platform — 一键部署到 GitHub + Streamlit Cloud
set -e
cd "$(dirname "$0")"

REPO="marvelous"
USER="siyuanma123-ux"
GH="$(command -v gh 2>/dev/null || echo /tmp/gh_2.40.1_macOS_arm64/bin/gh)"
if [[ ! -x "$GH" ]]; then
  echo "正在下载 gh CLI..."
  (cd /tmp && curl -sL "https://github.com/cli/cli/releases/download/v2.40.1/gh_2.40.1_macOS_arm64.zip" -o gh.zip && unzip -o -q gh.zip)
  GH="/tmp/gh_2.40.1_macOS_arm64/bin/gh"
fi

echo "=== 1. 检查 GitHub 登录 ==="
if ! $GH auth status &>/dev/null; then
  echo "请先在浏览器中完成 GitHub 登录..."
  $GH auth login --web --hostname github.com
fi

echo ""
echo "=== 2. 创建远程仓库并推送 ==="
git remote remove origin 2>/dev/null || true
$GH repo create "$REPO" --public --source=. --remote=origin --push --description "Virtual Skin Platform: multi-scale transdermal drug delivery model"

echo ""
echo "=== 3. 部署完成 ==="
echo "仓库: https://github.com/$USER/$REPO"
echo ""
echo "下一步：在 Streamlit Cloud 部署"
echo "1. 打开 https://share.streamlit.io"
echo "2. 用 GitHub 登录 → New app"
echo "3. 选择 siyuanma123-ux/marvelous，Main file: app.py"
echo "4. Deploy 后获得公开链接，可分享给任何人"
