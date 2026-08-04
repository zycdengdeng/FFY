# 一键提交并推送到 GitHub(在 FFY 目录里跑)
# 用法: .\sync.ps1 "commit message"
param([string]$m = "update")
git add -A
git commit -m $m
git push
Write-Host "`n已推送。A100 上执行: git pull" -ForegroundColor Green
