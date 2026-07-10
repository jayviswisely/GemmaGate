# Downloads Qwen2.5-3B-Instruct Q4_K_M into models/model.gguf.
New-Item -ItemType Directory -Force -Path models | Out-Null
$url = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
Invoke-WebRequest -Uri $url -OutFile "models/model.gguf"
$gb = [Math]::Round((Get-Item models/model.gguf).Length / 1GB, 2)
Write-Host "Done -> models/model.gguf ($gb GB)"
