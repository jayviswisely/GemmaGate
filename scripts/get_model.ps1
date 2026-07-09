# Downloads Qwen2.5-7B-Instruct Q4_K_M (~4.7GB) into models/model.gguf
New-Item -ItemType Directory -Force -Path models | Out-Null
$url = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
Invoke-WebRequest -Uri $url -OutFile "models/model.gguf"
Write-Host "Done -> models/model.gguf ($( (Get-Item models/model.gguf).Length / 1GB ) GB)"
