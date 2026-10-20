# Запуск через Docker: определяет IP/имя этого ПК для сертификата и поднимает контейнер.
$ip = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } |
       Select-Object -First 1).IPv4Address.IPAddress
$env:LAN_IPS = $ip
$env:HOST_NAME = $env:COMPUTERNAME
Set-Location $PSScriptRoot
Write-Host "Адрес в сети: https://${ip}:8443"
docker compose up --build -d
docker compose logs -f
