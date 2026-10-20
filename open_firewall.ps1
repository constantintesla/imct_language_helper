# Запусти один раз от имени администратора: открывает порт 8443 для локальной сети.
New-NetFirewallRule -DisplayName "IMCT Translator 8443" -Direction Inbound -Protocol TCP -LocalPort 8443 -Action Allow -Profile Private,Domain
