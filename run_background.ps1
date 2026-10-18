# PowerShell script to run OCR training in the background on Windows
$logFile = "train.log"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Starting OCR Training in Background (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "[*] Log file: $logFile"

$process = Start-Process -FilePath "python" -ArgumentList "-u", "train.py" -RedirectStandardOutput $logFile -RedirectStandardError "train_err.log" -PassThru -NoNewWindow

Write-Host "[+] Training started in background!" -ForegroundColor Green
Write-Host "    - Process ID (PID) : $($process.Id)"
Write-Host "    - Log Output File  : $logFile"
Write-Host ""
Write-Host "Commands to monitor and control:" -ForegroundColor Yellow
Write-Host "  1. View live logs in real-time:"
Write-Host "     Get-Content $logFile -Wait"
Write-Host ""
Write-Host "  2. Stop training:"
Write-Host "     Stop-Process -Id $($process.Id)"
Write-Host "==================================================================" -ForegroundColor Cyan
