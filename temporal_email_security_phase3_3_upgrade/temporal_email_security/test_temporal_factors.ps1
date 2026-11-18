# Temporal Email Security: controlled frequency and recency sensitivity tests.
# Run with the FastAPI server already listening on 127.0.0.1:8000.
$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:8000'
$endTime = [datetime]::Parse('2026-11-06T18:00:00Z').ToUniversalTime()
$sender = 'colleague@company.example'
$recipient = 'employee@company.example'
$results = New-Object System.Collections.Generic.List[object]

function Reset-User([string]$id) {
    Invoke-RestMethod -Method Post -Uri "$base/reset/$id" | Out-Null
}
function Send-Email([string]$id, [datetime]$when, [string]$subject, [string]$body) {
    $payload = @{
        user_id = $id
        sender = $sender
        recipient = $recipient
        subject = $subject
        body = $body
        timestamp = $when.ToString('yyyy-MM-ddTHH:mm:ssZ')
    } | ConvertTo-Json -Compress
    return Invoke-RestMethod -Method Post -Uri "$base/predict" -ContentType 'application/json' -Body $payload
}
function Final-Email([string]$id) {
    return Send-Email $id $endTime 'Updated project schedule' 'Hi, the project meeting is tomorrow at 10 AM. Please review the schedule before we meet.'
}

try { Invoke-RestMethod -Uri "$base/health" -Method Get | Out-Null }
catch { Write-Error 'API is not reachable. Start uvicorn in the other PowerShell window.'; exit 1 }

Write-Host "`nEXPERIMENT 1: Prior interaction count (last interaction fixed at 1 hour ago)" -ForegroundColor Cyan
foreach ($count in @(0, 5, 10, 20, 50)) {
    $id = "freq_$count"
    Reset-User $id
    if ($count -gt 0) {
        # All histories end at the same time; previous events spread over a fixed 20-day window.
        for ($i=0; $i -lt $count; $i++) {
            $hoursBefore = if ($count -eq 1) { 1.0 } else { 1.0 + (480.0 * ($count - 1 - $i) / ($count - 1)) }
            $when = $endTime.AddHours(-$hoursBefore)
            Send-Email $id $when 'Routine project update' 'Hi, here is the regular project status update. Thanks.' | Out-Null
        }
    }
    $response = Final-Email $id
    $results.Add([pscustomobject]@{ Experiment='Frequency'; Condition="$count previous emails"; RiskScore=[double]$response.risk_score; PredictedLabel=[int]$response.predicted_label })
}

Write-Host "`nEXPERIMENT 2: Time since last interaction (10 prior emails each)" -ForegroundColor Cyan
foreach ($days in @(0, 7, 30)) {
    $label = if ($days -eq 0) { '1 hour' } elseif ($days -eq 7) { '7 days' } else { '30 days' }
    $id = "recency_$days"
    Reset-User $id
    # Identical 10-message histories shifted backward in time, preserving event intervals.
    for ($i=0; $i -lt 10; $i++) {
        $hoursBefore = (24.0 * $days) + 1.0 + (9 - $i) * 24.0
        $when = $endTime.AddHours(-$hoursBefore)
        Send-Email $id $when 'Routine project update' 'Hi, here is the regular project status update. Thanks.' | Out-Null
    }
    $response = Final-Email $id
    $results.Add([pscustomobject]@{ Experiment='Recency'; Condition=$label; RiskScore=[double]$response.risk_score; PredictedLabel=[int]$response.predicted_label })
}

Write-Host "`nRESULTS (raw model score, not a calibrated probability)" -ForegroundColor Green
$results | Format-Table -AutoSize
$output = Join-Path (Get-Location) 'temporal_factor_results.csv'
$results | Export-Csv -Path $output -NoTypeInformation -Encoding UTF8
Write-Host "Saved: $output" -ForegroundColor Green
Write-Host 'Note: These are synthetic-model sensitivity tests, not evidence of real-world detection accuracy.'
