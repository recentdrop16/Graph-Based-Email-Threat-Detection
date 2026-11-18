$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:8000'
$users = @('history_user_A', 'history_user_B')
foreach ($u in $users) {
    Invoke-RestMethod -Method Post -Uri "$base/reset/$u" | Out-Null
}
function Send-Email($user, $sender, $recipient, $subject, $body, $timestamp) {
    $payload = @{
        user_id = $user
        sender = $sender
        recipient = $recipient
        subject = $subject
        body = $body
        timestamp = $timestamp
    } | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$base/predict" -ContentType 'application/json' -Body $payload
}
$sender = 'partner@vendor.example'
$recipient = 'employee@company.example'
$historySubject = 'Weekly project update'
$historyBody = 'Here is the regular project status update. Everything is on schedule.'
Write-Host 'Building User A history (20 ordinary messages from the same sender)...'
for ($i = 1; $i -le 20; $i++) {
    $time = (Get-Date '2026-11-01T09:00:00').AddHours($i).ToString('yyyy-MM-ddTHH:mm:ss')
    Send-Email 'history_user_A' $sender $recipient $historySubject $historyBody $time | Out-Null
}
Write-Host 'Building User B history (20 ordinary messages from other senders)...'
for ($i = 1; $i -le 20; $i++) {
    $time = (Get-Date '2026-11-01T09:00:00').AddHours($i).ToString('yyyy-MM-ddTHH:mm:ss')
    $otherSender = "colleague$i@company.example"
    Send-Email 'history_user_B' $otherSender $recipient $historySubject $historyBody $time | Out-Null
}
$testSubject = 'Please review the attached invoice'
$testBody = 'Hello, please review the invoice and confirm whether the amount is correct. Thank you.'
$testTime = '2026-11-06T19:00:00'
$a = Send-Email 'history_user_A' $sender $recipient $testSubject $testBody $testTime
$b = Send-Email 'history_user_B' $sender $recipient $testSubject $testBody $testTime
Write-Host ''
Write-Host 'FINAL IDENTICAL EMAIL RESULTS'
[pscustomobject]@{User='A: Familiar sender';RiskScore=$a.risk_score;Label=$a.predicted_label},
[pscustomobject]@{User='B: New sender';RiskScore=$b.risk_score;Label=$b.predicted_label} | Format-Table -AutoSize
Write-Host ('Absolute score difference: {0:N6}' -f [math]::Abs([double]$a.risk_score - [double]$b.risk_score))
