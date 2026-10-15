param([string]$Repo, [string]$StateDir, [string]$Python, [string]$Driver)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
Set-Location -LiteralPath $Repo
$self=Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$argsList=@('-u','tools/repo/task.py','lock','cpu','--task','T0037','--',$Python,'-u',$Driver)
$wrapper=Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $StateDir 'stdout.log') -RedirectStandardError (Join-Path $StateDir 'stderr.log')
$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$($wrapper.Id)"
$receipt=@{helper=@{ProcessId=$self.ProcessId;ParentProcessId=$self.ParentProcessId;birth_utc=$self.CreationDate.ToUniversalTime().ToString('o');CommandLine=$self.CommandLine};wrapper=@{ProcessId=$proc.ProcessId;ParentProcessId=$proc.ParentProcessId;birth_utc=$proc.CreationDate.ToUniversalTime().ToString('o');CommandLine=$proc.CommandLine};driver_sha256=(Get-FileHash -LiteralPath (Join-Path $Repo $Driver) -Algorithm SHA256).Hash.ToLower();no_auto_resume=$true}
$receiptPath=Join-Path $StateDir 'LAUNCH_RECEIPT.json'
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$observerScript=Join-Path $Repo 'bots/p_only/report/decision_collection/host_observer.ps1'
$observer=Start-Process -FilePath 'pwsh' -ArgumentList @('-NoProfile','-File',"`"$observerScript`"",'-Receipt',"`"$receiptPath`"",'-StateDir',"`"$StateDir`"") -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $StateDir 'observer_stdout.log') -RedirectStandardError (Join-Path $StateDir 'observer_stderr.log')
$wrapper.WaitForExit()
@{exit_code=$wrapper.ExitCode;completed_utc=[datetime]::UtcNow.ToString('o');driver_sha256=$receipt.driver_sha256} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StateDir 'COMPLETE.json') -Encoding utf8
$observer.WaitForExit()
