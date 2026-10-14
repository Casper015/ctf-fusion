param([string]$Receipt, [string]$StateDir)
$ErrorActionPreference='Stop'
$launch = Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
$wrapperPid = [int]$launch.wrapper.ProcessId
$wrapperBirth = $launch.wrapper.birth_utc
$wrapperBirthTicks = if ($wrapperBirth -is [datetime]) { $wrapperBirth.ToUniversalTime().Ticks } else { [datetime]::Parse([string]$wrapperBirth, [cultureinfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime().Ticks }
$outputPath = Join-Path $StateDir 'observer.jsonl'
$observerSelf = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
@{pid=$PID; birth_utc=$observerSelf.CreationDate.ToUniversalTime().ToString('o'); parent_pid=$observerSelf.ParentProcessId; command=$observerSelf.CommandLine} | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $StateDir 'OBSERVER_IDENTITY.json') -Encoding utf8
while ($true) {
    $snapshot=Get-CimInstance Win32_Process
    $wrapper=$snapshot | Where-Object { $_.ProcessId -eq $wrapperPid }
    $same=($null -ne $wrapper -and $wrapper.CreationDate.ToUniversalTime().Ticks -eq $wrapperBirthTicks -and $wrapper.CommandLine -eq $launch.wrapper.CommandLine)
    $treePids=[System.Collections.Generic.HashSet[int]]::new()
    if ($same) { $null=$treePids.Add($wrapperPid) }
    for ($depth=0; $depth -lt 8; $depth++) {
        foreach ($row in $snapshot) {
            if ($treePids.Contains([int]$row.ParentProcessId)) { $null=$treePids.Add([int]$row.ProcessId) }
        }
    }
    $rows=@()
    foreach ($row in $snapshot | Where-Object { $treePids.Contains([int]$_.ProcessId) }) {
        $privateBytes=$null
        $privateBirthVerified=$false
        try {
            $processObject=Get-Process -Id $row.ProcessId -ErrorAction Stop
            # CIM truncates process birth times to microseconds. Never read a
            # reused PID's private bytes as if they belonged to the old birth.
            $privateBirthVerified=[math]::Abs($processObject.StartTime.ToUniversalTime().Ticks-$row.CreationDate.ToUniversalTime().Ticks) -lt 10
            if ($privateBirthVerified) { $privateBytes=$processObject.PrivateMemorySize64 }
        } catch {}
        $rows+=@{pid=$row.ProcessId; parent_pid=$row.ParentProcessId; birth_utc=$row.CreationDate.ToUniversalTime().ToString('o'); command=$row.CommandLine; cpu_100ns=([uint64]$row.KernelModeTime+[uint64]$row.UserModeTime); private_bytes=$privateBytes; private_birth_verified=$privateBirthVerified; working_set_bytes=$row.WorkingSetSize}
    }
    $memory=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $os=Get-CimInstance Win32_OperatingSystem
    $lease=@{}
    foreach ($kind in @('cpu','gpu')) {
        $leasePath=Join-Path $env:USERPROFILE ".ctf-fusion/locks/$kind.json"
        $lease[$kind]=if(Test-Path -LiteralPath $leasePath){Get-Content -LiteralPath $leasePath -Raw | ConvertFrom-Json}else{$null}
    }
    @{at_utc=[datetime]::UtcNow.ToString('o'); verified_wrapper_alive=$same; processes=$rows; leases=$lease; physical_bytes=([uint64]$os.TotalVisibleMemorySize*1024); free_physical_bytes=([uint64]$os.FreePhysicalMemory*1024); committed_bytes=$memory.CommittedBytes; commit_limit_bytes=$memory.CommitLimit; available_bytes=$memory.AvailableBytes; pages_per_sec=$memory.PagesPersec} | ConvertTo-Json -Depth 7 -Compress | Add-Content -LiteralPath $outputPath -Encoding utf8
    if (-not $same) { break }
    Start-Sleep -Seconds 15
}
