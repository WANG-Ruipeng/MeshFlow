$ErrorActionPreference='Stop'
$out='E:\MeshFlow\maintenance\storage_cleanup_20261022'
$roots=@('E:\MeshFlow','E:\MeshFlow-Control-artifacts')
$groups=@{}
$large=[System.Collections.Generic.List[object]]::new()
$cache=[System.Collections.Generic.List[object]]::new()
$archives=[System.Collections.Generic.List[object]]::new()
$skipped=[System.Collections.Generic.List[string]]::new()
$totalCount=0
foreach($root in $roots){
  $stack=[System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
  $stack.Push([System.IO.DirectoryInfo]::new($root))
  while($stack.Count -gt 0){
    $dir=$stack.Pop()
    foreach($child in $dir.EnumerateDirectories()){
      if(($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or $child.Name -eq '.git' -or $child.Name -eq 'venv'){$skipped.Add($child.FullName);continue}
      $stack.Push($child)
    }
    foreach($file in $dir.EnumerateFiles()){
      $totalCount++
      $rel=$file.FullName.Substring($root.Length+1)
      $parts=$rel.Split('\')
      $key=$root+'\'+$parts[0]
      if($parts.Count -gt 2){$key+='\'+$parts[1]}
      if(-not $groups.ContainsKey($key)){$groups[$key]=[pscustomobject]@{Path=$key;Bytes=[long]0;Files=0}}
      $groups[$key].Bytes+=$file.Length;$groups[$key].Files++
      $row=[pscustomobject]@{Path=$file.FullName;Bytes=$file.Length;LastWriteUTC=$file.LastWriteTimeUtc.ToString('o');Attributes=[string]$file.Attributes}
      if($file.Length -ge 64MB){$large.Add($row)}
      if($rel -match '(^|\\)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(\\|$)' -or $file.Extension -in @('.tmp','.part','.partial','.bak')){$cache.Add($row)}
      if($file.Extension -in @('.zip','.whl','.tar','.gz','.7z')){$archives.Add($row)}
    }
  }
}
$groups.Values | Sort-Object Bytes -Descending | Export-Csv -LiteralPath ($out+'\directory_sizes.csv') -NoTypeInformation -Encoding UTF8
$large | Sort-Object Bytes -Descending | Export-Csv -LiteralPath ($out+'\large_files.csv') -NoTypeInformation -Encoding UTF8
$cache | Export-Csv -LiteralPath ($out+'\cache_candidates.csv') -NoTypeInformation -Encoding UTF8
$archives | Sort-Object Bytes -Descending | Export-Csv -LiteralPath ($out+'\archives.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{Files=$totalCount;Skipped=@($skipped);Roots=$roots;UTC=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ($out+'\inventory_scope.json') -Encoding UTF8
$groups.Values | Sort-Object Bytes -Descending | Select-Object -First 22 Path,@{N='GiB';E={[math]::Round($_.Bytes/1GB,3)}},Files | Format-Table -AutoSize
$large | Sort-Object Bytes -Descending | Select-Object -First 18 Path,@{N='GiB';E={[math]::Round($_.Bytes/1GB,3)}} | Format-Table -AutoSize
