$archive_to_unzip = 'file per prova analisi dati PAT-*.zip'
$folder_archive = 'file per prova analisi dati PAT\'
$folder_destination = 'misure_2024_25\'

$path_downloads = 'C:\Users\fmelchiori\Downloads\'
$path_pat_studiodt_mosdi = 'C:\franzmelchiori\projects\studiodt\pat_studiodt_monitoraggio_scuole_infanzia\'
$path_archive = $path_pat_studiodt_mosdi + $folder_archive
$path_destination = $path_pat_studiodt_mosdi + $folder_destination

Set-Location -Path $path_downloads -PassThru
Expand-Archive -Path $archive_to_unzip -DestinationPath $path_pat_studiodt_mosdi
Remove-Item -Path $archive_to_unzip
Set-Location -Path $path_archive -PassThru
Get-ChildItem * | Rename-Item -NewName { $_.Name -Replace 'Copia di ','' }
Get-ChildItem * | Rename-Item -NewName { $_.Name -Replace '[\s]+','_' }
Get-ChildItem * | Rename-Item -NewName { $_.Name -Replace '[.]+','_' }
Get-ChildItem * | Rename-Item -NewName { $_.Name -Replace '[_]+','_' }
Get-ChildItem * | Rename-Item -NewName { $_.Name -Replace '_xlsx','.xlsx' }
Get-ChildItem * | Rename-Item -NewName { $_.Name.TrimEnd('_') }
Get-ChildItem * | Rename-Item -NewName { $_.Name.TrimStart('_') }
Set-Location -Path $path_pat_studiodt_mosdi -PassThru
Move-Item -Path ($path_archive + '\*.xlsx') -Destination $path_destination -Force
Remove-Item -Path $path_archive
