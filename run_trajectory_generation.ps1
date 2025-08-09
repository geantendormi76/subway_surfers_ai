# --- Configuration ---
$ProjectRoot = $PSScriptRoot
$VideosDir = Join-Path $ProjectRoot "data/raw_videos"
$ActionsDir = Join-Path $ProjectRoot "data/actions"
$OutputDir = Join-Path $ProjectRoot "data/trajectories"

# [核心] 根据您的截图计算并填写的精确锚点帧号
$AnchorFrames = @(
    97,   # gameplay_01_actions.mp4
    313,  # gameplay_02_actions.mp4
    389,  # gameplay_03_actions.mp4
    410,  # gameplay_04_actions.mp4
    182,  # gameplay_05_actions.mp4
    204,  # gameplay_06_actions.mp4
    172,  # gameplay_07_actions.mp4
    161,  # gameplay_08_actions.mp4
    170   # gameplay_09_actions.mp4
)

# --- Execution ---
Write-Host "--- Starting Batch Trajectory Generation Task (Smart Scan Mode) ---" -ForegroundColor Green

# Find and sort all gameplay_*.mp4 files
$videoFiles = Get-ChildItem -Path $VideosDir -Filter "gameplay_*_actions.mp4" | Sort-Object Name

if ($videoFiles.Count -ne $AnchorFrames.Count) {
    Write-Host "Error: Found $($videoFiles.Count) video files, but you provided $($AnchorFrames.Count) anchor frames in the script. The counts must match exactly!" -ForegroundColor Red
    exit
}

for ($i = 0; $i -lt $videoFiles.Count; $i++) {
    $video = $videoFiles[$i]
    $base_name = $video.BaseName
    
    $video_path = $video.FullName
    $actions_path = Join-Path $ActionsDir "${base_name}.txt"
    $output_path = Join-Path $OutputDir "${base_name}.pkl.xz"
    
    $anchor_frame = $AnchorFrames[$i]

    if (Test-Path $actions_path) {
        Write-Host "`n[Processing $base_name]... Using anchor frame: $anchor_frame" -ForegroundColor Yellow
        python scripts/08_shengcheng_guiji_shuju.py --video $video_path --actions $actions_path --output $output_path --first-action-frame $anchor_frame
    } else {
        Write-Host "`n[Warning] Corresponding action file not found: $actions_path. Skipped." -ForegroundColor Red
    }
}

Write-Host "`n--- Batch Trajectory Generation Task Completed! ---" -ForegroundColor Green