#!/bin/bash
# ==============================================================================
# 脚本功能: 批量将处理好的CFR视频（恒定帧率视频）切成图片帧。
# 这是为后续手动标注和DTW对齐准备视觉素材的关键步骤。
# ==============================================================================

# --- 配置区 ---
# 获取脚本所在目录的上级目录，即项目根目录
# 这与 07_cfr_yuchuli.sh 中的路径逻辑保持一致
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# [核心] 输入目录：我们处理的是第一步生成的CFR视频
VIDEOS_DIR="$PROJECT_ROOT/data/internet_videos/processed"
# [核心] 输出目录：所有视频帧都将被提取到这里
FRAMES_DIR="$PROJECT_ROOT/data/frames"

# --- 执行区 ---

echo -e "\033[1;34m--- 开始执行视频批量切片任务 (WSL版) ---\033[0m"
echo "项目根目录: $PROJECT_ROOT"
echo "读取CFR视频于: $VIDEOS_DIR"
echo "输出图片帧到: $FRAMES_DIR"
echo ""

# 检查CFR视频目录是否存在
if [ ! -d "$VIDEOS_DIR" ]; then
    echo -e "\033[1;31m错误: CFR视频目录不存在: $VIDEOS_DIR\033[0m"
    echo -e "\033[1;33m请先成功运行 'scripts/07_cfr_yuchuli.sh'。\033[0m"
    exit 1
fi

# 检查目录下是否有 *_cfr.mp4 文件，防止脚本在空目录上运行
if ! ls "$VIDEOS_DIR"/*_cfr.mp4 1> /dev/null 2>&1; then
    echo -e "\033[1;31m错误: 在 '$VIDEOS_DIR' 中未找到任何 '_cfr.mp4' 文件。\033[0m"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$FRAMES_DIR"

# 遍历所有处理好的CFR视频文件
for video_path in "$VIDEOS_DIR"/*_cfr.mp4; do
    # 从完整路径中提取文件名 (例如: "video_01_cfr.mp4")
    video_filename=$(basename "$video_path")
    
    echo -e "\033[1;33m[处理中] 正在切片: $video_filename ...\033[0m"
    
    # 调用Python脚本，传入正确的参数
    # 注意：在Bash中，我们直接使用 $PROJECT_ROOT 来构建绝对路径
    python "$PROJECT_ROOT/scripts/01_video_qiege.py" --video "$video_path" --output_dir "$FRAMES_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m[成功] 已完成: $video_filename\033[0m"
    else
        echo -e "\033[1;31m[失败] 处理视频时发生错误: $video_filename\033[0m"
    fi
    echo ""
done

echo -e "\033[1;34m--- 所有视频切片任务已完成！ ---\033[0m"