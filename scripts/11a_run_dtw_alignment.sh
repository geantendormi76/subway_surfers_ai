#!/bin/bash
# ==============================================================================
# 脚本功能: 批量对所有CFR视频运行DTW时序对齐脚本。
#           它会自动查找 processed 目录下的视频和 annotated_actions 目录下
#           对应的json草稿文件，然后生成对齐后的轨迹。
# ==============================================================================

# --- 配置区 ---
# 获取脚本所在目录的上级目录，即项目根目录
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# 输入目录：处理好的CFR视频
VIDEOS_DIR="$PROJECT_ROOT/data/internet_videos/processed"

# --- 执行区 ---

echo -e "\033[1;34m--- 开始执行DTW批量对齐任务 ---\033[0m"
echo "项目根目录: $PROJECT_ROOT"
echo "读取CFR视频于: $VIDEOS_DIR"
echo ""

# 检查CFR视频目录是否存在
if [ ! -d "$VIDEOS_DIR" ]; then
    echo -e "\033[1;31m错误: CFR视频目录不存在: $VIDEOS_DIR\033[0m"
    echo -e "\033[1;33m请先成功运行 'scripts/07_cfr_yuchuli.sh'。\033[0m"
    exit 1
fi

# 检查目录下是否有 *_cfr.mp4 文件
if ! ls "$VIDEOS_DIR"/*_cfr.mp4 1> /dev/null 2>&1; then
    echo -e "\033[1;31m错误: 在 '$VIDEOS_DIR' 中未找到任何 '_cfr.mp4' 文件。\033[0m"
    exit 1
fi

# 遍历所有处理好的CFR视频文件
for video_path in "$VIDEOS_DIR"/*_cfr.mp4; do
    video_filename=$(basename "$video_path")
    
    echo -e "\033[1;33m[处理中] 正在为 $video_filename 运行DTW对齐...\033[0m"
    
    # [核心] 调用Python脚本，传入视频路径作为参数
    python "$PROJECT_ROOT/scripts/11_align_with_dtw.py" --video_path "$video_path"
    
    # 检查上一条命令是否成功执行
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m[成功] 已完成: $video_filename\033[0m"
    else
        echo -e "\033[1;31m[失败] 处理视频时发生错误: $video_filename\033[0m"
    fi
    echo ""
done

echo -e "\033[1;34m--- 所有DTW对齐任务已完成！ ---\033[0m"
echo -e "\033[1;32m下一步，请运行 'python scripts/12_generate_final_trajectories.py' 来生成最终的轨迹文件。\033[0m"