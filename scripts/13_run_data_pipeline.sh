#!/bin/bash

# ==============================================================================
# 脚本名称: 13_run_data_pipeline.sh
# 脚本功能: 驱动从CFR预处理到最终轨迹生成的完整数据处理流水线。
#           这是一个批处理脚本，会自动处理指定目录下的所有原始视频。
# 使用方法: ./scripts/13_run_data_pipeline.sh (在项目根目录 subway_surfers_ai/ 下执行)
# ==============================================================================

set -e # 如果任何命令失败，则立即退出脚本

# --- 配置区 ---
# 获取脚本所在的目录，并向上导航一级以获得项目根目录
XIANGMU_GEN_MULU=$(dirname "$(dirname "$(readlink -f "$0")")")
PROCESSED_VIDEO_DIR="$XIANGMU_GEN_MULU/data/internet_videos/processed"

echo -e "\033[1;34m--- 启动完整的数据处理流水线 ---\033[0m"
echo "项目根目录: $XIANGMU_GEN_MULU"

# --- 阶段一: CFR 预处理 ---
echo -e "\n\033[1;36m===== 阶段一: 批量CFR预处理 =====\033[0m"
# 直接执行CFR脚本
"$XIANGMU_GEN_MULU/scripts/07_cfr_yuchuli.sh"
echo -e "\033[1;32mCFR预处理完成。\033[0m"


# --- 阶段二 & 三: 辅助标注与DTW对齐 (循环处理每个视频) ---
echo -e "\n\033[1;36m===== 阶段二 & 三: 模型辅助标注与DTW对齐 =====\033[0m"

if [ ! -d "$PROCESSED_VIDEO_DIR" ]; then
    echo -e "\033[1;31m错误: 经CFR处理的视频目录不存在: $PROCESSED_VIDEO_DIR\033[0m"
    exit 1
fi

for video_path in "$PROCESSED_VIDEO_DIR"/*_cfr.mp4; do
    if [ -f "$video_path" ]; then
        video_filename=$(basename "$video_path")
        echo -e "\n\033[1;35m--- 开始处理视频: $video_filename ---\033[0m"

        # 核心修正：使用 python <file_path> 的方式执行
        echo -e "\033[1;33m[子步骤 1/2] 正在执行模型辅助标注...\033[0m"
        python "$XIANGMU_GEN_MULU/scripts/10_tool_model_assisted_annotation.py" --video_path "$video_path"
        if [ $? -ne 0 ]; then
            echo -e "\033[1;31m错误: 模型辅助标注失败: $video_filename\033[0m"
            continue
        fi

        echo -e "\033[1;33m[子步骤 2/2] 正在执行DTW精准对齐...\033[0m"
        python "$XIANGMU_GEN_MULU/scripts/11_align_with_dtw.py" --video_path "$video_path"
        if [ $? -ne 0 ]; then
            echo -e "\033[1;31m错误: DTW对齐失败: $video_filename\033[0m"
            continue
        fi
        
        echo -e "\033[1;32m--- 完成视频处理: $video_filename ---\033[0m"
    fi
done
echo -e "\033[1;32m所有视频的标注与对齐已完成。\033[0m"


# --- 阶段四: 生成最终轨迹文件 ---
echo -e "\n\033[1;36m===== 阶段四: 生成最终轨迹文件 =====\033[0m"
python "$XIANGMU_GEN_MULU/scripts/12_generate_final_trajectories.py"
if [ $? -ne 0 ]; then
    echo -e "\033[1;31m错误: 最终轨迹文件生成失败。\033[0m"
    exit 1
fi
echo -e "\033[1;32m所有轨迹文件生成完毕。\033[0m"


echo -e "\n\033[1;34m--- 完整数据处理流水线执行成功！ ---\033[0m"