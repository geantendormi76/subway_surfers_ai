#!/bin/bash

# ==============================================================================
# 脚本名称: gongju_piliang_cfr_yuchuli.sh
# 脚本功能: 批量将指定目录下的所有 .mp4 视频转换为 30 FPS 的恒定帧率 (CFR) 视频。
# 这是数据处理工业化流水线的关键第一步，旨在消除可变帧率(VFR)带来的时序不确定性。
# ==============================================================================


# 获取脚本所在的目录，并将其设置为项目根目录
XIANGMU_GEN_MULU=$(dirname "$(dirname "$(readlink -f "$0")")")

# 源视频目录 (包含您从网络下载的原始视频)
YUAN_SHIPIN_MULU="$XIANGMU_GEN_MULU/data/internet_videos/raw"

# 目标目录 (用于存放处理后的CFR视频)
MUBIAO_MULU="$XIANGMU_GEN_MULU/data/internet_videos/processed"

# 目标帧率
MUBIAO_ZHENGLV=30

# --- 执行区 ---

echo -e "\033[1;34m--- 开始执行视频批量CFR预处理任务 ---\033[0m"
echo "项目根目录: $XIANGMU_GEN_MULU"
echo "读取源视频于: $YUAN_SHIPIN_MULU"
echo "输出处理后视频到: $MUBIAO_MULU"
echo "目标恒定帧率: $MUBIAO_ZHENGLV FPS"
echo ""

# 检查源目录是否存在
if [ ! -d "$YUAN_SHIPIN_MULU" ]; then
    echo -e "\033[1;31m错误: 源视频目录不存在: $YUAN_SHIPIN_MULU\033[0m"
    exit 1
fi

# 检查源目录下是否有视频文件
if [ -z "$(find "$YUAN_SHIPIN_MULU" -maxdepth 1 -type f -name '*.mp4')" ]; then
    echo -e "\033[1;33m警告: 在源视频目录 '$YUAN_SHIPIN_MULU' 中未找到任何 .mp4 文件。\033[0m"
    exit 0
fi

# 创建目标目录 (如果它不存在的话)
# -p 参数可以确保如果父目录不存在，也会一并创建
mkdir -p "$MUBIAO_MULU"

# 遍历源目录下的所有 .mp4 文件
for shipin_lujing in "$YUAN_SHIPIN_MULU"/*.mp4; do
    # 从完整路径中提取文件名 (例如: "video_01.mp4")
    shipin_wenjianming=$(basename "$shipin_lujing")
    
    # 构建输出文件的完整路径
    # 我们在文件名后加上 "_cfr" 来区分处理前后的文件
    shuchu_lujing="$MUBIAO_MULU/${shipin_wenjianming%.mp4}_cfr.mp4"
    
    echo -e "\033[1;33m[处理中] \`$shipin_wenjianming\` -> \`${shuchu_lujing##*/}\`\033[0m"
    
    # 执行 ffmpeg 命令
    # -hide_banner: 隐藏ffmpeg的版本和编译信息，使输出更干净
    # -loglevel error: 只在发生错误时才打印日志，成功的转换将保持沉默
    # -y: 如果输出文件已存在，则自动覆盖
    ffmpeg -hide_banner -loglevel error -y -i "$shipin_lujing" -vf "fps=$MUBIAO_ZHENGLV" "$shuchu_lujing"
    
    # 检查上一条命令是否成功执行
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m[成功] 已完成: ${shuchu_lujing##*/}\033[0m"
    else
        echo -e "\033[1;31m[失败] 处理视频时发生错误: $shipin_wenjianming\033[0m"
    fi
    echo ""
done

echo -e "\033[1;34m--- 所有视频处理任务已完成！ ---\033[0m"