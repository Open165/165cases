#!/bin/bash

# 檢查參數
if [ $# -lt 2 ]; then
  echo "用法: $0 [開始數字] [結束數字] [可選: 最大並發數]"
  echo "例如: $0 21 31 5"
  exit 1
fi

# 從命令行參數取得開始和結束數字
START_NUM=$1
END_NUM=$2

# 設定最大並發進程數 (默認為5，但可以通過第三個參數指定)
MAX_PROCESSES=${3:-5}

# 用來追蹤運行中的進程
declare -a pids

# 定義等待函數
wait_for_processes() {
  # 等待任何進程完成
  while [ ${#pids[@]} -ge $MAX_PROCESSES ]; do
    for i in "${!pids[@]}"; do
      if ! kill -0 ${pids[i]} 2>/dev/null; then
        unset pids[i]
      fi
    done
    # 重整陣列
    pids=("${pids[@]}")
    # 如果仍然達到最大進程數，暫停一下再檢查
    if [ ${#pids[@]} -ge $MAX_PROCESSES ]; then
      sleep 0.5
    fi
  done
}

echo "開始並發處理 cases/p${START_NUM}.json 到 cases/p${END_NUM}.json (最大並發數: $MAX_PROCESSES)..."

# 遍歷從START_NUM到END_NUM的數字
for i in $(seq -f "%03g" $START_NUM $END_NUM); do
  filename="cases/p${i}.json"
  
  # 檢查文件是否存在
  if [ -f "$filename" ]; then
    echo "處理文件: $filename"
    
    # 等待，如果已達到最大並發數
    wait_for_processes
    
    # 啟動處理並將進程ID加入追蹤陣列
    ./165embedding.sh "$filename" &
    pids+=($!)
    
    echo "啟動進程 $! 處理 $filename (目前運行 ${#pids[@]} 個進程)"
  else
    echo "警告: 文件 $filename 不存在，跳過"
  fi
done

# 等待所有剩餘進程完成
echo "等待所有進程完成..."
for pid in "${pids[@]}"; do
  wait $pid
done

echo "所有處理已完成！"
