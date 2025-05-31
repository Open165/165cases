#!/bin/bash

# 檢查是否提供參數
if [ $# -ne 2 ]; then
    echo "使用方法: $0 <開始頁數> <循環次數>"
    exit 1
fi

# 建立 cases 目錄（如果不存在）
mkdir -p cases

# 循環執行
for ((i=$1; i<=$1+$2; i++)); do
    # 使用 printf 確保頁碼有前導零
    filename=$(printf "cases/p%03d.json" $i)

    # 執行 curl 命令並儲存結果
    curl 'https://165dashboard.tw/CIB_DWS_API/api/CaseSummary/GetCaseSummaryList' \
        -H 'accept: application/json, text/plain, */*' \
        -H 'accept-language: zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6' \
        -H 'cache-control: no-cache' \
        -H 'content-type: application/json' \
        -H 'origin: https://165dashboard.tw' \
        -H 'pragma: no-cache' \
        -H 'priority: u=1, i' \
        -H 'referer: https://165dashboard.tw/city-case-summary' \
        -H 'sec-ch-ua: "Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"' \
        -H 'sec-ch-ua-mobile: ?0' \
        -H 'sec-ch-ua-platform: "Linux"' \
        -H 'sec-fetch-dest: empty' \
        -H 'sec-fetch-mode: cors' \
        -H 'sec-fetch-site: same-origin' \
        -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' \
        --data-raw "{\"UsingPaging\":true,\"NumberOfPerPage\":1000,\"PageIndex\":$i,\"SortOrderInfos\":[{\"SortField\":\"CaseDate\",\"SortOrder\":\"ASC\"}],\"SearchTermInfos\":[],\"Keyword\":null,\"CityId\":null,\"CaseDate\":null}" \
        -o "$filename"

    echo "已下載第 $i 頁，儲存為 $filename"

    # 如果不是最後一次循環，則休息3秒
    if [ $i -lt $1 ]; then
        sleep 3
    fi
done

echo "完成所有下載"
