#!/bin/bash

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
  --data-raw '{"UsingPaging":true,"NumberOfPerPage":1000,"PageIndex":1,"SortOrderInfos":[{"SortField":"CaseDate","SortOrder":"DESC"}],"SearchTermInfos":[],"Keyword":null,"CityId":null,"CaseDate":null}'
