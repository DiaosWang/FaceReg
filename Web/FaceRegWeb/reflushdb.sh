#!/bin/bash

# keep input valid
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <server_url>"
    exit 1
fi

# ip:host  
SERVER=$1
ENDPOINT="/refreshdb"

# 使用 curl 调用接口
response=$(curl -s -X POST "$SERVER$ENDPOINT" -H "Content-Type: application/json")

# 打印响应
echo "Response from $ENDPOINT: $response"

