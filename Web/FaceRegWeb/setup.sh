# 从 redis.yaml 文件中读取 Redis 相关配置信息
REDIS_CONFIG=$(cat redis.yaml)

# 工作进程数 workers 
NUM_WORKERS=${3}

REDIS_HOST=$(echo $REDIS_CONFIG | grep -oP 'REDIS_HOST=\K[^ ]+')
REDIS_PORT=$(echo $REDIS_CONFIG | grep -oP 'REDIS_PORT=\K[^ ]+')
REDIS_PASSWORD=$(echo $REDIS_CONFIG | grep -oP 'REDIS_PASSWORD=\K[^ ]+')

echo "num_workers: $NUM_WORKERS"
echo "redis_host: $REDIS_HOST; redis_port: $REDIS_PORT"

REDIS_HOST=$REDIS_HOST REDIS_PORT=$REDIS_PORT REDIS_PASSWORD=$REDIS_PASSWORD NUM_WORKERS=$NUM_WORKERS python flushredis.py  

REDIS_HOST=$REDIS_HOST REDIS_PORT=$REDIS_PORT REDIS_PASSWORD=$REDIS_PASSWORD NUM_WORKERS=$NUM_WORKERS MAX_READERS=${4}  uvicorn webmain:app  --host ${1} --port ${2}  --workers $NUM_WORKERS 