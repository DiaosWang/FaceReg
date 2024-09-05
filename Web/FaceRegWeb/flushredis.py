import redis
import os 

# Redis config
redis_host = str(os.getenv("REDIS_HOST", 'localhost')) 
redis_port = int(os.getenv("REDIS_PORT", 2012))
redis_password = str(os.getenv("REDIS_PASSWORD", 'Xjsfzb@Redis123!')) 
num_workers = int(os.getenv("NUM_WORKERS", 10))

# connected
r = redis.Redis(host=redis_host, port=redis_port, password=redis_password, db=0)

# delete specific key in redisdb
keys_to_delete = ['write_lock', 'read_lock']  + [f"worker_{i}"for i in range(num_workers)]

print("Deleted key:", end = " ")
for key in keys_to_delete:
    r.delete(key)
    print(f"{key}", end ="  ")
print()

print("Specified keys deleted successfully.")