import time

class RedisReadWriteLock:
    def __init__(self, redis_client, lock_timeout=3600, max_readers=20):
        self.redis_client = redis_client
        self.read_lock_key = "read_lock"
        self.write_lock_key = "write_lock"
        self.lock_timeout = lock_timeout
        self.max_readers = max_readers

    def acquire_read(self):
        while True:
            if not self.redis_client.get(self.write_lock_key) and \
               int(self.redis_client.get(self.read_lock_key) or 0) < self.max_readers:
                self.redis_client.incr(self.read_lock_key)
                return
            time.sleep(0.01)

    def release_read(self):
        self.redis_client.decr(self.read_lock_key)

    def acquire_write(self):
        while not self.redis_client.setnx(self.write_lock_key, 1):
            time.sleep(0.01)
        self.redis_client.expire(self.write_lock_key, self.lock_timeout)
        
        while int(self.redis_client.get(self.read_lock_key) or 0) > 0:
            time.sleep(0.01)

    def release_write(self):
        self.redis_client.delete(self.write_lock_key)
