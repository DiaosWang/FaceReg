# introduce  
这是人脸识别的web端代码，其依赖于onnx,fastapi,uvicorn,redis. 项目能在windows,linux 和macos(未测试) 上运行. 支持并发请求.   
项目细节详见docx文档.  

# requirements   
```  
fastapi                      0.103.2  
uvicorn[standard]            0.22.0     
opencv-python                4.9.0.80      
requests                     2.31.0      
Pillow                       9.4.0     
redis                        5.0.6   
onnxruntime                  1.18.1  
```  
推荐使用 `pip install` 安装上述库，可以不遵循其版本号（这里写出具体版本是为了复现原运行环境）.  

# file struct  
```
./             
    checkpoints     储存模型的目录  
    models          方法函数目录  
    webmain.py      入口函数  
    process.py      入口函数和方法函数之间的桥接函数  
    flushredis.py   处理redis数据库的函数(进而能使用redis)  
    redisLock.py    基于redis的读写锁  
    redis.yaml      redis的配置文件  
    config.yaml     项目的配置文件  
    setup.sh        启动项目的脚本(linux版本)
    reflushdb.sh    手动刷新数据库(usage: ./reflushdb.sh ip:port)    
    readme.md       本文件  
```
# setup 
`bash setup.sh [ip] [port] [进程数] [并发工作线程数]`   
e.g.: `bash setup.sh 0.0.0.0 8017 1 60`  

