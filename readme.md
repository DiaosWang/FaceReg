本项目底层基于开源C++项目[seetaface6](https://github.com/seetafaceengine/SeetaFace6)，项目上层分别采用Python和Java进行调用实现Web端服务和移动端服务。项目实现了活体检测、人脸识别等必需功能，并有着超过99%的识别准确率。项目支持并发请求并使用Redis解决资源冲突，Postman并发量20的测试中99%请求响应时间小于4s。  

./Web  为web端服务  
./Android 中为安卓端服务  
点击进入查看详细内容  

模型和项目测试报告涉及到隐私问题，请在`issue`上联系我来获取，然后将不同模型分别放在：  
`./Android/app/src/main/assets/model`  and `./Web/FaceRegWeb/checkpoints`


reference:
[seetaFace6Python](https://github.com/tensorflower/seetaFace6Python)  
[seetaface6](https://github.com/seetafaceengine/SeetaFace6)
[MNN](https://github.com/alibaba/MNN)
