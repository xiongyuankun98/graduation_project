旨在完成毕业设计，学习了解CNN，并用于人脸验证和人脸识别方面，人脸识别部分由于设备原因，实际情况为多分类任务。

人脸验证，基于SiameseNet：
CISIA-FaceV5
FERET(200人)

人脸识别，基于AlexNet：
JAFEE(本应该是表情识别的，但相对人脸样本丰富，作为基本实验)
CISIA-Webface(50人，人均137张)：质量感觉有时候有点迷，验证准确率一开始为30%，最近一次为68%

需要将code/configure.py有关地址变更，如数据集目录、模型地址等
未上传数据集，模型及权重。

介绍地址：25u9489z62.qicp.vip:52651
网页基于Streamlit，
但没有服务器，只做了内网穿透，
大致预览图片在目录source里面,
要想具体查看，需要本地开启服务。

发邮件联系或者qq:2913397682@qq.com

毕业论文待答辩之后上传

code介绍：
Auxiliary.py	辅助绘制loss与accuracy，但后来发现有tensorboard更好用
Image_preprocessing.py	有关人脸检测与裁剪，及后续预处理
__init__.py	
amsoftmax.py 引入AM-Softmax，具体看https://github.com/happynear/AMSoftmax
configure.py	相关参数
data_input.py	数据输入，人脸识别和验证
detect_gpu.py检测是否安装好tf-gpu
model_train.py	人脸识别模型，基于AlexNet，简易流程都有
model_train_2.py	人脸验证模型，基于SiameseNet，简易流程都有
some_data_deal.py	有关统计、复制等杂项
test_alex.py	测试人脸识别
test_sia.py  测试人脸验证
test_streamlit.py  网页展示
auxiliary_streamlit.py  网页分支管理
auxiliary_page_face_recognition.py  网页-人脸识别部分
auxiliary_page_face_validation.py  网页-人脸验证部分
auxiliary_page_homepage.py  网页-主页介绍
auxiliary_page_summary.py  网页-小结
