# Advanced-computer-vision
高级计算机视觉作业
运行说明

实现实例分割：
 pip install cython
 pip install opencv-oython pillow pycocotools matplotlib
 在YOLACT网址https://github.com/dbolya/yolact 下载模型
 cd yolact
 #处理相关的图片并将它存储
 python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.jpg:output_image.jpg

生成相应的mask：
 #运行时注意将代码中所指的图片位置进行相应的修改
 运行mask_test.py代码
 
实现人体三维姿态估计：
  pip install configargparse pyyaml scipy easydict
  #将路径定位到下载的CVPR-OOH下 将测试的图片可以加入demo文件夹中，
  #demo文件夹含有我自己测试图片
  python demo.py --config cfg_files\demo.yaml
  
参考相关网址：
 https://github.com/dbolya/yolact
 https://gitee.com/seuvcl/CVPR2020-OOH.git
