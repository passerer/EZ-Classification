# EZ-Classification
独立完成的基于Pytorch的图像分类框架
## MOTIVATION
目前有很多在pytorch上进一步封装的开源框架比如fastai等，但存在可定制性差等问题。本框架完全争对图像分类任务且没有引入其他概念，更简单易读和修改
##  SUPPORT
- [x] fp16训练
- [x] 训练集和验证集随机划分
- [x] 额外的增强方式
    - [x] cutout
    - [x] random erase
- [x] warm up
- [x] 展示增强后的图像
- [x] 常用网络模型
    - [x] ResNet  (引入residual模块)
    - [x] Xception  (引入Inception模块)
    - [x] DarkNet  (对ResNet的改进)
    - [x] DenseNet  (引入dense模块)
    - [x] ResNext  (加宽的ResNet)
    - [x] cbam_ResNet  (spatial attention)
    - [x] SeNet  (channel attention)
    - [x] EfficientNet
## DEMO
[demo.py](链接网址 "标题")展示了如何在imet-2020数据集(27G)上训练
## TODO
- [ ] 各种 loss
- [ ] 各种 学习策略
- [ ] 各种 评价指标
- [ ] K折交叉验证
- [ ] 可视化训练过程
- [ ] 可视化混淆矩阵
- [ ] GAN
- [ ] 模型集成
- [ ] early stop
- [ ] find first lr
- [ ] 显存优化
