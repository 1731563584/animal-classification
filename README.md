# 动物分类深度学习项目
> 本项目是本人学习深度学习课程最后要求做的课程设计，也算是脑子一热就随便挑了个数据集，就开始训练做下去了。
>

本项目使用深度学习技术对 10 种动物进行分类。我们使用了 EfficientNetV2B0 架构，并对其进行了改进以提高性能和准确率。

## 概述
该项目旨在学习并训练一个高效的动物图像分类模型，能够识别 10 种不同的动物类别。我们使用了EfficientNetV2B0 作为基础模型，并对其进行了一系列优化和改进。

## 文件结构
```python
animal-classification/
│
├── traindata/               # 训练数据集，包含压缩文件
├── testdata/                # 测试数据集，包含压缩文件
├── pre/                     # 预处理和预测相关文件
│   ├── train.py  			 # 训练脚本
│   ├── predict.py			 # 预测脚本
│   ├── results.json         # 测试集图片序号、真实标签文件
│	├── train_curves.png	 # 训练过程图
│	├── model.keras          # 训练好的模型
│   └── confusion_matrix.png # 预测结果的混淆矩阵图
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明文档
```

## 数据集
+ **来源：**[https://aistudio.baidu.com/datasetdetail/140388/0](https://aistudio.baidu.com/datasetdetail/140388/0)（测试集没有标签，需要自行标注，本人的解决方案是使用免费大模型 GLM-4V-Flash 配合合适的 prompt 进行标注的。）
+ **训练集**: 14,246张图片
+ **测试集**: 8,150张图片
+ **动物类别**: 蝴蝶、猫、鸡、牛、狗、大象、马、蜘蛛、羊、松鼠

注意，本项目中的 results.json 就是借助大模型对测试集进行标注的图片序号与真实标签对，可能部分存在错误，null 指的是不属于该 10 类动物的任何一种。

## 模型架构
![](https://cdn.nlark.com/yuque/0/2025/png/40475367/1749363742740-0e953750-7bee-4112-81d7-757401859356.png)

我们采用了预训练的 EfficientNetV2B0 模型作为基础架构，并在此基础上添加了以下层：

+ GlobalAveragePooling2D
+ Dropout (0.5)
+ Dense (512个单元，swish 激活函数)
+ BatchNormalization
+ Dropout (0.3)
+ Dense (256个单元，swish 激活函数)
+ Dense (10个单元，softmax 激活函数)

## 技术细节
1. **数据增强**:
    - 使用 Albumentations 库实现类别特定的数据增强策略
    - 针对猫和狗类动物实施专门的增强方法
2. **类别权重平衡**:
    - 使用 sklearn 的 compute_class_weight 计算类别权重
    - 对猫类样本权重加倍，以应对数据不平衡问题
3. **训练策略**:
    - 第一阶段: 冻结基础模型进行头部训练
    - 第二阶段: 解冻部分深层特征进行微调
    - 使用动态更新的类别权重
4. **优化器和损失函数**:
    - Adam 优化器 (初始学习率为1e-4)
    - categorical_crossentropy 损失函数
5. **回调函数**:
    - ReduceLROnPlateau (监控召回率)
    - EarlyStopping (重点关注猫类召回率)
6. **评估指标**:
    - 准确率(Accuracy)
    - 召回率(Recall)
    - 精度(Precision)
    - 类别特定的召回率和精度(特别是猫和狗类)

## 训练过程
> 小声 bb：其实第一阶段 1 个 epoch、第二阶段微调 1 个 epoch 就可以达到较高性能，得益于基础模型EfficientNetV2B0 的强大。（虽然我还是没想到该如何解决后续遇到的问题）
>

1. **第一阶段训练**:
    - 冻结基础模型
    - 训练头部网络 10 个epoch
    - 使用类别权重平衡
2. **第二阶段训练**:
    - 解冻最后 15 层
    - 继续训练 10 个epoch
    - 动态更新类别权重
    - 使用动态学习率衰减和早停机制

![](https://cdn.nlark.com/yuque/0/2025/png/40475367/1749361895751-1eb52dbd-d55b-487a-a53e-ab94c9d3d33a.png)

## 结果
我们的模型在测试数据集上达到了优异的性能表现:

+ **总体准确率**: 95~96%
+ **平均精度**: 95%
+ **平均召回率**: 95%

![](https://cdn.nlark.com/yuque/0/2025/png/40475367/1749361999920-e6720a87-fc1a-47d3-aa76-1fcc8bbd824f.png)

```python
255/255 ━━━━━━━━━━━━━━━━━━━━ 55s 210ms/step
Precision per class:
butterfly: 0.9883
cat: 0.9710
chicken: 0.9883
cow: 0.9343
dog: 0.8250
elephant: 0.9699
horse: 0.9746
ragno: 0.9961
sheep: 0.9713
squirrel: 0.9898

Class counts:
butterfly: 594
cat: 783
chicken: 762
cow: 598
dog: 1321
elephant: 455
horse: 798
ragno: 1523
sheep: 601
squirrel: 589
unknown: 126

Classification Report:
              precision    recall  f1-score   support

   butterfly       0.99      0.99      0.99       594
         cat       0.97      0.64      0.77       783
     chicken       0.99      0.99      0.99       762
         cow       0.93      0.97      0.95       598
         dog       0.83      0.98      0.90      1321
    elephant       0.97      0.99      0.98       455
       horse       0.97      0.96      0.97       798
       ragno       1.00      1.00      1.00      1523
       sheep       0.97      0.96      0.96       601
    squirrel       0.99      0.98      0.99       589

    accuracy                           0.95      8024
   macro avg       0.96      0.95      0.95      8024
weighted avg       0.95      0.95      0.95      8024
```

## 存在的问题
猫类召回率较低（约60%），狗类精度不足（约80%），且猫类存在明显过拟合。尽管尝试了包括过采样、裁剪训练集、增强猫狗特征、调整初始权重及动态权重等多种方法，但问题仍未解决。期待网友能提供宝贵建议，帮助优化模型性能。

## 鸣谢
在此，谨向胡伊洋同学、露露网友为完善本模型提出的宝贵建议致以最诚挚的谢意！

同时，也要特别感谢TensorFlow、Keras等开源框架，以及EfficientNetV2B0模型的开发团队。他们的卓越贡献为我们的研究工作提供了重要的支持和帮助。

## 最后
初次尝试将项目上传到 GitHub 平台，恳请多加指导，感谢支持！

