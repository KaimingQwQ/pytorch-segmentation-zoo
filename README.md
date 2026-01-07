#  PyTorch Segmentation Zoo

##  项目简介 (Description)

本项目是基于 **SBD (Semantic Boundaries Dataset) 增强的 Pascal VOC 2012 数据集**，复现了从经典 CNN（FCN, DeepLab, RefineNet）到前沿 Transformer（Swin-UNet, SETR, ViT-OCR）的主流分割模型。

该项目不仅包含了各个模型的复现代码，还实现了一套包含 **混合精度训练 (AMP)**、**Lovász-Softmax Loss 微调** 以及 **自动可视化** 的完整训练流水线，希望能帮助到更多CV领域的初学者。
> 本人仍在持续学习中，实现可能并非最优。<br>
> 如有不当之处，恳请各位大佬不吝指教，欢迎交流讨论 🙏

##  Model Zoo (支持模型列表)

本项目集成了以下模型（截至 2026/1/7），均可通过 config.py 一键切换：


| Category | Models |
| -------- | ------ |
| Transformer-based | SwinUNet<br>SwinDeepLab<br>SETR (Naive / PUP)<br>ViT_OCR |
| DeepLab Series | DeepLabV3+<br>DeepLabV3 |
| Attention Modules | DeepLabV3+ (SE)<br>DeepLabV3+ (CBAM) |
| Multi-Scale / Refine | RefineNet (ResNet152)<br>RefineNet-MultiScale<br>PSPNet |
| Classic CNN | ResNet-UNet<br>FCN-8s (ResNet / VGG) |


## Benchmark (性能榜单) 
Based on *Pascal VOC 2012 (Augmented)* validation set  
Image Size: **512 × 512**

# Coming soon

| Model Name        | Backbone     | Pretrained | mIoU (Val) | Pixel Acc |
| ----------------- | ------------ | ---------- | ---------- | --------- |
| --         | --    | -        | —          | —         |


## **📂 Dataset Preparation (数据集配置)**

本项目默认支持 Pascal VOC 2012（含 SBD 增强数据集）。  
由于 SBD 数据集提供了更精细的边缘标注且数据量更大，请严格按照以下步骤进行数据准备。

### **1\. 最终数据集目录结构**

准备完成后，你的数据集根目录 Dataset/ 应如下所示：

```plaintext
Dataset/
└── VOC2012/
    ├── JPEGImages/            # 存放所有 .jpg 原图
    ├── SegmentationClass/     # 存放官方提供的 .png 标签 (21 classes)
    ├── SegmentationClassAug/  # [重要] 存放增强后的 .png 标签 (TrainAug)
    └── ImageSets/
        └── Segmentation/
            ├── train.txt      # 官方训练列表
            ├── val.txt        # 官方验证列表
            └── train_aug.txt  # [重要] 包含增强数据的训练列表
```

### **2\. SBD 数据集处理流程**

SBD 原始格式为 .mat，需经过以下三步处理才能适配本项目：

#### **🛠️ Step 1: 标签格式转换 (Format Conversion)**

SBD 标签通常存储在 dataset/cls/\*.mat 中。

* **操作**: 编写脚本遍历 .mat 文件，读取 GTcls.Segmentation 矩阵。  
* **注意**: 将其保存为 .png 时，**必须应用 Pascal VOC 的官方调色板 (PASCAL Palette)**。否则可视化颜色会混乱，且训练时的 Ignore Index (255) 可能失效。

#### **📂 Step 2: 标签与图片合并 (Merge Data)**

你需要创建一个新的文件夹 SegmentationClassAug。

* **操作 A**: 将 VOC2012 原生 SegmentationClass/ 中的所有图片复制到 SegmentationClassAug/。  
* **操作 B**: 将转换好的 SBD .png 标签也复制到该文件夹。  
* **冲突策略**: 若同一图片存在于两者中，建议使用 SBD 版本覆盖 VOC 版本（SBD 标注通常更精细）。同时，请确保 SBD 中新增的原始图片 (.jpg) 也被复制到了 JPEGImages/ 中。

#### **📝 Step 3: 生成索引列表 (Generate List)**

生成关键的 train\_aug.txt 文件。

1. 读取 VOC 的 train.txt (1464 张)。  
2. 读取 SBD 的 train.txt (8000+ 张)。  
3. **取并集**：合并两者。  
4. **去重与剔除**：**重要！** 必须从合并列表中剔除掉 VOC val.txt (1449 张) 中的图片，以防止验证集泄露。  
5. 将最终剩余的 ID ( 10,582 张) 写入 train\_aug.txt。

## **🚀 Getting Started (快速开始)**

### **1\. 环境依赖**
```plaintext
pip install torch torchvision albumentations pandas matplotlib tqdm opencv-python
```
### **2\. 配置参数**

所有超参数均在 config.py 中集中管理：

* **切换模型**: 修改 MODEL 变量及相关 Import。  
* **微调策略**: 设置 USE\_LOVASZ\_FINETUNE \= True 可开启后期 Lovász Loss 优化。  
* **路径配置**: 修改 DATA\_DIR 指向你的数据集根目录。

### **3\. 开始训练**

\# 自动识别单卡/多卡模式 
```
python train.py
```
训练日志将自动保存至 output/ 目录，格式为 ModelName\_Timestamp.csv。

### **4\. 结果可视化**

训练完成后，使用内置脚本一键生成图表：

\# 自动绘制 Loss 曲线, mIoU 趋势, Pixel Accuracy 和 每类 IoU 对比  
```
python plt.py \--csv output/Your\_Log\_File.csv
```
生成的图片将保存在 output/plots/ 目录下。




## License

This project is released under the MIT License.
