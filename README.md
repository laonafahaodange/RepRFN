# RepRFN
## Reparameterized Residual Feature Network For Lightweight Image Super-Resolution
Weijian Deng, Hongjie Yuan, Lunhui Deng, Zengtong Lu; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2023, pp. 1712-1721

用于轻量化超分辨率的重参数残差特征网络（RepRFN）

如果论文或项目对您有所帮助，欢迎引用。
[论文地址](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Deng_Reparameterized_Residual_Feature_Network_for_Lightweight_Image_Super-Resolution_CVPRW_2023_paper.html)
```
@InProceedings{Deng_2023_CVPR,
    author    = {Deng, Weijian and Yuan, Hongjie and Deng, Lunhui and Lu, Zengtong},
    title     = {Reparameterized Residual Feature Network for Lightweight Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {1712-1721}
}
```

---
**更新日志**
* 2023.04.03
  * 更新NTIRE2023 ESR相关文件
* 2023.06.05
  * 更新RepRFN训练/验证相关代码

---
## NTIRE2023 ESR
1. 解压缩`./data`下的测试集压缩包
2. 运行`test_reprfn_ntire2023esr.py`，以获得结果

---
## RepRFN训练/推理框架
### 已实现
- [x] 训练/测试SISR模型框架
- [x] 自定义损失函数功能
### 待验证
- [ ] 不同的data range训练
- [ ] 断点续练功能
- [ ] 随机数种子
- [ ] 只调用CPU
### 待实现
暂无，欢迎提issue。
### 坑
有很多，待触发，欢迎提issue。

---
### 使用方式
#### 参数介绍
##### 1. 模式相关选项说明
- **`--mode`**
  可选参数`train`或`test`控制`main.py`执行训练或测试
##### 2. 硬件设备相关选项说明
- **`--n_threads`**
  加载数据的线程数
- **`--cpu`**
  只使用CPU，**该功能待验证**
- **`--n_GPUs`**
  使用GPU的数量
##### 3. 数据集相关选项说明
- **`--train_hr_dir`**
  训练集HR图像路径
- **`--train_lr_dir`**
  训练集LR图像路径
- **`--valid_hr_dir`**
  验证集HR图像路径
- **`--valid_lr_dir`**
  验证集LR图像路径
- **`--test_hr_dir`**
  测试集HR图像路径
- **`--test_lr_dir`**
  测试集LR图像路径
- **`--augment`**
  是否使用数据增强，包括水平/垂直翻转，旋转90°，概率均为50%
##### 4. 模型相关选项说明
- **`--model`**
  模型名称，可自定义
- **`--scale`**
  缩放因子|尺度因子|放大倍数
- **`--data_range`**
  训练/测试输入模型数据的范围，可选`1`或`255`，分别表示输入数据的范围是`0-1`和`0-255`
- **`--data_format`**
  训练/测试输入模型数据的颜色空间，可选`bgr`或`rgb`或`ycbcr`
- **`--pre_train`**
  训练模式下表示预训练模型路径，测试模式下表示需要加载的模型权重文件路径
- **`--precision`**
  模型精度，可选`single`单精度(f32)，或`half`半精度(f16)
- **`--rep`**
  模型是否使用重参数
- **`--model_save_dir`**
  模型保存路径
- **`--checkpoint_save_dir`**
  模型检查点保存路径，以供断点续训使用
##### 5. 训练相关选项说明
- **`--seed`**
  随机数种子，**该功能待验证**
- **`--batch_size`**
  批大小
- **`--epochs`**
  训练轮数
- **`--loss_function`**
  损失函数，目前可选`L1`、`L2`、`Charbonnier`损失函数，`Custom`表示自定义损失函数
- **`--val_per_epoch`**
  每n个epoch验证一次模型在验证集上的PSNR/SSIM
- **`--log_dir`**
  保存训练日志的路径
##### 6. 断点续训相关选项说明
- **`--resume`**
  是否启动断点续训，**该功能待验证**
##### 7. 优化器相关选项说明
- **`--lr`**
  学习率
- **`--lr_scheduler`**
  学习率策略，目前支持`StepLR`和`MultiStepLR`
- **`--decay_step`**
  适用`StepLR`学习率策略，每decay_step个epoch学习率下降
- **`--decay_milestone`**
  适用于`MultiStepLR`学习率策略，学习率在设置的epoch处下降，设置参考`200-400-600-800`
- **`--gamma`**
  学习率衰减因子
- **`--optimizer`**
  优化器，可选`SGD`或`ADAM`或`RMSprop`
- **`--momentum`**
  默认`0.9`
- **`--betas`**
  默认`(0.9,0.999)`
- **`--epsilon`**
  默认`1e-8`，使用半精度时建议设置为`1e-3`
- **`--weight_decay`**
  权重衰减系数，默认`0`
- **`--gclip`**
  是否裁剪梯度，默认`0`表示不裁剪
##### 8. 测试相关选项说明
- **`--psnr_ssim_y`**
  是否在ycbcr颜色空间的y上通道上测试PSNR/SSIM，默认`True`
- **`--self_ensemble`**
  模型推理时是否使用self_ensemble，即在测试阶段，对输入低分辨率图像进行水平翻转、垂直翻转、旋转等操作，共生成八种低分辨率输入图像（包含原始低分辨率图像），利用这八种输入生成对应的超分辨率图像，随后对这些超分辨率图像执行逆变换返回原始角度并进行平均得到最终输出超分辨率图像
- **`--test_result_dir`**
  模型生成的SR图像保存路径

---
感谢以下工作。
- RFDN：https://github.com/njulj/RFDN
- FightingCV 代码库：https://github.com/xmu-xiaoma666/External-Attention-pytorch
- ECBSR：https://github.com/xindongzhang/ECBSR
- RLFN：https://github.com/bytedance/RLFN
- EDSR：https://github.com/sanghyun-son/EDSR-PyTorch
