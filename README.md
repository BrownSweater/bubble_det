# 一、项目背景

帮助朋友完成一个简单的图像检测任务。

- 1、球体检测。
- 2、运行在树莓派上，要求速度足够快。

由于自己之前做过的视觉任务都是运行在GPU或者服务器上的CPU上，所以手中现有的项目可能无法满足此需求。由此展开调研，最后选用了最新开源的[nanodet](https://github.com/RangiLyu/nanodet)项目，简而言之就是轻量化了**FCOS**。有关**FCOS**的论文细节我在简书里已经总结了：https://www.jianshu.com/p/e988c15f7d33。

![bubble](docs/bubble.gif)

# 二、安装

- 主要依赖：

  - Linux or MacOS
  - Python >= 3.6
  - Pytorch >= 1.3
  - 其他依赖见：`requirments.txt`

- 安装:

  ```shell
  git clone https://github.com/BrownSweater/bubble_det.git
  cd bubble_det
  python setup.py develop
  ```



# 三、项目流程

## 1、图像标注

使用Labelme进行矩形框的标注。

## 2、图像格式转换

- **labelme2voc**：将labelme的标注格式转换为VOC2007的格式，默认将15%的数据作为验证集。

  ```shell
  # 修改labelme2voc.py中的labelme_path变量为标注图片的路径
  python labelme2voc.py
  ```

- **voc2coco**：将VOC2007的格式转换位coco2012格式

  ```shell
  # 训练集的转换
  python voc2coco.py --ann_dir VOC2007/Annotations/ --ann_ids VOC2007/ImageSets/Main/train.txt  --labels label.txt --output train.json --ext xml
  # 验证集的转换
  python voc2coco.py --ann_dir VOC2007/Annotations/ --ann_ids VOC2007/ImageSets/Main/val.txt  --labels label.txt --output val.json --ext xml
  ```

## 3、算法训练

根据**[nanodet](!https://github.com/RangiLyu/nanodet)**的文档，准备配置文件，进行训练。

```shell
python tools/train.py config/bubble.yml
```

## 4、模型评估

由于数据集比较小，任务简单，所以效果非常好。

- `VOC MAP`：1.0
- `COCO MAP`：0.714



# 四、模型调用

- 训练的模型文件：`models/model_bubble.pth`
- 接口的配置文件：`config.json`

## 1、配置config.json

| 参数名       | 类型  | 描述                                   |
| ------------ | ----- | -------------------------------------- |
| config       | str   | `yaml`格式的模型配置文件，使用绝对路径 |
| device       | str   | `cpu`、`gpu`                           |
| model        | str   | 模型文件路径，使用绝对路径             |
| score_thresh | float | 阈值                                   |

```json
# 参考示例，项目路径下的config.json
{
    "config":"/Users/wangjunjie/project/nanodet/config/bubble.yml",
    "device": "cpu",
    "model": "/Users/wangjunjie/project/nanodet/models/model_bubble.pth",
    "score_thresh": 0.5
}
```

设置环境变量：`DET_CONFIG_PATH`为你配置的`config.json`的路径

- 使用`shell`进行配置：`export DET_CONFIG_PATH=YOUR_CONFIG_PATH`
- 使用`python`进行配置：`os.environ['DET_CONFIG_PATH'] = YOUR_CONFIG_PATH`

## 2、接口格式

- 输入：BGR格式的ndarray数组，可以使用opencv读取
- 输出：`List[LIST[x0, y0, x1, y1, score, label_id]]`，嵌套的列表，外层的`List`表示图片中检测到的不同对象，里层的`LIST`存储了该检测对象的结果。依次为左上顶点的x坐标、左上顶点的y坐标、右下顶点的x坐标、右下顶点的y坐标、得分、标签id（只有一类可以忽略）。如：`[[569.60693359375, 554.2196044921875, 652.0204467773438, 637.24267578125, 0.8876911997795105, 0]]`。

## 3、接口调用方法

```python
from nanodet.Interface import pedictor
# img是一个BGR的ndarray数组，可以使用opencv读取
# 如：img = cv2.imread(YOUR_IMG_PATH)

# 模型预测
res = predictor(img)
# 结果展示
predictor.visualize()

---------------------------------------------------------------------------------------------------

# 模型预测
res = predictor(img)
# 结果展示及保存图片
predictor.visualize(save_path='YOUR_SAVE_IMG_PATH')
```



# 五、对比原仓库

- 1、修改原仓库读取COCO格式的数据集`img_id`字段一定是`int`类型为`str`类型
- 2、原仓库使用`torch>=1.6`版本进行训练，所得到的模型低于此版本的`torch`无法调用
- 3、增加了`labelme2voc`和`voc2coco`的脚本
- 4、增加了`Interface`接口

......

