# Segmentation_gaussian_strategy

---
## 1. 下载数据集
![image](https://github.com/user-attachments/assets/3bfcbd4b-30ce-485f-8991-070c8cab088e)

传送： `http://sceneparsing.csail.mit.edu/`

---
可放于根目录，**记得** 在 `configs/ade20k.yaml` 里面修改对应数据集的位置
```
  DATASET:
  NAME          : ADE20K                                              # dataset name to be trained with (camvid, cityscapes, ade20k)
  ROOT          : 'C:\Users\magic\Desktop\semantic-segmentation-main\semantic-segmentation\ADEChallengeData2016\'                         # dataset root path
  IGNORE_LABEL  : -1
```

---
## 2. 下载模型
![image](https://github.com/user-attachments/assets/585e5c1e-872f-4e9a-b342-e04479847ccc)

传送： 在上面的



---
## 2. 文件解释
***double_side_pert.py*** 是双边（正负加噪），这个用于跑 ***metrics***

---
***multi_pert.py*** 是用于生成分割图的 `多类别` 加噪
具体在 ***infer_gaussian.py*** 里面，可以将：
```
  # 定义目标类别的扰动设置
        target_perturbations = [
            {'index': self.person_idx, 'positive': True},  # 对人类别添加正数噪声
            {'index': self.car_idx, 'positive': False},
            {'index': self.tree_idx, 'positive': False} # 对车or树类别添加负数噪声
        ]
```
里面的对应代码注释掉，即可只对单类别加噪

---
