# Segmentation_gaussian_strategy

***base：***
`https://github.com/sithu31296/semantic-segmentation/tree/main`

---
## 1. 下载数据集
![image](https://github.com/user-attachments/assets/3bfcbd4b-30ce-485f-8991-070c8cab088e)

传送： `http://sceneparsing.csail.mit.edu/`

可放于根目录，**记得** 在 `configs/ade20k.yaml` 里面修改对应数据集的位置
```
  DATASET:
  NAME          : ADE20K     # dataset name to be trained with (camvid, cityscapes, ade20k)
  ROOT          : 'C:\Users\magic\Desktop\semantic-segmentation-main\semantic-segmentation\ADEChallengeData2016\'   # dataset root path
  IGNORE_LABEL  : -1
```

---
## 2. 下载backbone
![image](https://github.com/user-attachments/assets/585e5c1e-872f-4e9a-b342-e04479847ccc)

传送：我忘了在哪下的了😅

记得改configs：
```
  MODEL:                                    
  NAME          : SegFormer                                           # name of the model you are using
  BACKBONE      : MiT-B2                                                  # model variant
  PRETRAINED    : 'checkpoints/backbones/mit/segformer.b2.ade.pth'              # backbone model's weight
```

---
## 3. 文件解释
- ***double_side_pert.py*** 是双边（正负加噪），这个用于跑 ***metrics***

---
- ***multi_pert.py*** 是用于生成分割图的 `多类别` 加噪
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
- ***val_dual.py*** 是跑双边噪声的全部`metrics`的（已经跑完了）

---
- ***infer_gaussian.py*** 是生成分割图的文件。
  在`postprocess`里面：
  ```
     # 应用颜色映射
        seg_image[seg_map == self.person_idx] = yellow  # 人
        seg_image[seg_map == self.car_idx] = blue  # 车
        seg_image[seg_map == self.tree_idx] = green  # 树
        seg_image[(seg_map != self.person_idx) &
                  (seg_map != self.car_idx) &
                  (seg_map != self.tree_idx)] = purple # 其他类别

        # if overlay:
        #     seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)
  ```
  可以修改染色，以及是否要半透明（`overlay`）

  运行分割代码为： `python tools/val_dual.py --cfg configs/ade20k.yaml`

---
- ***val_person_positive.py*** 是跑`metrics`的， 人加正噪，其他不变

---
- ***one_target_pert.py*** 是跑上面那个`metric`会用到的加噪方式（只对`target_index`这里是人为12加正噪）
