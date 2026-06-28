# 排查指南：绿幕上的彩色噪声（色度噪声）

在处理绿幕（或大面积单色均匀背景）时，如果画面上出现斑驳的彩色噪点（红蓝绿交织的色斑），这属于 ISP 中经典的**色度噪声（Chroma Noise）**问题。

---

## 1. 复现步骤 (Reproduction Steps)

在当前未配置去噪算法的 `cuda_isp` 环境下，可以通过运行以下命令来复现该现象：

```bash
# 运行 ISP 管道处理室内 10-bit RAW 图像并输出为 PNG
./build/cuda_isp data/infinite/Indoor1_2592x1536_10bit_GRBG.raw data/infinite/Indoor1_2592x1536_10bit_GRBG.png
```

**现象观察**：
* 观察生成的图像 `data/infinite/Indoor1_2592x1536_10bit_GRBG.png`，在画面左上方大面积绿色的毛毡板区域中，可以明显观察到彩色的噪点和不均匀的低频色斑。

---

## 2. 诊断与排查方法 (How to Troubleshoot)

在调试 ISP 时，通常通过**控制变量法（模块旁路 / Bypass）**和**中间步骤图像转储（Dumping）**来精确定位哪个模块对噪声的贡献或放大作用最大：

1. **旁路 CCM（色彩校正矩阵）测试**：
   * **操作**：将 CCM 临时设为单位矩阵（对角线为 1.0，其余为 0.0，即不改变色彩）。
   * **观察**：如果画面彩色噪点瞬间减弱大半（虽然画面颜色会变淡/偏灰），则证实 **CCM 的饱和度增益是彩色噪声最主要的数学放大器**。
2. **旁路 WB（白平衡）测试**：
   * **操作**：将白平衡的 R/Gr/Gb/B 增益全部设为 1.0（不做白平衡调整）。
   * **观察**：画面会严重偏绿（绿幕下正常现象），但如果红蓝彩色噪点的闪烁和强度明显减弱，说明 **白平衡增益对 R/B 通道的低信噪比噪声进行了二次放大**。
3. **分析 Demosaic 前后的图像**：
   * **操作**：转储 Demosaic 前的 RAW 图像（查看单通道噪声分布）和 Demosaic 后的 RGB 图像。
   * **观察**：如果在 RAW 域上只是非常微小、呈像素点状的噪声，而 Demosaic 后变成了成片的大色斑，这说明 **Demosaic 算法在插值时缺乏色差平滑，导致高频噪声向空间邻域扩散，演变为低频彩色噪声**。
4. **旁路 LSC（镜头阴影校正）测试**（如果开启了该模块）：
   * **操作**：在管道中关闭 LSC 模块。
   * **观察**：如果画面边缘区域的彩色噪点显著降低，回落到与中心区域相同的水平，说明 **LSC 对边缘施加的高补偿增益（1.5x ~ 3x）是导致边缘噪声恶化的主因**。

---

## 3. 核心根因分析 (Root Causes)

彩色噪声并非单一模块产生，而是由图像传感器物理噪声经 ISP 多个模块放大和扩散导致的：

| 模块/阶段 | 物理与数学机制 | 影响结果 |
| :--- | :--- | :--- |
| **Sensor (传感器)** | 存在散粒噪声（Shot Noise）、热噪声等基础物理噪声。 | 提供了最初的微小物理噪声源。 |
| **White Balance (白平衡)** | 绿幕中绿光极强，R 和 B 通道信号极弱，WB 对其应用了较大增益。 | 成倍放大了 R 和 B 通道的基础噪声。 |
| **Demosaic (去马赛克)** | 邻域插值将单像素的 R/B 随机高频噪点扩散到周围像素。 | 将高频点状噪声转化为了**空间更大、更显眼的低频色斑**。 |
| **CCM (色彩校正矩阵)** | 增强饱和度的 CCM 矩阵（对角线系数大、非对角线系数为负）会放大输入噪声功率。 | **极大地放大了色度噪声**，使其变得色彩鲜艳且刺眼。 |
| **LSC (镜头阴影校正)** | 对镜头边缘暗角进行亮度补偿（边缘增益高达 1.5x ~ 3x）。 | 导致**画面边缘的彩色噪声明显比中心更严重**。 |

---

## 4. 行业标准解决方案 (ISP Solutions)

在标准的 ISP 管道中，通常通过以下几步协同解决：

1. **RAW 域去噪 (RAW Denoise / BNR)**：在 Demosaic 之前，对 Bayer 四通道分别进行空间滤波（如双边滤波 Bilateral Filter 或非局部均值 NLM）。**在源头上阻止噪声被 Demosaic 扩散**。
2. **YUV 域色度去噪 (Chroma Denoise / CNR)**：将图像转换至 YUV 空间，对 U/V 通道进行较强力度的平滑（人眼对色度细节不敏感，模糊 U/V 通道不会影响图像的主观清晰度）。**这是消灭彩色斑块最有效的手段**。
3. **CCM 降噪约束 (CCM Noise Constraint)**：在高 ISO 或暗光下，降低 CCM 的饱和度增益，优先保证信噪比。
4. **色度抑制 (Chroma Suppression)**：对暗区或高饱和区域进行适度降饱和（脱色），使彩色噪声退化为不刺眼的灰色噪声。

---

## 5. 本项目 (Cuda-ISP) 的改进建议

当前项目的 ISP 管道（参见 [include/blocks.h](file:///home/gxh1991/cuda_isp/include/blocks.h)）包含以下模块：
`RawUnpack` $\rightarrow$ `BlackLevelCorrection` $\rightarrow$ `DeadPixelCorrection` $\rightarrow$ `WhiteBalance` $\rightarrow$ `Demosaic` $\rightarrow$ `CCM` $\dots$

由于**当前管道中没有去噪（Denoise）模块**，因此直接暴露出严重的彩色噪声是正常现象。如果想在本项目中解决此问题，可以尝试以下改进：

* **短期/快捷方案 (YUV Denoise)**：
  在 `OutputPack` 之前（或 Gamma 之后），将 RGB 转为 YUV 空间，对 `U` 和 `V` 分量实现一个简单的 **CUDA 2D 双边滤波器 (Bilateral Filter)** 或 **中值滤波器 (Median Filter)**，然后再转回 RGB。这能立竿见影地消除绿色背景上的彩色杂点。
* **中期方案 (RAW Denoise)**：
  在 `DeadPixelCorrection` 之后、`Demosaic` 之前，添加一个 `RawDenoise` 模块，对 Bayer 图像的 R/Gr/Gb/B 四个通道分别进行滤波。
