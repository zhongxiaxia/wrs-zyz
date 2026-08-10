# 辣椒采摘实验（单次运行、单次记录）

本目录中的 `Pepper01.py`～`Pepper07.py` 是七个正式实验入口。每次只运行
当前文件中选择的一组条件，不会自动展开全部组合；运行结果会向对应的 CSV
文件追加一行。

七个实验统一通过 `single_run_experiment_common.py` 创建采摘器、读取公共参数、
执行采摘并记录结果，不再使用 `experiment_common.py`。

## 运行方式

例如运行实验2：

```powershell
python wrs/Fruit_picker_project/Experiment/Pepper02.py
```

程序显示实验条件后会等待按 Enter。确认机械臂、相机、夹爪、辣椒和周围障碍物
均已准备好，再开始实验。

## 七个实验

- `Pepper01.py`：10 cm、中速、夹爪最快；比较四种规划/运动学组合的单果采摘。
- `Pepper02.py`：10 cm、中速、夹爪最快；比较无 RRT 与
  RRT-Connect + Native JLChain/KDL 的单果采摘。
- `Pepper03.py`：10 cm、夹爪最快；比较慢、中、快三档机械臂速度。
- `Pepper04.py`：10 cm、夹爪最快；比较慢、中、快三档机械臂速度。
- `Pepper05.py`：10 cm、中速、夹爪最快；初始识别失败后依次尝试
  +30°、-30°观察位的单果遮挡采摘。
- `Pepper06.py`：10 cm、中速、夹爪最快；按置信度优先或 pick 点深度优先，
  连续采摘十个辣椒中的5个。
- `Pepper07.py`：10 cm、快速、夹爪最快；按置信度优先或 pick 点深度优先，
  连续采摘十个辣椒中的5个。

## 在哪里调整机械臂速度

速度设置分为“选择档位”和“修改档位数值”两部分：

1. 选择某次实验使用的档位：修改对应 `PepperXX.py` 顶部配置区中的
   `SPEED_NAME`，可选值为 `"slow"`、`"medium"`、`"fast"`。
   实验1按实验设计固定为 `"medium"`；实验3和实验4用于比较三档速度。
2. 修改慢速、中速、快速对应的实际关节速度或加速度：只修改
   `single_run_experiment_common.py` 顶部的 `SPEED_PROFILES`。七个实验都会读取
   这里的统一数值，不要在各个 `PepperXX.py` 中重复填写关节速度数组。

当前统一参数如下，数组依次对应机械臂的6个关节：

| 档位 | 转移段与接近段速度（°/s） | 转移段与接近段加速度（°/s²） |
| --- | --- | --- |
| slow | `[100, 100, 150, 150, 150, 150]` | `[200, 200, 300, 300, 300, 300]` |
| medium | `[150, 150, 200, 200, 200, 200]` | `[300, 300, 400, 400, 400, 400]` |
| fast | `[200, 200, 300, 300, 300, 300]` | `[400, 400, 600, 600, 600, 600]` |

转移段和接近段使用相同的速度、加速度限制。夹爪速度统一由
`single_run_experiment_common.py` 中的 `GRIPPER_SPEED_COMMAND = 255` 控制。

## 规划方式

实验1的 `PLANNING_METHOD` 可以选择：

- `norrt_python`：无 RRT + Python JLChain。
- `rrt_python`：传统 RRT + Python JLChain。
- `rrt_connect_python`：RRT-Connect + Python JLChain。
- `rrt_connect_native`：RRT-Connect + Native JLChain/KDL。

其余实验允许的规划方式以对应 `PepperXX.py` 中的 `ALLOWED_METHODS` 为准。

## 目标选择策略

- `highest_confidence`：按照检测关键点置信度与 pick 目标框置信度从高到低选择。
- `nearest_depth`：优先选择 pick 点在深度图中的有效深度最小、最靠近相机的目标。
- `nearest_tcp`：优先选择世界坐标中最靠近当前机械臂末端的目标。

三种策略都会先过滤 IK 无解或超出关节安全范围的目标。

## 计时边界

- 拍照时间：等待相机帧并取得一帧对齐的彩色图、深度图及内参。
- 识别时间：YOLO 推理并提取目标框和关键点。
- 抓取到果实时间：识别完成后开始，包括 IK、路径规划、机械臂到达抓取位及
  夹爪闭合等待，至夹爪闭合阶段结束。
- 投放果实时间：夹爪闭合阶段结束后开始，至机械臂到达投放位且夹爪张开命令
  返回。

实验5的拍照和识别时间为实际尝试过的所有观察位之和。实验6、实验7同时记录
5个果实的总时间和按实际成功采摘数量计算的单果平均时间。

## 输出

- 汇总表：`Experiment/results/PepperXX_results.csv`
- 单次详细记录：`Experiment/results/PepperXX_runs/<运行编号>/`
- 机械臂运动路径：单次详细记录目录中的 `trajectory/`

即使采摘失败，程序也会写入一行结果；状态和错误原因会保存在 CSV 与
`result_details.json` 中，避免遗漏失败样本。
