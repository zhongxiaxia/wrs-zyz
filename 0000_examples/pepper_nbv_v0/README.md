# 辣椒 NBV v0

保留 v0 的概率体素 / TSDF 视角评分算法，场景模型复用相邻的
`pepper_nbv_v1` 工程。请保留这两个目录及仓库中的共享资源。

当前视角策略为候选采样。向论文 Gradient-NBV 补全的方法差距、官方代码差异、
Windows/WRS 适配顺序和验收标准见
[Gradient-NBV MVP 补全方案](docs/GRADIENT_NBV_MVP_ROADMAP.md)
（2026-10-21，方案尚未实现为新规划器）。

在仓库根目录运行机械臂版本：

```powershell
& .venv/Scripts/python.exe 0000_examples/pepper_nbv_v0/pepper_robot_nbv.py
```

按 **M** 执行一次 NBV 规划，按 **Q** 退出。

## 场景配置

- 机械臂使用 v1 的 Realman、小车模型、初始关节角及
  `manual_calibration_realman22.json` 手眼标定。
- D405 相机位姿由机械臂 FK 和手眼变换计算；相机外形、主窗口视角、地面、
  植株摆放与 v1 的 `scene_viewer.py --no-leaf` 默认场景一致。
- 显示的整株模型来自 `0000_models/pepper/peppers_all.stl`、
  `stems_all.stl`、`plant_other_all.stl`，颜色与 v1 相同。
- 每次机械臂 NBV 移动复用 v1 的 IK/FK 误差、关节限位和离散关节路径检查，
  包括自碰撞、小车/整株植株及相机与障碍碰撞。
  这仍是仿真中的离散路径检查，不包含连续碰撞认证或绕障路径搜索。

## 其他入口

| 入口 | 功能 |
| --- | --- |
| `pepper_cloud.py` | 自由相机、点云和概率体素 NBV |
| `pepper_yolo.py` | 自由相机、YOLO 检测和概率体素 NBV |
| `pepper_tsdf.py` | 自由相机、YOLO 和 TSDF NBV |

这三个入口同样显示 v1 的整株语义模型，沿用各自原来的自由相机位置与控制。
后两者直接加载本目录 `yolo11/runs/detect/train3/weights/best.pt`，
不再依赖旧工程的绝对权重路径。权重未重训，模型替换不代表检测精度已验证。

## 目标区域与验证范围

v0 的规划仍针对原来的单个辣椒：旧 `pepper.stl` 和 `stem.stl`
仅作为不渲染的目标几何引用，用于包围盒、体素/TSDF 范围及原投影辅助逻辑。
它们与整株模型使用同一局部坐标，不会重复显示旧模型。
目标区域之外的植株参与真实渲染遮挡与机械臂碰撞检查，但不进入 v0
的信息增益体素范围；本次没有将算法扩展为多目标规划。

兼容检查同时修复了 TSDF 融合中 Panda/OpenCV 相机坐标列基不一致的问题，
增强检查确认已有非零融合权重；并修复了机械臂成功提示在 GBK 终端的编码异常。

2026-10-20 已通过四个入口的 RGB-D 渲染及一次 NBV 冒烟检查，机械臂入口
还检查了手眼往返变换、FK 同步和完整起终点间的离散关节路径。
已有手眼/可达性相关 14 项测试通过。这些检查不代表长时运行、重建质量或采摘效果验收。
验证产物位于仓库 `tmp/pepper_v0_validation/`。

独立核查已完成。另有本次修改前即存在的 TSDF 交互问题：其手动移动/转向
按键调用已注释的 `sync_main_cam()`，会报 `NameError`；本次未验收这些按键。
机械臂入口的 M/Q 控制独立于此逻辑。
