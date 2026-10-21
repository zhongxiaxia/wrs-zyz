# pepper_nbv_v0 修改前恢复点

本备份直接保存在现有仓库 `zhongxiaxia/wrs-zyz` 的原目录结构中。
日期：2026-10-21；新 Gradient-NBV 规划器尚未实现。

- 项目：`0000_examples/pepper_nbv_v0/`，377 个原始文件。
- 共享依赖：v1 六个模块、辣椒模型、小车和手眼标定，共 14 个文件。
- `BACKUP_MANIFEST.json` 保存上述 391 个文件的大小与 SHA-256。
- `environment/` 保存 Python 和包版本信息。
- 原项目源码、数据和权重保持原样；包含原有缓存及已知问题。

固定恢复标签：`pepper-nbv-v0-pre-gradient-2026-10-21`。

## 恢复到新目录

在另一个目录运行 PowerShell，保留当前工作供对照：

```powershell
git -c core.autocrlf=false clone --branch pepper-nbv-v0-pre-gradient-2026-10-21 --single-branch https://github.com/zhongxiaxia/wrs-zyz.git wrs-zyz-restored
cd wrs-zyz-restored
python backups/pepper_nbv_v0/2026-10-21/verify_backup.py
```

校验器只读取文件。Git 不保存空目录；清单记录了空的输出目录，程序运行时可重建。
按环境清单配置 Python 后，在仓库根目录按项目 README 启动。

本次提交只纳入 v0、必要共享依赖与备份/版本管理说明，不纳入其他未提交工作。
