# DTest — A121 雷达精度测试工具

基于 [Acconeer Python Exploration](https://github.com/acconeer/acconeer-python-exploration) 二次开发，新增 DTest 精度测试面板，用于身高杆雷达芯片（A121 + XC120）的硬件精度验证。

## 安装运行

```cmd
git clone https://github.com/Zzhworkplace/height-rod-test.git
cd height-rod-test
pip install -e ".[app]"
python -m acconeer.exptool.app
```

> 前提：XC120 + XE121 开发套件已连接，WinUSB 驱动已装好（Zadig 工具安装）。

## 修改内容

| 文件 | 修改 |
|------|------|
| `src/acconeer/exptool/app/new/ui/height_test_tab.py` | 新增 DTest 精度测试面板 |
| `src/acconeer/exptool/app/new/ui/main_window.py` | 左侧栏 Stream 下方注册 "DTest" 按钮 |
| `src/acconeer/exptool/app/new/ui/icons.py` | 新增 DTest 图标定义 |

## DTest 面板功能

- **连接** XC120 → 自动校准 → **开始测量**
- 实时距离大字显示，滚动窗口 100 次统计
- **8 项统计**：均值、σ 稳定度、±0.2cm 命中率、极差、最小、最大、采样数、当前偏差
- **多点快照记录** + **导出数据**（UTF-8 文本）
- 默认配置：Profile 1 / step=1 (2.5mm) / sq=30 / 10–250cm

## ⚠️ 核心原则

- **纯读数工具**：直接读取 Distance Detector 原始输出，不添加任何滤波/平滑/去噪
- 数据 = 芯片真实水平，用于验证硬件精度

## 更新日志

### 2026-06-25

- 单位从 mm 改为 cm，所有统计阈值同步缩放
- 推送至自有仓库，支持同事下载测试

### 2026-06-24

- 新增 DTest 精度测试面板（`height_test_tab.py`）
- 左侧栏注册 DTest 按钮
- 连接→校准→测量→统计→快照→导出 完整流程
- Apple HIG 风格 UI（#007AFF 色板）
