# Python-translation-cn-en
中英直播实时翻译脚本/Real time translation script for Chinese English live streaming


🎯 实时双语字幕翻译工具

一个基于 Whisper 语音识别和 Ollama 大模型的实时双语字幕翻译工具，支持智能语言检测和实时翻译显示。

✨ 核心功能

🎤 智能语音识别
Whisper 离线识别：完全本地化语音识别，无需网络

自动语言检测：智能识别中英文及其他多种语言

实时流式处理：边说边识别，低延迟响应


🔄 智能翻译

Ollama 大模型：基于本地部署的 AI 大模型翻译

正确翻译方向：自动识别语言并选择正确翻译方向

中文 → 英文

英文 → 中文

高质量翻译：保持语义准确性和流畅性


🖥️ 优雅界面

悬浮字幕：始终置顶，不影响其他操作

透明背景：可调节透明度，视觉友好

实时显示：原文和译文分开展示

操作便捷：支持拖拽、右键菜单、快捷键


⚙️ 丰富设置

多模型支持：可切换不同 Whisper 和 Ollama 模型

音频设备选择：支持立体声混音等输入设备

自定义样式：字体、颜色、透明度个性化设置

快捷键操作：F2 开始/停止，ESC 退出


🛠️ 环境要求

系统要求
操作系统：Windows 10/11, macOS, Linux

Python 版本：3.7 或更高版本

内存：建议 8GB 或以上

存储空间：至少 5GB 可用空间

必备软件
Python 3.7+：

Ollama：https://ollama.com/

FFmpeg（Whisper 依赖）：https://ffmpeg.org/



📦安装步骤

1. 安装 Python 依赖库

【bash】

pip install openai-whisper pyaudio PyQt5 pynput requests numpy

2. 安装 Ollama
   
（1）访问 Ollama官网 下载安装包

（2）运行安装程序

（3）打开命令提示符，拉取模型：

【bash】

ollama pull qwen2.5:3b               【小技巧：若运行到最后，下载速度只有几KB/S，可以通过快速禁用网卡，再打开网卡，使下载速度提高】

3.安装 FFmpeg（Whisper 依赖）

（1）下载 FFmpeg：  https://ffmpeg.org/download.html

（2）解压到 C:\ffmpeg\

（3）添加 C:\ffmpeg\bin 到系统 PATH 环境变量

🚀 使用指南

1.启动 Ollama 服务

【bash】

ollama serve

2.运行主程序

【bash】

python translation.py

3.音频设备设置

点击"音频设备"按钮

选择"立体声混音"或您的麦克风设备

4.可以通过代码中参数设置，使声音采集率提高

代码片段【默认是在第40行代码】：self.silence_threshold = 40  # 降低阈值，更容易触发

点击"确定"

