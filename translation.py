import sys
import pyaudio
import wave
import numpy as np
import threading
import queue
import requests
import time
import os
import json
import whisper
import tempfile
from collections import deque
from pynput import keyboard
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QColorDialog,
                             QFontDialog, QGroupBox, QComboBox, QMessageBox,
                             QSlider, QMenu, QDialog, QCheckBox, QTextEdit, QLineEdit, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QCursor


class WhisperSpeechRecognizer(QThread):
    """Whisper语音识别线程 - 自动语言检测"""
    text_recognized = pyqtSignal(str, str)  # 文本, 检测到的语言
    status_updated = pyqtSignal(str)  # 状态更新
    volume_updated = pyqtSignal(int)  # 音量更新

    def __init__(self, device_index=1, model_size="base"):
        super().__init__()
        self.device_index = device_index
        self.model_size = model_size
        self._is_running = True

        # 音频参数 - 调整参数提高灵敏度
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.silence_threshold = 100  # 降低阈值，更容易触发

        # 录音参数 - 调整断句逻辑
        self.audio_buffer = []
        self.silence_frames = 0
        self.is_speaking = False
        self.silence_duration_threshold = 1.2  # 1.2秒静音断句
        self.min_speech_duration = 0.5  # 降低最小时长
        self.max_speech_duration = 8.0

        # 调试计数器
        self.debug_counter = 0

        # Whisper模型
        print(f"🔄 加载Whisper模型: {model_size}")
        self.model = whisper.load_model(model_size)
        print("✅ Whisper模型加载完成")

        # 音频队列
        self.audio_queue = queue.Queue()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频输入回调"""
        if self._is_running:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def run(self):
        """主识别循环"""
        print("🎤 Whisper语音识别启动 - 自动语言检测")
        self.status_updated.emit("状态: Whisper识别启动")

        if not self._setup_audio_stream():
            return

        self.status_updated.emit("状态: 🎤 监听中...")
        print("🔊 开始音频处理循环...")
        self._process_audio_stream()

    def _setup_audio_stream(self):
        """设置音频流"""
        try:
            self.audio = pyaudio.PyAudio()
            device_info = self.audio.get_device_info_by_index(self.device_index)
            print(f"🎧 使用音频设备: {device_info['name']}")

            self.stream = self.audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=self.device_index,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("✅ 音频流启动成功")
            print(f"📊 静音阈值: {self.silence_threshold}")
            return True

        except Exception as e:
            print(f"❌ 音频流设置失败: {e}")
            self.status_updated.emit(f"状态: 音频错误 - {str(e)}")
            return False

    def _process_audio_stream(self):
        """处理音频流数据"""
        print("🎯 开始语音检测...")

        while self._is_running:
            try:
                # 获取音频数据
                data = self.audio_queue.get(timeout=0.1)
                if not data:
                    continue

                audio_data = np.frombuffer(data, dtype=np.int16)

                # 计算音量
                if len(audio_data) > 0:
                    volume = np.mean(np.abs(audio_data))
                    if np.isnan(volume) or np.isinf(volume):
                        volume = 0
                else:
                    volume = 0

                self.volume_updated.emit(int(volume))

                # 调试输出（每50次输出一次）
                self.debug_counter += 1
                if self.debug_counter % 50 == 0:
                    speaking_status = "说话中" if self.is_speaking else "静音"
                    print(f"📊 音量: {volume:.1f}, 状态: {speaking_status}, 缓冲区: {len(self.audio_buffer)}帧")

                # 语音活动检测
                if volume > self.silence_threshold:
                    self.silence_frames = 0
                    if not self.is_speaking:
                        # 开始说话
                        self.is_speaking = True
                        self.audio_buffer = [data]
                        print(f"🎤 检测到语音开始！音量: {volume:.1f}")
                        self.status_updated.emit("状态: 🎤 检测到语音")
                    else:
                        # 持续说话
                        self.audio_buffer.append(data)
                else:
                    # 静音
                    self.silence_frames += 1
                    if self.is_speaking:
                        self.audio_buffer.append(data)  # 静音帧也收集

                # 计算音频时长
                if self.audio_buffer:
                    audio_duration = len(self.audio_buffer) * self.chunk_size / self.sample_rate
                else:
                    audio_duration = 0

                # 断句条件
                should_process = False
                reason = ""

                if self.is_speaking:
                    silence_duration = self.silence_frames * self.chunk_size / self.sample_rate

                    # 条件1: 静音断句
                    if (silence_duration >= self.silence_duration_threshold and
                            audio_duration >= self.min_speech_duration):
                        should_process = True
                        reason = f"静音断句 ({silence_duration:.1f}秒静音)"

                    # 条件2: 超长断句
                    elif audio_duration >= self.max_speech_duration:
                        should_process = True
                        reason = f"超长断句 ({audio_duration:.1f}秒)"

                    # 条件3: 短句快速处理
                    elif (audio_duration >= 1.0 and
                          silence_duration >= 0.8 and
                          volume < self.silence_threshold * 0.7):
                        should_process = True
                        reason = f"短句断句 ({audio_duration:.1f}秒)"

                # 处理语音段
                if should_process and self.audio_buffer:
                    print(f"🎯 {reason}, 音频时长: {audio_duration:.1f}秒")
                    audio_data = b''.join(self.audio_buffer)
                    self._process_speech(audio_data)

                    # 重置状态，保留少量上下文
                    keep_frames = int(0.3 * self.sample_rate / self.chunk_size)
                    if len(self.audio_buffer) > keep_frames:
                        self.audio_buffer = self.audio_buffer[-keep_frames:]
                    else:
                        self.audio_buffer = []

                    self.is_speaking = False
                    self.silence_frames = 0
                    print("🔄 重置语音检测状态")

            except queue.Empty:
                # 处理静音超时
                if self.is_speaking and self.audio_buffer:
                    audio_duration = len(self.audio_buffer) * self.chunk_size / self.sample_rate
                    if audio_duration >= self.min_speech_duration:
                        print(f"⏰ 队列超时，处理音频: {audio_duration:.1f}秒")
                        audio_data = b''.join(self.audio_buffer)
                        self._process_speech(audio_data)
                        self.audio_buffer = []
                        self.is_speaking = False
                continue

            except Exception as e:
                print(f"音频处理错误: {e}")
                continue

        self._cleanup()

    def _process_speech(self, audio_data):
        """处理语音识别"""
        try:
            # 保存临时音频文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            # 保存为WAV文件
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

            print("🔍 Whisper识别中...")
            self.status_updated.emit("状态: 🔍 识别中...")

            # 使用Whisper识别（自动检测语言）
            result = self.model.transcribe(
                temp_path,
                fp16=False,  # 使用FP32提高精度
                language=None  # 自动检测语言
            )

            text = result["text"].strip()
            detected_language = result.get("language", "unknown")

            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass

            if text and len(text) > 1:
                language_name = self._get_language_name(detected_language)
                print(f"✅ Whisper识别: [{language_name}] {text}")
                self.text_recognized.emit(text, detected_language)
                self.status_updated.emit(f"状态: ✅ 识别完成 ({language_name})")
            else:
                print("❌ 识别失败: 无有效文本")
                self.status_updated.emit("状态: ❌ 识别失败")

        except Exception as e:
            print(f"语音识别错误: {e}")
            self.status_updated.emit("状态: ❌ 识别错误")

    def _get_language_name(self, lang_code):
        """获取语言名称"""
        language_names = {
            "en": "英文",
            "zh": "中文",
            "ja": "日文",
            "ko": "韩文",
            "fr": "法文",
            "de": "德文",
            "es": "西班牙文",
            "ru": "俄文"
        }
        return language_names.get(lang_code, lang_code)

    def stop(self):
        """停止识别"""
        self._is_running = False

    def _cleanup(self):
        """清理资源"""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if hasattr(self, 'audio') and self.audio:
            try:
                self.audio.terminate()
            except:
                pass
        print("🛑 Whisper识别线程退出")


class TranslationWorker(QThread):
    """翻译工作线程"""
    translation_finished = pyqtSignal(str, str, str)  # original, translation, source_lang
    translation_failed = pyqtSignal(str, str)  # original, error

    def __init__(self, model_name="qwen2.5:3b"):
        super().__init__()
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.request_queue = queue.Queue()
        self._is_running = True

    def add_translation_task(self, text, source_language):
        """添加翻译任务"""
        if text and text.strip():
            self.request_queue.put((text, source_language))
            print(f"📨 添加翻译任务: [{source_language}] {text}")

    def run(self):
        """翻译处理循环"""
        print("🌐 翻译线程启动")
        while self._is_running:
            try:
                task = self.request_queue.get(timeout=1.0)
                if task:
                    text, source_language = task
                    self._process_translation(text, source_language)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"翻译线程错误: {e}")

    def _process_translation(self, text, source_language):
        """处理单个翻译任务"""
        print(f"🔄 开始翻译: [{source_language}] {text}")

        try:
            # 根据检测到的语言决定翻译方向
            if source_language == "zh":  # 中文 -> 英文
                prompt = f"Translate this Chinese to English: {text}"
            else:  # 其他语言（主要是英文）-> 中文
                prompt = f"Translate this to Chinese: {text}"

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            # 发送翻译请求
            start_time = time.time()
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                translation = result.get("response", "").strip()
                translation = self._clean_translation(translation)

                print(f"✅ 翻译完成 ({response_time:.0f}ms): {translation}")
                self.translation_finished.emit(text, translation, source_language)
            else:
                error_msg = f"HTTP错误: {response.status_code}"
                print(f"❌ {error_msg}")
                self.translation_failed.emit(text, error_msg)

        except requests.exceptions.Timeout:
            error_msg = "翻译超时"
            print(f"❌ {error_msg}")
            self.translation_failed.emit(text, error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "连接Ollama失败"
            print(f"❌ {error_msg}")
            self.translation_failed.emit(text, error_msg)
        except Exception as e:
            error_msg = f"翻译异常: {str(e)}"
            print(f"❌ {error_msg}")
            self.translation_failed.emit(text, error_msg)

    def _clean_translation(self, translation):
        """清理翻译结果"""
        remove_prefixes = [
            "以下英文翻译成中文：", "以下中文翻译成英文：",
            "翻译：", "Translation:", ":", "：",
            "Translate this English to Chinese:", "Translate this Chinese to English:",
            "Translate to Chinese:", "Translate to English:",
            "中文翻译：", "英文翻译：",
            "Here is the translation:", "The translation is:",
            "好的，", "Okay,", "嗯，", "Certainly,"
        ]

        # 移除前缀
        for prefix in remove_prefixes:
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()

        return translation.strip()

    def stop(self):
        """停止翻译线程"""
        self._is_running = False


class DraggableSubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化变量
        self.model_name = "qwen2.5:3b"
        self.is_recording = False
        self.audio_device_index = 1
        self.hide_ui = False
        self.text_opacity = 255
        self.background_opacity = 180
        self.whisper_model_size = "base"  # base, small, medium

        # 模型列表
        self.custom_models = ["qwen2.5:3b", "deepseek-r1:7b"]

        # 工作线程
        self.speech_recognizer = None
        self.translation_worker = None

        # 字幕数据
        self.previous_subtitle = {"original": "", "translation": "", "language": ""}
        self.current_subtitle = {"original": "", "translation": "", "language": ""}

        # UI设置
        self.font_size = 18
        self.bg_color = QColor(0, 0, 0, self.background_opacity)
        self.original_color = QColor(255, 255, 0, self.text_opacity)
        self.translation_color = QColor(0, 255, 255, self.text_opacity)

        # 初始化UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.init_ui()
        self.setup_keyboard_listener()

        # 启动翻译线程
        self._start_translation_worker()

    def _start_translation_worker(self):
        """启动翻译工作线程"""
        self.translation_worker = TranslationWorker(self.model_name)
        self.translation_worker.translation_finished.connect(self.on_translation_finished)
        self.translation_worker.translation_failed.connect(self.on_translation_failed)
        self.translation_worker.start()
        print("✅ 翻译线程启动")

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("实时双语字幕 - Whisper智能版")
        self.setGeometry(100, 100, 1000, 300)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 设置样式
        self.central_widget.setStyleSheet(
            f"background-color: rgba({self.bg_color.red()}, {self.bg_color.green()}, {self.bg_color.blue()}, {self.background_opacity}); border-radius: 10px;")

        layout = QVBoxLayout(self.central_widget)
        layout.setSpacing(5)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题栏
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(25)
        self.title_bar.setStyleSheet("background-color: rgba(50, 50, 50, 200); border-radius: 5px;")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)

        title_label = QLabel("🎯 实时双语字幕 - Whisper智能识别 • 拖动移动 • 右键菜单")
        title_label.setStyleSheet("color: white; font-size: 12px;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 最小化和关闭按钮
        self.min_btn = QPushButton("—")
        self.min_btn.setFixedSize(20, 20)
        self.min_btn.setStyleSheet(
            "QPushButton{background-color: #555; color: white; border: none; border-radius: 3px;} QPushButton:hover{background-color: #666;}")
        self.min_btn.clicked.connect(self.showMinimized)
        title_layout.addWidget(self.min_btn)

        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet(
            "QPushButton{background-color: #d00; color: white; border: none; border-radius: 3px;} QPushButton:hover{background-color: #f00;}")
        self.close_btn.clicked.connect(self.close)
        title_layout.addWidget(self.close_btn)

        # 字幕显示区域
        self.previous_original_label = QLabel("")
        self.previous_original_label.setAlignment(Qt.AlignCenter)
        self.previous_original_label.setWordWrap(True)

        self.previous_translation_label = QLabel("")
        self.previous_translation_label.setAlignment(Qt.AlignCenter)
        self.previous_translation_label.setWordWrap(True)

        self.separator = QWidget()
        self.separator.setFixedHeight(1)
        self.separator.setStyleSheet("background-color: rgba(255, 255, 255, 100);")

        self.current_original_label = QLabel("准备就绪 - 按F2开始翻译")
        self.current_original_label.setAlignment(Qt.AlignCenter)
        self.current_original_label.setWordWrap(True)

        self.current_translation_label = QLabel("Ready - Press F2 to start")
        self.current_translation_label.setAlignment(Qt.AlignCenter)
        self.current_translation_label.setWordWrap(True)

        # 语言显示标签
        self.language_label = QLabel("")
        self.language_label.setAlignment(Qt.AlignCenter)
        self.language_label.setStyleSheet(
            "color: #FFA500; background-color: rgba(50, 50, 50, 200); padding: 2px; border-radius: 3px;")

        # 控制面板
        self.control_group = QGroupBox()
        self.control_group.setStyleSheet(
            "QGroupBox{color: white; border: 1px solid rgba(255,255,255,100); border-radius: 5px; margin-top: 10px;} QGroupBox::title{subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px;}")
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始翻译 (F2)")
        self.start_btn.clicked.connect(self.toggle_recording)
        self.start_btn.setStyleSheet(
            "QPushButton{background-color: rgba(0, 100, 0, 200); color: white; border: none; padding: 8px 15px; border-radius: 3px;} QPushButton:hover{background-color: rgba(0, 150, 0, 200);}")

        self.settings_btn = QPushButton("音频设备")
        self.settings_btn.clicked.connect(self.show_device_dialog)
        self.settings_btn.setStyleSheet(
            "QPushButton{background-color: rgba(100, 100, 100, 200); color: white; border: none; padding: 8px 15px; border-radius: 3px;} QPushButton:hover{background-color: rgba(150, 150, 150, 200);}")

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.settings_btn)
        control_layout.addStretch()
        self.control_group.setLayout(control_layout)

        # 状态显示
        self.status_label = QLabel("状态: 等待开始")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color: white; background-color: rgba(50, 50, 50, 200); padding: 3px; border-radius: 3px;")

        # 音量显示
        self.volume_label = QLabel("音量: 0")
        self.volume_label.setAlignment(Qt.AlignCenter)
        self.volume_label.setStyleSheet(
            "color: lime; background-color: rgba(0, 50, 0, 200); padding: 2px; border-radius: 3px;")

        # 添加到布局
        layout.addWidget(self.title_bar)
        layout.addWidget(self.previous_original_label)
        layout.addWidget(self.previous_translation_label)
        layout.addWidget(self.separator)
        layout.addWidget(self.current_original_label)
        layout.addWidget(self.current_translation_label)
        layout.addWidget(self.language_label)
        layout.addStretch()
        layout.addWidget(self.control_group)
        layout.addWidget(self.volume_label)
        layout.addWidget(self.status_label)

        self.apply_fonts()
        self.update_text_style()

    def on_speech_recognized(self, text, detected_language):
        """处理识别到的语音文本"""
        print(f"🎯 收到识别文本: [{detected_language}] {text}")

        # 更新语言显示
        language_name = self._get_language_name(detected_language)
        self.language_label.setText(f"检测语言: {language_name}")

        # 第一步：立即更新界面显示识别的文本
        if self.current_subtitle["original"]:
            self.previous_subtitle = self.current_subtitle.copy()

        self.current_subtitle = {
            "original": text,
            "translation": "🔄 翻译中...",
            "language": detected_language
        }
        self.update_display()

        # 第二步：发送翻译任务（包含源语言信息）
        if self.translation_worker:
            self.translation_worker.add_translation_task(text, detected_language)

    def on_translation_finished(self, original_text, translated_text, source_language):
        """翻译完成回调"""
        if self.current_subtitle["original"] == original_text:
            self.current_subtitle["translation"] = translated_text
            self.update_display()
            print(f"✅ 翻译结果显示完成: {translated_text}")

    def on_translation_failed(self, original_text, error_msg):
        """翻译失败回调"""
        if self.current_subtitle["original"] == original_text:
            self.current_subtitle["translation"] = f"❌ 翻译失败"
            self.update_display()
            print(f"❌ 翻译失败: {error_msg}")

    def _get_language_name(self, lang_code):
        """获取语言名称"""
        language_names = {
            "en": "English",
            "zh": "中文",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "unknown": "未知"
        }
        return language_names.get(lang_code, lang_code)

    def toggle_recording(self):
        """切换录音状态"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """开始语音识别"""
        # 检查Ollama服务
        if not self.check_ollama_availability():
            QMessageBox.warning(self, "警告", "无法连接到Ollama服务，请确保Ollama正在运行！")
            return

        self.is_recording = True
        self.start_btn.setText("停止翻译 (F2)")
        self.start_btn.setStyleSheet(
            "QPushButton{background-color: rgba(200, 0, 0, 200); color: white; border: none; padding: 8px 15px; border-radius: 3px;} QPushButton:hover{background-color: rgba(255, 0, 0, 200);}")
        self.status_label.setText("状态: 启动Whisper识别...")

        # 启动Whisper语音识别线程
        self.speech_recognizer = WhisperSpeechRecognizer(
            device_index=self.audio_device_index,
            model_size=self.whisper_model_size
        )
        self.speech_recognizer.text_recognized.connect(self.on_speech_recognized)
        self.speech_recognizer.status_updated.connect(self.status_label.setText)
        self.speech_recognizer.volume_updated.connect(self.on_volume_updated)
        self.speech_recognizer.start()

        print("🎤 开始Whisper智能语音识别")

    def stop_recording(self):
        """停止语音识别"""
        self.is_recording = False
        self.start_btn.setText("开始翻译 (F2)")
        self.start_btn.setStyleSheet(
            "QPushButton{background-color: rgba(0, 100, 0, 200); color: white; border: none; padding: 8px 15px; border-radius: 3px;} QPushButton:hover{background-color: rgba(0, 150, 0, 200);}")
        self.status_label.setText("状态: 已停止")
        self.volume_label.setText("音量: 0")
        self.language_label.setText("")

        # 停止语音识别
        if self.speech_recognizer:
            self.speech_recognizer.stop()
            self.speech_recognizer.wait(3000)

        print("🛑 语音识别停止")

    def on_volume_updated(self, volume):
        """更新音量显示"""
        self.volume_label.setText(f"音量: {volume}")

    def check_ollama_availability(self):
        """检查Ollama服务是否可用"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama服务连接成功")
                return True
            else:
                print("❌ 无法连接到Ollama服务")
                return False
        except Exception as e:
            print(f"❌ Ollama连接失败: {e}")
            return False

    def update_display(self):
        """更新界面显示"""
        self.previous_original_label.setText(self.previous_subtitle["original"])
        self.previous_translation_label.setText(self.previous_subtitle["translation"])
        self.current_original_label.setText(self.current_subtitle["original"])
        self.current_translation_label.setText(self.current_subtitle["translation"])

    def show_device_dialog(self):
        """显示音频设备设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("音频设备设置")
        dialog.setFixedSize(400, 200)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("选择音频输入设备:"))

        device_combo = QComboBox()
        try:
            audio = pyaudio.PyAudio()
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    device_name = device_info['name']
                    device_combo.addItem(f"{i}: {device_name}", i)
            audio.terminate()
        except Exception as e:
            print(f"获取音频设备失败: {e}")

        # 选择立体声混音设备
        for i in range(device_combo.count()):
            if "stereo" in device_combo.itemText(i).lower() or "混音" in device_combo.itemText(i):
                device_combo.setCurrentIndex(i)
                self.audio_device_index = device_combo.itemData(i)
                break

        layout.addWidget(device_combo)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(lambda: self.on_device_selected(device_combo.currentData(), dialog))
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def on_device_selected(self, device_index, dialog):
        """设备选择确认"""
        self.audio_device_index = device_index
        print(f"选择音频设备: {device_index}")
        dialog.accept()

    def set_whisper_model(self, model_size):
        """设置Whisper模型大小"""
        self.whisper_model_size = model_size
        print(f"切换Whisper模型: {model_size}")

    def update_background_style(self):
        """更新背景样式"""
        self.central_widget.setStyleSheet(
            f"background-color: rgba({self.bg_color.red()}, {self.bg_color.green()}, {self.bg_color.blue()}, {self.background_opacity}); border-radius: 10px;")

    def update_text_style(self):
        """更新文字样式"""
        original_style = f"color: rgba({self.original_color.red()}, {self.original_color.green()}, {self.original_color.blue()}, {self.text_opacity}); background-color: transparent;"
        translation_style = f"color: rgba({self.translation_color.red()}, {self.translation_color.green()}, {self.translation_color.blue()}, {self.text_opacity}); background-color: transparent;"

        self.current_original_label.setStyleSheet(original_style)
        self.previous_original_label.setStyleSheet(original_style)
        self.current_translation_label.setStyleSheet(translation_style)
        self.previous_translation_label.setStyleSheet(translation_style)

    def apply_fonts(self):
        """应用字体设置"""
        previous_font = QFont("Microsoft YaHei", self.font_size - 3)
        current_font = QFont("Microsoft YaHei", self.font_size, QFont.Bold)

        self.previous_original_label.setFont(previous_font)
        self.previous_translation_label.setFont(previous_font)
        self.current_original_label.setFont(current_font)
        self.current_translation_label.setFont(QFont("Microsoft YaHei", self.font_size))

    def toggle_ui_visibility(self):
        """切换UI可见性"""
        self.hide_ui = not self.hide_ui

        if self.hide_ui:
            # 隐藏UI元素
            self.title_bar.hide()
            self.control_group.hide()
            self.volume_label.hide()
            self.status_label.hide()
            self.separator.hide()
            self.language_label.hide()
            # 调整窗口大小
            self.resize(self.width(), 150)
            # 调整边距
            self.central_widget.layout().setContentsMargins(10, 10, 10, 10)
        else:
            # 显示UI元素
            self.title_bar.show()
            self.control_group.show()
            self.volume_label.show()
            self.status_label.show()
            self.separator.show()
            self.language_label.show()
            # 恢复窗口大小
            self.resize(self.width(), 300)
            # 恢复边距
            self.central_widget.layout().setContentsMargins(15, 15, 15, 15)

    def set_background_opacity(self, opacity):
        """设置背景透明度"""
        self.background_opacity = int(opacity * 2.55)
        self.bg_color.setAlpha(self.background_opacity)
        self.update_background_style()

    def set_text_opacity(self, opacity):
        """设置文字透明度"""
        self.text_opacity = int(opacity * 2.55)
        self.original_color.setAlpha(self.text_opacity)
        self.translation_color.setAlpha(self.text_opacity)
        self.update_text_style()

    def set_font_size(self, size):
        """设置字体大小"""
        self.font_size = size
        self.apply_fonts()

    def set_model(self, model_name):
        """设置AI模型"""
        self.model_name = model_name
        # 重启翻译线程
        if self.translation_worker:
            self.translation_worker.stop()
            self.translation_worker.wait(2000)
        self._start_translation_worker()
        print(f"切换模型: {model_name}")

    def add_custom_model(self):
        """添加自定义模型"""
        model_name, ok = QInputDialog.getText(self, "添加模型", "请输入模型名称:")
        if ok and model_name.strip():
            if model_name not in self.custom_models:
                self.custom_models.append(model_name)
                print(f"添加模型: {model_name}")
            else:
                QMessageBox.warning(self, "提示", "该模型已存在!")

    def remove_current_model(self):
        """删除当前模型"""
        if len(self.custom_models) <= 1:
            QMessageBox.warning(self, "提示", "至少需要保留一个模型!")
            return

        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要删除模型 '{self.model_name}' 吗?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.custom_models.remove(self.model_name)
            # 切换到第一个模型
            self.set_model(self.custom_models[0])
            print(f"删除模型: {self.model_name}")

    def show_color_settings(self):
        """显示颜色设置对话框"""
        bg_color = QColorDialog.getColor(self.bg_color, self, "选择背景颜色")
        if bg_color.isValid():
            self.bg_color = bg_color
            self.bg_color.setAlpha(self.background_opacity)
            self.update_background_style()

        original_color = QColorDialog.getColor(self.original_color, self, "选择原文颜色")
        if original_color.isValid():
            self.original_color = original_color
            self.original_color.setAlpha(self.text_opacity)
            self.update_text_style()

        translation_color = QColorDialog.getColor(self.translation_color, self, "选择翻译颜色")
        if translation_color.isValid():
            self.translation_color = translation_color
            self.translation_color.setAlpha(self.text_opacity)
            self.update_text_style()

    def contextMenuEvent(self, event):
        """右键菜单事件"""
        context_menu = QMenu(self)

        # Whisper模型选择
        whisper_menu = context_menu.addMenu("Whisper模型")
        model_sizes = [
            ("base (推荐)", "base"),
            ("small (快速)", "small"),
            ("medium (高精度)", "medium")
        ]

        for name, size in model_sizes:
            action = whisper_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(size == self.whisper_model_size)
            action.triggered.connect(lambda checked, s=size: self.set_whisper_model(s))

        # 透明度设置菜单
        opacity_menu = context_menu.addMenu("背景透明度")
        current_bg_opacity = int(self.background_opacity / 2.55)  # 转换为百分比
        for opacity in [100, 80, 60, 40, 20]:
            action = opacity_menu.addAction(f"{opacity}%")
            action.setCheckable(True)
            action.setChecked(opacity == current_bg_opacity)
            action.triggered.connect(lambda checked, o=opacity: self.set_background_opacity(o))

        text_opacity_menu = context_menu.addMenu("文字透明度")
        current_text_opacity = int(self.text_opacity / 2.55)  # 转换为百分比
        for opacity in [100, 80, 60, 40, 20]:
            action = text_opacity_menu.addAction(f"{opacity}%")
            action.setCheckable(True)
            action.setChecked(opacity == current_text_opacity)
            action.triggered.connect(lambda checked, o=opacity: self.set_text_opacity(o))

        font_menu = context_menu.addMenu("字体大小")
        for size in [14, 16, 18, 20, 24]:
            action = font_menu.addAction(f"{size}px")
            action.triggered.connect(lambda checked, s=size: self.set_font_size(s))

        context_menu.addSeparator()

        # AI模型选择
        model_menu = context_menu.addMenu("AI模型")

        # 添加现有模型
        for model in self.custom_models:
            action = model_menu.addAction(model)
            action.setCheckable(True)
            action.setChecked(model == self.model_name)
            action.triggered.connect(lambda checked, m=model: self.set_model(m))

        model_menu.addSeparator()

        # 添加自定义模型选项
        add_model_action = model_menu.addAction("➕ 添加自定义模型")
        add_model_action.triggered.connect(self.add_custom_model)

        remove_model_action = model_menu.addAction("🗑️ 删除当前模型")
        remove_model_action.triggered.connect(self.remove_current_model)

        # 隐藏UI选项
        hide_ui_action = context_menu.addAction("隐藏UI控件")
        hide_ui_action.setCheckable(True)
        hide_ui_action.setChecked(self.hide_ui)
        hide_ui_action.triggered.connect(self.toggle_ui_visibility)

        context_menu.addSeparator()

        color_action = context_menu.addAction("颜色设置")
        color_action.triggered.connect(self.show_color_settings)

        context_menu.addSeparator()

        exit_action = context_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        context_menu.exec_(event.globalPos())

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_start_position'):
            self.move(event.globalPos() - self.drag_start_position)
            event.accept()

    def setup_keyboard_listener(self):
        """设置键盘监听"""

        def on_press(key):
            try:
                if key == keyboard.Key.f2:
                    self.toggle_recording()
                elif key == keyboard.Key.esc:
                    self.close()
            except:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

    def closeEvent(self, event):
        """程序关闭事件"""
        self.stop_recording()

        # 停止翻译线程
        if self.translation_worker:
            self.translation_worker.stop()
            self.translation_worker.wait(3000)

        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DraggableSubtitleWindow()
    window.show()

    print("=" * 60)
    print("🎯 双语字幕启动 - Whisper智能版本!")
    print("📝 执行顺序: Whisper识别 → 自动语言检测 → 智能翻译 → 显示译文")
    print("🌐 语音识别: 完全离线 | 自动检测中英文 | 翻译: 需要Ollama")
    print("🎹 快捷键: F2开始/停止翻译 | ESC退出程序")
    print("=" * 60)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
