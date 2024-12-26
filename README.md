# Meeting Minutes Assistant

## 项目名称

Meeting_Minutes_Assistant

## 项目简介

Meeting Minutes Assistant: An AI-powered tool for quickly converting meeting recordings into meeting minutes.

会议纪要助手：一款 AI 驱动的工具，用于快速将会议录音转换为会议纪要。

> ***Relevant course***
> * Speech Recognition 2024 (2024年同济大学语音识别)

## 项目组成

* `/backend`
后端应用程序

* `/frontend`
前端应用程序

* `Documentation.md`
项目说明文档

* `meeting-minutes.md`
会议纪要

* `recording-example.wav`
会议录音示例

* `transcription-example.md`
会议转录示例

## 环境配置

```bash
source /etc/network_turbo
git clone https://github.com/MinmusLin/Meeting_Minutes_Assistant
cd Meeting_Minutes_Assistant/src
pip install -r requirements.txt
```

## 运行程序

```bash
python main.py --audio ../recording-example.wav --output_file ../meeting-minutes.md --language chinese
```

## 免责声明

The code and materials contained in this repository are intended for personal learning and research purposes only and may not be used for any commercial purposes. Other users who download or refer to the content of this repository must strictly adhere to the **principles of academic integrity** and must not use these materials for any form of homework submission or other actions that may violate academic honesty. I am not responsible for any direct or indirect consequences arising from the improper use of the contents of this repository. Please ensure that your actions comply with the regulations of your school or institution, as well as applicable laws and regulations, before using this content. If you have any questions, please contact me via [email](mailto:minmuslin@outlook.com).

本仓库包含的代码和资料仅用于个人学习和研究目的，不得用于任何商业用途。请其他用户在下载或参考本仓库内容时，严格遵守**学术诚信原则**，不得将这些资料用于任何形式的作业提交或其他可能违反学术诚信的行为。本人对因不恰当使用仓库内容导致的任何直接或间接后果不承担责任。请在使用前务必确保您的行为符合所在学校或机构的规定，以及适用的法律法规。如有任何问题，请通过[电子邮件](mailto:minmuslin@outlook.com)与我联系。

## 文档更新日期

2024年12月26日