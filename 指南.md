# Minutes GPT / 会议纪要 GPT

会议纪要 GPT 能够快速地通过一段会议录音，生成一篇翔实完整的会议纪要。

本项目基于 python + pytorch 环境开发，使用了本地运行的 whisper-large-v3 与  openAI 的 ChatGPT。您可以在 [pytorch 官网](https://pytorch.org/get-started/locally/) 中查找并安装 pytorch

## quick start

将本仓库克隆到本地

```
git clone git@github.com:DavinciEvans/minutes-GPT.git
```

安装相关依赖

```
pip install -r requirements.txt
```

将 config.py 中的 api_key 修改为您的 OpenAI API Key，您可以在这里查找到您的 API Key：[https://platform.openai.com/account](https://platform.openai.com/account)

请确保你的 API Key 拥有可用的额度

准备好你的会议录音，在控制台运行 `MinutesGPT.py`

```
python MinutesGPT.py --audio /path/to/meeting_record.wav --language chinese
```

我们推荐使用 .wav 或是 .mp3 格式的录音文件。初次进行的语音转录内容会在本地的 `.asr_cache` 目录下进行缓存，之后的执行如果本地有缓存则会直接跳过语音转录，减少程序的执行时间。因此如果不满意生成地内容，依然可以快速地多次调用会议纪要 GPT

等待片刻后会在文件夹下生成 `output.md` 文件，即为生成的会议纪要，你也可以手动指定导出的路径地址

## Commands

```
python .\MinutesGPT.py --help

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO         The meeting record path. (Recommended .wav and .mp3)
  --output_file OUTPUT_FILE
                        The output file path. default output.md
  --language LANGUAGE   The language of the record, default english.
  --device DEVICE       The automatic speed recognition model running on.
  --batch_size BATCH_SIZE
                        The batch size of asr model. default 16.
  --clip_length CLIP_LENGTH
                        Length of each cut clip. default 6000.
  --clip_abandoned CLIP_ABANDONED
                        The length of the shortest clip, below which it will be discarded. default 1000.
  --word_level_timestamps WORD_LEVEL_TIMESTAMPS
                        Output word level timestamps. default false.
  --no_cache NO_CACHE   Results that are already cached without using ASR are also not generated. default false.
  --minutes_template MINUTES_TEMPLATE
                        Minutes template file path.
```

## TODO

- GUI 界面
- 实时转录
