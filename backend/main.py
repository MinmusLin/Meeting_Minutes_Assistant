from ASR import WhisperASR
from GPT import ChatGPT
from Role import MeetingSecretary, SummaryWriter, MeetingMinutesEditor
from config import Config
import argparse
import os

config = Config()
client = ChatGPT(api_key=config.api_key)


class NoAudioError:
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--audio', help='The meeting record path. (Recommended .wav and .mp3)')
parser.add_argument('--output_file', help='The output file path. default output.md')
parser.add_argument('--language', help='The language of the record, default english.')
parser.add_argument('--device', help='The automatic speed recognition model running on.')
parser.add_argument('--batch_size', help='The batch size of asr model. default 16.')
parser.add_argument('--clip_length', help='Length of each cut clip. default 6000.')
parser.add_argument('--clip_abandoned', help='The length of the shortest clip, below which it will be discarded. default 1000.')
parser.add_argument('--word_level_timestamps', help='Output word level timestamps. default false.')
parser.add_argument('--no_cache', help='Results that are already cached without using ASR are also not generated. default false.')
parser.add_argument('--minutes_template', help='Minutes template file path.')
args = parser.parse_args()

if args.audio is None:
    raise (NoAudioError('You must provide an audio file.'))

audio = args.audio
language = args.language
output_file = args.output_file if args.output_file else 'output.md'
batch_size = int(args.batch_size) if args.batch_size else None
device = args.device
clip_length = int(args.clip_length) if args.clip_length else 6000
clip_abandoned = int(args.clip_abandoned) if args.clip_abandoned else 1000
word_level_timestamps = bool(args.word_level_timestamps)
no_cache = bool(args.no_cache)
minutes_template = args.minutes_template


def write_output(text):
    print('Start to write the meeting minutes.')
    meeting_secretary = MeetingSecretary(client, minutes_template)
    return meeting_secretary.request(text)


def write_summary(meeting_text):
    clip_count = len(meeting_text) // clip_length + 1 if len(meeting_text) % clip_length >= clip_abandoned else len(meeting_text) // clip_length
    clips = [meeting_text[(clip_length * i): (clip_length * (i + 1)) if len(meeting_text) > (clip_length * (i + 1) - 1) else len(meeting_text)] for i in range(clip_count)]
    print(f'There are {len(meeting_text)} letter in this paragraph, and it will be divided into {clip_count} short sections.')
    summary = ''
    summary_writer = SummaryWriter(client)
    for i in range(len(clips)):
        summary += f'## the f{i + 1} segment: \n{summary_writer.request(clips[i])}\n'
        print(f'Finish the {i + 1} segment summary.')
    return summary


def check_output(summary: str, output: str):
    meeting_minutes_editor = MeetingMinutesEditor(client, minutes_template)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(meeting_minutes_editor.request(f'## 原始会议内容 \n {summary} \n ## 会议纪要 \n {output}'))


def asr():
    if os.path.exists(f'./.asr_cache/{os.path.basename(audio)}_cache.txt'):
        print(f'Use {audio} ASR cache.')
        with open(f'./.asr_cache/{os.path.basename(audio)}_cache.txt', 'r', encoding='utf-8') as f:
            return f.read()
    asr = WhisperASR(language=language, batch_size=batch_size, device=device, word_level_timestamps=word_level_timestamps)
    print(f'using {asr.device}. Start the ASR task.')
    meeting_text = ','.join(list(map(lambda x: x['text'], asr(audio))))
    if not no_cache:
        with open(f'.asr_cache/{os.path.basename(audio)}_cache.txt', 'w', encoding='utf-8') as f:
            f.write(meeting_text)
    return meeting_text


if __name__ == '__main__':
    meeting_text = asr()
    summary = write_summary(meeting_text)
    output = write_output(summary)
    check_output(summary, output)
