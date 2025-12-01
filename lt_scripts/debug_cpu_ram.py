import os
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import io, json
import argparse
import subprocess
from functools import partial
import time
from memory_profiler import profile
import psutil

from tqdm import tqdm
import soundfile

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer


#@profile
def main(args):
    dictionary_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
    acoustic_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
    g2p_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"


    acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=None)

    # NOTE(longtou):
    phoneme_tokenizer = PhonemeTokenizer("espeak_ng_korean_word_sep")
    g2p_model.espnet_tokenizer = phoneme_tokenizer

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for audio_path in list(Path(args.indir).glob("*.wav")):
        uttid = audio_path.stem
        txt_path = audio_path.with_suffix(".lab")
        transcript = open(txt_path, 'r').read().strip()

        audio_format = "wav"
        audio_buf = io.BytesIO(open(audio_path, 'rb').read())

        audio_buf.seek(0)
        audio_buf.name = f"file.{audio_format}"
        with soundfile.SoundFile(audio_buf) as inf:
            frames = inf.frames
            sample_rate = inf.samplerate
            duration = frames / sample_rate
            num_channels = inf.channels

        # NOTE(longtou): custom option
        conf["beam"] = args.beam
        conf["retry_beam"] = args.retry_beam

        try:
            start = time.perf_counter()
            cpu_time = get_total_cpu_time()
            ret = align_one(
                audio_buf,
                audio_format,
                transcript,
                "json",
                acoustic_model,
                g2p_model,
                lexicon_compiler,
                tokenizer,
                conf
            )
            print(f"cpu time for align_one: {get_total_cpu_time() - cpu_time:.4f} s")
            end = time.perf_counter()
            print(f"{end - start}s elapsed.")
            RT = (end-start) / duration
            #print(f"dur: {duration}, {end-start} elapsed")
            print(f"RT: {(end-start) / duration}\n")
            #print(f"write to {outdir}/{uttid}.json")
            with open(f"{outdir}/{uttid}.json", 'w') as f:
                json.dump(ret, f, ensure_ascii=False)

        except AlignerError as e:
            print(f"{uttid} align fail")
            with open(f"{outdir}/{uttid}.txt", 'w') as f:
                f.write(transcript)
            with open(f"{outdir}/{uttid}.{audio_format}", 'wb') as f:
                f.write(audio_buf.getvalue())

# 1. 현재 파이썬 프로세스 객체 가져오기
pid = os.getpid()
current_process = psutil.Process(pid)

def get_total_cpu_time():
    """현재 프로세스의 누적된 사용자 시간 + 커널 시간(초)을 반환"""
    cpu_times = current_process.cpu_times()
    # 함수가 순수하게 CPU를 사용한 총 시간을 측정합니다.
    return cpu_times.user + cpu_times.system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="ssml_poc")
    #parser.add_argument("dictionary_path")
    #parser.add_argument("acoustic_model_path")
    #parser.add_argument("g2p_model_path")
    parser.add_argument("--outdir", default="outdir")
    parser.add_argument("--temporary_directory")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--retry_beam", type=int, default=40)
    args = parser.parse_args()

    # 2. 함수 호출 전 초기 CPU 시간 측정
    cpu_time = get_total_cpu_time()
    #print(f"cpu time before main: {cpu_time:.4f} s")
    #print("-" * 40)
    main(args)
    #cpu_time = get_total_cpu_time()
    #print(f"cpu time after main: {cpu_time:.4f} s")

