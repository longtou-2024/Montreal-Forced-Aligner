from pathlib import Path
import io, json
import argparse
import subprocess
from functools import partial
import time

from tqdm import tqdm
import soundfile

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp",
                        default="align_sample/wav.scp")
    parser.add_argument("--text",
                        default="align_sample/text")
    #parser.add_argument("dictionary_path")
    #parser.add_argument("acoustic_model_path")
    #parser.add_argument("g2p_model_path")
    parser.add_argument("--outdir", default="outdir")
    parser.add_argument("--temporary_directory")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--retry_beam", type=int, default=40)
    args = parser.parse_args()

    dictionary_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
    acoustic_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
    g2p_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"


    acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=None)

    # NOTE(longtou):
    phoneme_tokenizer = PhonemeTokenizer("espeak_ng_korean_word_sep")
    g2p_model.espnet_tokenizer = phoneme_tokenizer

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    scp = open(args.scp, 'r').readlines()
    text = open(args.text, 'r').readlines()
    utt2scp = dict()
    utt2text = dict()
    for line in scp:
        uttid, audio_path = line.strip().split(' ', maxsplit=1)
        utt2scp[uttid] = audio_path
    for line in text:
        uttid, transcript = line.strip().split(' ', maxsplit=1)
        utt2text[uttid] = transcript

    for uttid in utt2scp:
        audio_format = "wav"
        audio_path = utt2scp[uttid]
        audio_buf = io.BytesIO(open(audio_path, 'rb').read())
        transcript = utt2text[uttid]

        audio_buf.seek(0)
        audio_buf.name = f"file.{audio_format}"
        with soundfile.SoundFile(audio_buf) as inf:
            frames = inf.frames
            sample_rate = inf.samplerate
            duration = frames / sample_rate
            num_channels = inf.channels
        #if duration < 3 or duration > 30:
        #    continue

        # NOTE(longtou): custom option
        conf["beam"] = args.beam
        conf["retry_beam"] = args.retry_beam

        try:
            start = time.perf_counter()
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
            end = time.perf_counter()
            print(f"{end - start}s elapsed.")
            print(f"write to {outdir}/{uttid}.json")
            with open(f"{outdir}/{uttid}.json", 'w') as f:
                json.dump(ret, f, ensure_ascii=False)

        except AlignerError as e:
            print(f"{uttid} align fail")
            with open(f"{outdir}/{uttid}.txt", 'w') as f:
                f.write(transcript)
            with open(f"{outdir}/{uttid}.{audio_format}", 'wb') as f:
                f.write(audio_buf.getvalue())
