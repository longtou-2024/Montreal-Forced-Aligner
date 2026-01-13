from pathlib import Path
import io, json
import os
os.environ['OMP_NUM_THREADS'] = "1"

import webdataset as wds
from tqdm import tqdm
import soundfile

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer


dictionary_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
acoustic_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
g2p_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"

shard_url = "/home/longtou.2024/mount/longtou/db/literature/wds_v3/shard-000000.tar"
json_text_key = "transcript"

def main():
    acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=None)
    phoneme_tokenizer = PhonemeTokenizer("espeak_ng_korean_word_sep")
    g2p_model.espnet_tokenizer = phoneme_tokenizer
    conf['beam'] = 10
    conf['retry_beam'] = 40

    dataset = wds.WebDataset(shard_url)

    f_log = open("log.txt", 'w')
    cnt = 0
    mfa_results = []
    for sample in tqdm(dataset):
        cnt += 1
        if cnt > 100: break

        uttid = sample["__key__"]
        json_data = json.load(io.BytesIO(sample["json"]))
        audio_format = "wav" if "wav" in sample else "mp3"
        audio_buf = io.BytesIO(sample[audio_format])
        transcript = json_data
        for text_key in json_text_key.split(','):
            transcript = transcript[text_key]
        transcript = transcript.strip()

        audio_buf.seek(0)
        audio_buf.name = f"file.{audio_format}"
        with soundfile.SoundFile(audio_buf) as inf:
            frames = inf.frames
            sample_rate = inf.samplerate
            duration = frames / sample_rate
            num_channels = inf.channels

        try:
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
        except AlignerError as e:
            f_log.write(f"{uttid} AlignerError\n")
            ret = {}
        except Exception as e:
            f_log.write(f"{uttid} Exception\n")
            ret = {}

        #breakpoint()
        mfa_results.append(ret)

    # NOTE(longtou): analyze result
    n_null = 0
    for result in mfa_results:
        if len(result) == 0:
            n_null += 1
            continue
        phn_entries = result['tiers']['phones']['entries']
        phns = [item[2] for item in phn_entries]
        if 'spn' in phns:
            print("spn reported!")

    print(f"n_null: {n_null}/{len(mfa_results)}")
    print("All Done")


if __name__ == '__main__':
    main()
