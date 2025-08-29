from pathlib import Path
import io, json
import argparse
import subprocess
from functools import partial

from tqdm import tqdm
import webdataset as wds
import soundfile

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError


def gcp_cp(fname, gcs_url="gs://prod-ai-lab-speech-bucket/longtou/tmp"):
    dirname = str(Path(fname).parent)
    subprocess.run(f"gcloud storage cp -R {dirname} {gcs_url}", shell=True)
    subprocess.run(f"rm {fname}", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_url")
    parser.add_argument("dictionary_path")
    parser.add_argument("acoustic_model_path")
    parser.add_argument("g2p_model_path")
    parser.add_argument("outdir")
    parser.add_argument("gcs_url")
    parser.add_argument("json_text_key")
    parser.add_argument("--temporary_directory")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--retry_beam", type=int, default=40)
    args = parser.parse_args()

    #shard_url = "/home/longtou.2024/mount/longtou/db/literature/wds_v2/shard-000000.tar"
    #dictionary_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
    #acoustic_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
    #g2p_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"

    dataset = wds.WebDataset(args.shard_url)
    acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(args.dictionary_path, args.acoustic_model_path, args.g2p_model_path, args.temporary_directory)

    outdir = Path("outdir")
    outdir.mkdir(exist_ok=True)
    shard_name = Path(args.shard_url).stem
    #writer = wds.ShardWriter(f"{outdir}/{shard_name}.tar", maxsize=10e9, post=partial(gcp_cp, gcs_url=args.gcs_url))
    tar_fname = f"{outdir}/{shard_name}.tar"
    writer = wds.TarWriter(tar_fname)
    idx = int(shard_name.split('-')[1])
    f_log_name = f"{str(outdir)}/align_fail{idx}.log"
    f_log = open(f_log_name, 'w')

    cnt = 0
    for sample in tqdm(dataset):
        #cnt += 1
        #if cnt > 50:
        #    break
        uttid = sample["__key__"]
        json_data = json.load(io.BytesIO(sample["json"]))
        audio_format = "wav"
        audio_buf = io.BytesIO(sample[audio_format])
        transcript = json_data
        for text_key in args.json_text_key.split(','):
            transcript = transcript[text_key]
        transcript = transcript.strip()

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

        # add mfa result
        json_data["mfa"] = ret
        example = {
            "__key__": uttid,
            "json": json.dumps(json_data, ensure_ascii=False),
            audio_format: audio_buf.getvalue(),
        }
        writer.write(example)
    writer.close()

    # write to bucket
    f_log.flush()
    f_log.close()
    subprocess.run(f"gcloud storage cp {tar_fname} {args.gcs_url}/", shell=True)
    subprocess.run(f"gcloud storage cp {f_log_name} {args.gcs_url}/", shell=True)
    #subprocess.run(f"rm {tar_fname}", shell=True)
