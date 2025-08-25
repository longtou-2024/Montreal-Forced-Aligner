from pathlib import Path
import io, json
import argparse

from tqdm import tqdm
import webdataset as wds
import soundfile

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_url")
    parser.add_argument("dictionary_path")
    parser.add_argument("acoustic_model_path")
    parser.add_argument("g2p_model_path")
    parser.add_argument("outdir")
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
    #writer = wds.ShardWriter(f"{outdir}/shard-%06d.tar", maxsize=1e9)

    for sample in tqdm(dataset):
        uttid = sample["__key__"]
        json_data = json.load(io.BytesIO(sample["json"]))
        audio_format = "wav"
        audio_buf = io.BytesIO(sample[audio_format])
        transcript = json_data["transcript"].strip()

        audio_buf.seek(0)
        audio_buf.name = f"file.{audio_format}"
        with soundfile.SoundFile(audio_buf) as inf:
            frames = inf.frames
            sample_rate = inf.samplerate
            duration = frames / sample_rate
            num_channels = inf.channels
        if duration < 3 or duration > 30:
            continue

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
            print(f"{uttid} align fail")
            with open(f"{args.outdir}/{uttid}.json", 'w') as f:
                json.dump(json_data, f, ensure_ascii=False)
            with open(f"{args.outdir}/{uttid}.{audio_format}", 'wb') as f:
                f.write(audio_buf.getvalue())
            #example = {
            #    "__key__": uttid,
            #    "json": json.dumps(json_data, ensure_ascii=False),
            #    audio_format: audio_buf.getvalue(),
            #}
            #writer.write(example)

