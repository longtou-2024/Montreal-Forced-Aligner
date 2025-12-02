import io
from pathlib import Path
import json

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from endpoints.utils import limit_silence_duration

audio_path = "ssml_poc/sample.mp3"
audio_format = Path(audio_path).suffix.removeprefix('.')
audio_buf = io.BytesIO(open(audio_path, 'rb').read())
audio_buf.seek(0)
audio_buf.name = f"file.{audio_format}"
transcript = Path(audio_path).with_suffix('.lab').open('r').read().strip()

outdir = Path("outdir")
outdir.mkdir(parents=True, exist_ok=True)

# setup mfa
dictionary_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
acoustic_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
g2p_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"


acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=None)

# NOTE(longtou):
phoneme_tokenizer = PhonemeTokenizer("espeak_ng_korean_word_sep")
g2p_model.espnet_tokenizer = phoneme_tokenizer

mfa_result = align_one(
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

audio_buf.seek(0)
out_audio_buf, out_mfa_result = limit_silence_duration(audio_buf, audio_format, mfa_result, max_dur_s=0.5)
breakpoint()
with open(f"{outdir}/trim.{audio_format}", 'wb') as f:
    f.write(out_audio_buf.read())
with open(f"{outdir}/trim.json", 'w') as f:
    json.dump(out_mfa_result, f, ensure_ascii=False)
