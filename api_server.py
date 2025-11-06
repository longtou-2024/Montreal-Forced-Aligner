from typing import Union
import os
from pathlib import Path
import json
from typing import List, Any, Dict
import base64
import io # 메모리 내 파일 처리를 위한 라이브러리

from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
import soundfile

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

dictionary_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
acoustic_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
g2p_model_path="/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"

# 이 딕셔너리를 앱 전체에서 공유할 "상태"로 사용합니다.
# (예: 로드된 ML 모델, DB 커넥션 풀 등)
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ 서버 시작... 무거운 리소스를 로드합니다.")

    acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=None)
    # NOTE(longtou):
    phoneme_tokenizer = PhonemeTokenizer("espeak_ng_korean_word_sep")
    g2p_model.espnet_tokenizer = phoneme_tokenizer

    app_state["acoustic_model"] = acoustic_model
    app_state["g2p_model"] = g2p_model
    app_state["lexicon_compiler"] = lexicon_compiler
    app_state["tokenizer"] = tokenizer
    conf["beam"] = 10
    conf["retry_beam"] = 40
    app_state["conf"] = conf
    print("🎉 리소스 로드 완료! 서버가 요청을 받을 준비가 되었습니다.")

    yield # 서버 실행

    print("🧹 서버 종료... 리소스를 정리합니다.")
    app_state.clear()
    print("👋 정리 완료.")

#app = FastAPI()
# lifespan 관리자를 FastAPI 앱에 등록합니다.
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

class MFAIn(BaseModel):
    audio: str
    transcript: str

class MFAOut(BaseModel):
    mfa: Dict[str, Any]

@app.post("/mfa", response_model=MFAOut)
async def mfa(data: MFAIn):
    audio_format = "wav"
    transcript = data.transcript
    decoded_audio = base64.b64decode(data.audio)
    audio_buf = io.BytesIO(decoded_audio)
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
            app_state["acoustic_model"],
            app_state["g2p_model"],
            app_state["lexicon_compiler"],
            app_state["tokenizer"],
            app_state["conf"],
        )
    except AlignerError as e:
        ret = {}
        print(e)

    return {"mfa": ret}


if __name__ == "__main__":
    try:
        print("🚀 Uvicorn 서버를 시작합니다...")
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8002,
            log_level="info",
            reload=True  # 개발 환경에서 유용
        )
    except KeyboardInterrupt:
        print("\n👋 서버를 종료합니다.")
