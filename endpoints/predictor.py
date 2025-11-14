from typing import Any
import base64
import io

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils


class MFAPredictor(Predictor):
    def __init__(self):
        super().__init__()

    def load(self, artifacts_uri: str) -> None:
        # NOTE(longtou): copy files 
        # from "/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/"
        # from "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/mfa/espeak/"
        # to "." (working dir)
        prediction_utils.download_model_artifacts(artifacts_uri)

        # setup
        dictionary_path = "korean_espeak.dict"
        acoustic_model_path = "acoustic/korean_espeak.zip"
        g2p_model_path = "g2p/korean_espeak.zip"
        x = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=None)
        self._acoustic_model = x[0]
        self._g2p_model = x[1]
        self._lexicon_compiler = x[2]
        self._tokenizer = x[3]
        self._conf = x[4]

        phoneme_tokenizer = PhonemeTokenizer("espeak_ng_korean_word_sep")
        self._g2p_model.espnet_tokenizer = phoneme_tokenizer
        self._conf["beam"] = 10
        self._conf["retry_beam"] = 40

    def preprocess(self, prediction_input: dict) -> dict:
        instances = prediction_input["instances"]

        audio_format = "wav"
        transcript = instances["transcript"]
        audio = instances["audio"]
        decoded_audio = base64.b64decode(audio)
        audio_buf = io.BytesIO(decoded_audio)
        audio_buf.seek(0)
        audio_buf.name = f"file.{audio_format}"

        result = {
            "transcript": transcript,
            "audio_buf": audio_buf,
            "audio_format": audio_format,
        }
        return result


    def predict(self, instances: dict) -> dict:
        transcript = instances["transcript"]
        audio_buf = instances["audio_buf"]
        audio_format = instances["audio_format"]
        try:
            result = align_one(
                audio_buf,
                audio_format,
                transcript,
                "json",
                self._acoustic_model,
                self._g2p_model,
                self._lexicon_compiler,
                self._tokenizer,
                self._conf
            )
        #except AlignerError as e:
        except Exception as e:
            error_msg = str(e)
            result = {"error": error_msg}

        return result


    def postprocess(self, prediction_results: dict) -> dict:
        result = {
            "predictions": prediction_results
        }
        return result
