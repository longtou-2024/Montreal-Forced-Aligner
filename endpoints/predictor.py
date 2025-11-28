from typing import Optional
import base64
import io
import json
import time
import uuid

from montreal_forced_aligner.lt_mfa2 import setup_mfa, align_one
from montreal_forced_aligner.exceptions import AlignerError
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

from endpoints.utils import parse_ssml, parse_timepoints


class MFAPredictor(Predictor):
    def __init__(self):
        super().__init__()

    def load(self, artifacts_uri: str) -> None:
        # NOTE(longtou): SDK automatically copy files 
        # from "gs://ai-lab-speech-bucket/longtou/db/commbooks/mfa/espeak/"
        # to "." (working dir)
        prediction_utils.download_model_artifacts(artifacts_uri)

        # setup
        dictionary_path = "korean_espeak.dict"
        acoustic_model_path = "acoustic/korean_espeak.zip"
        g2p_model_path = "g2p/korean_espeak.zip"
        # temporary_directory: default="~/Documents/MFA"
        # NOTE(longtou): mp 에서 충돌하지 않게?
        tmp_dir_name = f"~/Documents/MFA/{uuid.uuid4()}"
        x = setup_mfa(dictionary_path, acoustic_model_path, g2p_model_path, temporary_directory=tmp_dir_name)
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
        instances: list = prediction_input["instances"]
        # NOTE(longtou): not support batch prediction
        instance = instances[0]
        parameters: Optional[dict] = prediction_input.get("parameters")

        ssml: Optional[str] = instance.get("ssml")
        transcript: Optional[str] = instance.get("transcript")
        if ssml:
            ssml_results = parse_ssml(ssml)
            transcript = [item['text'] for item in ssml_results if 'text' in item]
            transcript = " ".join(transcript)
        elif transcript:
            ssml_results = None
            pass
        else:
            return {"error": "ssml or transcript must be provided"}

        audio = instance["audio"]
        audio_format = instance.get("audio_format", "wav")
        decoded_audio = base64.b64decode(audio)
        audio_buf = io.BytesIO(decoded_audio)
        audio_buf.seek(0)
        audio_buf.name = f"file.{audio_format}"

        result = {
            "ssml_results": ssml_results,
            "transcript": transcript,
            "audio_buf": audio_buf,
            "audio_format": audio_format,
        }
        return result


    def predict(self, instances: dict) -> dict:
        if "error" in instances:
            return instances
        transcript = instances["transcript"]
        audio_buf = instances["audio_buf"]
        audio_format = instances["audio_format"]
        try:
            s_time = time.perf_counter()
            mfa_result = align_one(
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
            e_time = time.perf_counter()
            align_time = e_time - s_time
            del mfa_result['tiers']['phones'] # not used

            predict_results = {}
            if instances.get('ssml_results'):
                timepoints = parse_timepoints(instances['ssml_results'], mfa_result)
                predict_results.update({
                    'timepoints': timepoints,
                })
            else:
                predict_results.update({
                    'mfa_result': mfa_result
                })
            predict_results.update({'align_time': align_time})
        except Exception as e:
            error_msg = str(e)
            predict_results = {"error": error_msg}

        return predict_results


    def postprocess(self, prediction_results: dict) -> dict:
        result = {
            "predictions": [prediction_results]
        }
        return result
