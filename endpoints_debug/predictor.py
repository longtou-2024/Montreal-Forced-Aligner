from typing import Any
import base64
import io
import json
import time

from google.cloud import aiplatform
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils



class DEBUGPredictor(Predictor):
    def __init__(self):
        super().__init__()

    def load(self, artifacts_uri: str) -> None:
        aiplatform.init(project="prod-ai-project", location="us-central1")
        OTHER_ENDPOINT_RESOURCE_NAME = "5764914286778384384"
        OTHER_ENDPOINT = aiplatform.Endpoint(
                    endpoint_name=OTHER_ENDPOINT_RESOURCE_NAME
                )
        self.endpoint = OTHER_ENDPOINT

    def preprocess(self, prediction_input: dict) -> dict:
        #instances: list = prediction_input["instances"]
        # NOTE(longtou): not support batch prediction
        #instances = instances[0]
        return prediction_input

    def predict(self, instances: dict) -> dict:
        try:
            #result = self.endpoint.predict(instances=instances)
            result = self.endpoint.predict(instances=instances["instances"])
            result = result[0]
        except Exception as e:
            error_msg = str(e)
            result = {"error": error_msg}

        return result


    def postprocess(self, prediction_results: dict) -> dict:
        return {"predictions": prediction_results}
        #result = {
        #    #"predictions": json.dumps(prediction_results, ensure_ascii=False)
        #    "predictions": [prediction_results]
        #}
        #return result
