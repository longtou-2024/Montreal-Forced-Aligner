import os
import argparse
import json
import base64

from google.cloud.aiplatform.prediction import LocalModel

from endpoints_debug.predictor import DEBUGPredictor

USER_SRC_DIR = "endpoints_debug"
#ARTIFACT_URI = "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/mfa/espeak"

def get_sample_request():
    result = {}
    with open("ssml_poc/sample_01.wav", 'rb') as f:
        audio_bytes = f.read()
        # Base64 인코딩을 수행합니다. 결과는 bytes 객체입니다.
        encoded_bytes = base64.b64encode(audio_bytes)
        # JSON 요청에 포함하기 위해 Base64 bytes를 UTF-8 문자열로 디코드합니다.
        encoded_string = encoded_bytes.decode("utf-8")
        result['audio'] = encoded_string
    with open("ssml_poc/sample_01.lab", 'r') as f:
        line = f.readlines()[0]
        result["transcript"] = line.strip()
        result["audio_format"] = "wav"

    return result


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    local_model = LocalModel.build_cpr_model(
        src_dir=USER_SRC_DIR,
        output_image_uri="us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:endpoint_debug",
        predictor=DEBUGPredictor,
        base_image="us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:endpoint_base_v1.0",
    )
    #breakpoint()
    print(local_model.get_serving_container_spec())
    local_model.push_image()
    print("push image done")

    # run local
    #with local_model.deploy_to_local_endpoint(
    #    #artifact_uri=ARTIFACT_URI,
    #    artifact_uri=None,
    #    #credential_path="local/path/to/your/credentials",
    #) as local_endpoint:
    #    health_check_response = local_endpoint.run_health_check()
    #    print(health_check_response, health_check_response.content)

    #    sample_dict = get_sample_request()
    #    request_dict = {"instances": [sample_dict]}
    #    predict_response = local_endpoint.predict(
    #        request=json.dumps(request_dict, ensure_ascii=False),
    #        headers={"Content-Type": "application/json"},
    #    )
    #    result_dict = json.loads(predict_response.content.decode('utf8'))
    #    breakpoint()
    #    #with open("out.json", 'w') as f:
    #    #    json.dump(result_dict, f, ensure_ascii=False)
    #    #print(predict_response, predict_response.content)

    #    #print(local_endpoint.print_container_logs())



if __name__ == "__main__":
    main()
