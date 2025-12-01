import os
import argparse
import json
import base64

from google.cloud.aiplatform.prediction import LocalModel

from endpoints.predictor import MFAPredictor

USER_SRC_DIR = "endpoints"
ARTIFACT_URI = "gs://ai-lab-speech-bucket/longtou/db/commbooks/mfa/espeak"

#OUTPUT_IMAGE_URI = "us-central1-docker.pkg.dev/prod-ai-project/tts/mfa:endpoint_v1.3"
OUTPUT_IMAGE_URI = "asia-northeast3-docker.pkg.dev/dev-ai-project-357507/tts/mfa:endpoint_v2.3"
BASE_IMAGE = "asia-northeast3-docker.pkg.dev/dev-ai-project-357507/tts/mfa:endpoint_base_v2.3"

debug_samples = [
    #{
    #"text": """
    #아니, 왜 하필 제일 높은 사람한테 돌진한 거냐고!
    #"""},
    {'ssml': """
    <speak>아니, 왜 하필 제일 높은 사람한테 돌진한 거냐고!</speak>
    """},
    #{'ssml': """
    #<speak>아니, 왜<mark name="mark_01" /> 하필 제일 높은 사람한테 돌진한 거냐고!</speak>
    #"""},
    #{'ssml': """
    #<speak>
    #아니, 왜<mark name="mark_01" /> 하필 제일 높은 사람한테 돌진한 거냐고!<mark name="mark_02" />
    #</speak>
    #"""},
    #{'ssml': """
    #<speak>
    #<mark name="mark_00"/>아니, 왜<mark name="mark_01" /> 하필 제일 높은 사람한테 <mark name="mark_02" /> 돌진한 거냐고!<mark name="mark_03" />
    #</speak>
    #""" },
                 ]

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
        output_image_uri=OUTPUT_IMAGE_URI,
        predictor=MFAPredictor,
        base_image=BASE_IMAGE,
    )
    #breakpoint()
    from google.cloud.aiplatform.compat.types import env_var
    local_model.serving_container_spec.env = [env_var.EnvVar(name="VERTEX_CPR_WEB_CONCURRENCY", value='1')]

    print(local_model.get_serving_container_spec())
    #local_model.push_image()
    #print("push image done")
    #import sys; sys.exit()

    # run local
    with local_model.deploy_to_local_endpoint(
        artifact_uri=ARTIFACT_URI,
        #credential_path="local/path/to/your/credentials",
    ) as local_endpoint:
        health_check_response = local_endpoint.run_health_check()
        print(health_check_response, health_check_response.content)

        #sample_dict = get_sample_request()
        #request_dict = {"instances": [sample_dict]}
        #predict_response = local_endpoint.predict(
        #    request=json.dumps(request_dict, ensure_ascii=False),
        #    headers={"Content-Type": "application/json"},
        #)
        #result_dict = json.loads(predict_response.content.decode('utf8'))
        #breakpoint()
        #with open("out.json", 'w') as f:
        #    json.dump(result_dict, f, ensure_ascii=False)
        #print(predict_response, predict_response.content)

        #print(local_endpoint.print_container_logs())

        for item in debug_samples:
            sample_dict = get_sample_request()
            for k, v in item.items():
                sample_dict[k] = v
            request_dict = {"instances": [sample_dict]}
            predict_response = local_endpoint.predict(
                request=json.dumps(request_dict, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
            result_dict = json.loads(predict_response.content.decode('utf8'))

            print(result_dict)
            #breakpoint()
            #with open("out.json", 'w') as f:
            #    json.dump(result_dict, f, ensure_ascii=False)



if __name__ == "__main__":
    main()
