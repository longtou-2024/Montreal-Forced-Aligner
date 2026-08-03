#!/usr/bin/env bash
# longtou.2024
# kent-kikiri MFA aligner 이미지 빌드 + Artifact Registry 푸시.
#
# 태그 규약: kent-kikiri develop 트랙과 **같은 AR 레포**를 쓰고 `mfa-` prefix 로 구분한다.
#   us-central1-docker.pkg.dev/prod-ai-project/tts/kent-kikiri:mfa-<tag>
# (구 `tts/mfa:lt` 계보는 endpoint/webdataset 정렬용이라 분리 유지)
#
# 사용: Montreal-Forced-Aligner/h100/build_docker_kent.sh v1
#   ※ **context = 레포의 상위 디렉토리**(~/projects). Dockerfile 이
#     `ADD Montreal-Forced-Aligner /root/...` 를 하기 때문. ~/projects 는 170GB 이므로
#     `~/projects/.dockerignore` 가 반드시 있어야 한다(없으면 데몬 전송에서 사실상 멈춤).
set -euo pipefail

TAG="${1:?usage: build_docker_kent.sh <tag>   (예: v1 → mfa-v1)}"
IMAGE="us-central1-docker.pkg.dev/prod-ai-project/tts/kent-kikiri:mfa-${TAG}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONTEXT="$(dirname "${REPO_DIR}")"
REPO_NAME="$(basename "${REPO_DIR}")"

if [ ! -f "${CONTEXT}/.dockerignore" ]; then
    echo "ERROR: ${CONTEXT}/.dockerignore 가 없다. context 가 170GB 로 잡혀 빌드가 멈춘다." >&2
    exit 1
fi

echo "context = ${CONTEXT} (repo: ${REPO_NAME})"
echo "image   = ${IMAGE}"

docker build -f "${REPO_DIR}/h100/Dockerfile" -t "${IMAGE}" "${CONTEXT}"

gcloud auth print-access-token \
    | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
docker push "${IMAGE}"
echo "pushed: ${IMAGE}"
