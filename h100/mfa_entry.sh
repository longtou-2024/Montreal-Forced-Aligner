#!/usr/bin/env bash
# longtou.2024
#
# GKE CPU pod 엔트리 — skt8 MFA 음향모델 학습 + 정렬 풀런.
# 설계 정본 = kent-kikiri `docs/wiki/mfa-kent-g2p-aligner.md`
#
# env (run_mfa_kent.py 가 주입):
#   NJ                  MFA job 수 (= 요청 vCPU 와 맞춘다)
#   SHARDS_PER_SPEAKER  pseudo-speaker 샤딩 배수. NJ = 8 × 이 값이 되게 잡는다.
#                       (MFA 는 화자 수보다 많은 job 을 못 쓴다 — skt8 은 8화자)
#   GCS_AUDIO           skt8 오디오 prefix (…/skt8_full/audio)
#   GCS_TEXT_TSV        복원한 원문 TSV (uttid<TAB>transcript)
#   GCS_OUT             산출물 업로드 prefix
#   LOCAL_ROOT          로컬(ephemeral) 작업 루트 — gcsfuse 아님!
#   RUN_TAG             산출물 태그
#   MANIFEST_EXTRA      MANIFEST 에 병합할 JSON (git SHA·이미지 등)
set -euo pipefail

: "${NJ:?}" "${SHARDS_PER_SPEAKER:?}" "${GCS_AUDIO:?}" "${GCS_TEXT_TSV:?}"
: "${GCS_OUT:?}" "${LOCAL_ROOT:?}" "${RUN_TAG:?}"

log() { echo "[mfa-entry $(date -u +%H:%M:%S)] $*"; }
T0=$SECONDS

AUDIO="${LOCAL_ROOT}/audio"
WORK="${LOCAL_ROOT}/work"
OUT="${LOCAL_ROOT}/out"
TSV="${LOCAL_ROOT}/skt8_text.tsv"
mkdir -p "${AUDIO}" "${WORK}" "${OUT}"

MFA_ROOT=/root/Montreal-Forced-Aligner
RECIPE="${MFA_ROOT}/espnet/egs2/skt_emotion/tts1"

log "환경: NJ=${NJ} SHARDS=${SHARDS_PER_SPEAKER} LOCAL_ROOT=${LOCAL_ROOT}"
log "코어 $(nproc) / 메모리 $(free -g | awk '/^Mem:/{print $2}')GB"
df -h "${LOCAL_ROOT}"
# kfp 2.9.0 에 ephemeral-storage 요청 API 가 없어 스케줄러 보장을 못 받는다 →
# 여기서 하드 체크. 스테이징(35GB) + 피처·중간산출(~250GB) 전에 죽어야 한다.
FREE_GB=$(df -BG --output=avail "${LOCAL_ROOT}" | tail -1 | tr -dc '0-9')
log "여유 디스크 ${FREE_GB}GB (최소 ${MIN_DISK_GB:-300}GB)"
[ "${FREE_GB}" -ge "${MIN_DISK_GB:-300}" ] || {
    log "FATAL: 로컬 디스크 부족 (${FREE_GB}GB < ${MIN_DISK_GB:-300}GB)"; exit 1; }

# ── 1. 스테이징 ────────────────────────────────────────────────────────────
# NOTE(longtou): 개별 wav 150,631 객체를 받는다(≈15분). `wds_v3_mfa/large/*.tar`
#   152개를 스트리밍하면 객체 수가 1/1000 로 줄어 더 빠르지만, skt8_full/audio 는
#   **kent-kikiri 가 실제 학습에 쓰는 바로 그 파일**이라 라벨과의 정합이 보장된다.
#   13~17h 잡에서 15분은 싸므로 안전한 쪽을 택한다.
log "오디오 스테이징 시작: ${GCS_AUDIO}"
gcloud storage rsync -r "${GCS_AUDIO}" "${AUDIO}"
N_WAV=$(find "${AUDIO}" -name '*.wav' | wc -l)
log "스테이징 완료: ${N_WAV} wav ($(du -sh "${AUDIO}" | cut -f1)), ${SECONDS}s"
[ "${N_WAV}" -gt 100000 ] || { log "FATAL: wav 수가 비정상(${N_WAV})"; exit 1; }

gcloud storage cp "${GCS_TEXT_TSV}" "${TSV}"
log "TSV: $(wc -l < "${TSV}") 줄"

# ── 2. 학습 + 정렬 ─────────────────────────────────────────────────────────
cd "${RECIPE}"
# shellcheck disable=SC1091
. "${MFA_ROOT}/activate_python.sh"
export SETUPTOOLS_SCM_PRETEND_VERSION=3.3.5.dev15
rm -rf data raw_data

log "run_mfa_skt8.sh 시작 (stage 0→4)"
T_TRAIN=$SECONDS
./local/run_mfa_skt8.sh \
    --nj "${NJ}" \
    --shards_per_speaker "${SHARDS_PER_SPEAKER}" \
    --audio_root "${AUDIO}" \
    --text_tsv "${TSV}" \
    --workdir "${WORK}"
SEC_TRAIN=$(( SECONDS - T_TRAIN ))
log "학습+정렬 완료: ${SEC_TRAIN}s"

# ── 3. 집계 ────────────────────────────────────────────────────────────────
N_ALN=$(find "${WORK}/alignments" -name '*.json' | wc -l)
log "정렬 json ${N_ALN}개"
[ "${N_ALN}" -gt 100000 ] || { log "FATAL: 정렬 결과가 비정상(${N_ALN})"; exit 1; }

EXTRA=$(python - <<PYEOF
import json, os
print(json.dumps({
    "nj": int(os.environ["NJ"]),
    "shards_per_speaker": int(os.environ["SHARDS_PER_SPEAKER"]),
    "n_alignments": ${N_ALN},
    "n_wav_staged": ${N_WAV},
    "seconds_train_align": ${SEC_TRAIN},
    **json.loads(os.environ.get("MANIFEST_EXTRA") or "{}"),
}))
PYEOF
)
python "${MFA_ROOT}/h100/aggregate_alignments.py" \
    --alignments "${WORK}/alignments" \
    --out "${OUT}" --tag "${RUN_TAG}" --jobs "${NJ}" \
    --manifest-extra "${EXTRA}"

# ── 4. 업로드 ──────────────────────────────────────────────────────────────
# 모델 3종 = 재현·재사용의 핵심. workdir 전체 tar(선례 2.38GB)는 올리지 않는다
# — 모델 + 코드 SHA 로 재현되고, corpus lab 은 TSV 에서 언제든 재생성된다.
mkdir -p "${OUT}/model"
cp "${WORK}/korean_kent.dict"           "${OUT}/model/"
cp "${WORK}/acoustic/korean_kent.zip"   "${OUT}/model/"
cp "${WORK}/g2p/korean_kent.zip"        "${OUT}/model/g2p_korean_kent.zip"
if [ -f "${WORK}/alignments/alignment_analysis.csv" ]; then
    gzip -c "${WORK}/alignments/alignment_analysis.csv" \
        > "${OUT}/alignment_analysis.csv.gz"
fi

log "업로드: ${GCS_OUT}"
gcloud storage cp -r "${OUT}"/* "${GCS_OUT}/"
log "총 소요 $(( SECONDS - T0 ))s"
log "DONE"
