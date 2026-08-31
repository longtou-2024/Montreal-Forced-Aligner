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
#   GCS_AUDIO           skt8 오디오 prefix (…/skt8_full/audio) — 개별 wav 경로
#   GCS_AUDIO_TARS      (선택) tar 샤드 prefix (…/skt8_full_silnorm/audio_tars).
#                       설정되면 이쪽이 우선하고 GCS_AUDIO 는 쓰지 않는다.
#   STAGE_PAR           (선택) tar 스테이징 병렬도 (기본 16)
#   EXPECT_TARS         (선택) 기대 tar 샤드 수. 불일치 시 즉시 중단
#   EXPECT_WAV          (선택) 기대 wav 수. 불일치 시 즉시 중단
#   EXPECT_TSV          (선택) 기대 TSV 줄 수. 불일치 시 즉시 중단
#                       ↑ 셋 다 "부분 코퍼스로 13시간 학습" 을 앞단에서 막는 게이트다
#                         (Codex 검수 2026-08-31 MAJOR). 미설정이면 구 느슨한 검사로 폴백.
#   GCS_TEXT_TSV        복원한 원문 TSV (uttid<TAB>transcript)
#   GCS_OUT             산출물 업로드 prefix
#   LOCAL_ROOT          로컬(ephemeral) 작업 루트 — gcsfuse 아님!
#   RUN_TAG             산출물 태그
#   MANIFEST_EXTRA      MANIFEST 에 병합할 JSON (git SHA·이미지 등)
set -euo pipefail

: "${NJ:?}" "${SHARDS_PER_SPEAKER:?}" "${GCS_TEXT_TSV:?}"
: "${GCS_OUT:?}" "${LOCAL_ROOT:?}" "${RUN_TAG:?}"
# 오디오 소스는 둘 중 하나가 반드시 있어야 한다.
if [ -z "${GCS_AUDIO:-}" ] && [ -z "${GCS_AUDIO_TARS:-}" ]; then
    echo "FATAL: GCS_AUDIO 또는 GCS_AUDIO_TARS 중 하나는 필요하다" >&2; exit 1
fi

log() { echo "[mfa-entry $(date -u +%H:%M:%S)] $*"; }
T0=$SECONDS

AUDIO="${LOCAL_ROOT}/audio"
WORK="${LOCAL_ROOT}/work"
OUT="${LOCAL_ROOT}/out"
TSV="${LOCAL_ROOT}/skt8_text.tsv"
mkdir -p "${AUDIO}" "${WORK}" "${OUT}"

# 실패 시에도 학습 로그를 보존한다 (2026-08-03 풀런에서 train_align.log 유실 교훈).
# ${WORK}/tmp/logs = train_cmd 가 남기는 단계별 로그 (features 등 대용량은 제외).
upload_logs() {
    if [ -d "${WORK}/tmp/logs" ]; then
        tar -C "${WORK}/tmp" -czf "/tmp/logs_${RUN_TAG}.tar.gz" logs || return 0
        gcloud storage cp "/tmp/logs_${RUN_TAG}.tar.gz" \
            "${GCS_OUT}/logs_${RUN_TAG}.tar.gz" || true
    fi
}
trap upload_logs EXIT

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
if [ -n "${GCS_AUDIO_TARS:-}" ]; then
    # tar 샤드 경로. silnorm 코퍼스는 개별 wav prefix 없이 tar 로만 영속돼 있고,
    # 객체 수가 150,631 → 16 이라 스테이징이 훨씬 싸다([[h100-tar-transfer-optimization]]).
    # tar 멤버 경로가 `skt_F0001/F0001_000001.wav` 라 rsync 판과 동일한 레이아웃이 된다.
    log "오디오 스테이징(tar) 시작: ${GCS_AUDIO_TARS}"
    TAR_LIST="${LOCAL_ROOT}/tars.txt"
    gcloud storage ls "${GCS_AUDIO_TARS}/*.tar.gz" > "${TAR_LIST}"
    N_TAR=$(wc -l < "${TAR_LIST}")
    [ "${N_TAR}" -gt 0 ] || { log "FATAL: tar 샤드가 없다: ${GCS_AUDIO_TARS}"; exit 1; }
    if [ -n "${EXPECT_TARS:-}" ] && [ "${N_TAR}" -ne "${EXPECT_TARS}" ]; then
        log "FATAL: tar 샤드 수 불일치 — 기대 ${EXPECT_TARS}, 실제 ${N_TAR}"; exit 1
    fi
    log "샤드 ${N_TAR}개, 병렬 ${STAGE_PAR:-16}"
    # xargs 는 자식이 하나라도 실패하면 123 을 반환 → set -e 로 즉시 중단된다.
    xargs -a "${TAR_LIST}" -P "${STAGE_PAR:-16}" -I@@ \
        bash -c 'set -o pipefail; gcloud storage cat "$0" | tar xzf - -C "$1"' @@ "${AUDIO}"
else
    log "오디오 스테이징 시작: ${GCS_AUDIO}"
    gcloud storage rsync -r "${GCS_AUDIO}" "${AUDIO}"
fi
N_WAV=$(find "${AUDIO}" -name '*.wav' | wc -l)
log "스테이징 완료: ${N_WAV} wav ($(du -sh "${AUDIO}" | cut -f1)), ${SECONDS}s"
if [ -n "${EXPECT_WAV:-}" ]; then
    # 정확 대조. 부분 추출(샤드 일부 실패)로 13시간을 태우지 않기 위한 게이트다.
    [ "${N_WAV}" -eq "${EXPECT_WAV}" ] || {
        log "FATAL: wav 수 불일치 — 기대 ${EXPECT_WAV}, 실제 ${N_WAV}"; exit 1; }
else
    [ "${N_WAV}" -gt 100000 ] || { log "FATAL: wav 수가 비정상(${N_WAV})"; exit 1; }
fi

gcloud storage cp "${GCS_TEXT_TSV}" "${TSV}"
N_TSV=$(wc -l < "${TSV}")
log "TSV: ${N_TSV} 줄"
if [ -n "${EXPECT_TSV:-}" ] && [ "${N_TSV}" -ne "${EXPECT_TSV}" ]; then
    log "FATAL: TSV 줄 수 불일치 — 기대 ${EXPECT_TSV}, 실제 ${N_TSV}"; exit 1
fi

# uttid 교차 대조 — TSV 의 모든 발화가 스테이징된 오디오에 있는가.
# (코퍼스 wav 수 > TSV 줄 수 인 것은 정상이다: 전사 없는 클립이 존재한다.
#  반대 방향 = TSV 에만 있는 uttid 는 곧 정렬 누락이므로 여기서 막는다.)
find "${AUDIO}" -name '*.wav' -printf '%f\n' | sed 's/\.wav$//' | sort -u > "${LOCAL_ROOT}/wav_ids.txt"
cut -f1 "${TSV}" | sort -u > "${LOCAL_ROOT}/tsv_ids.txt"
N_MISS=$(comm -13 "${LOCAL_ROOT}/wav_ids.txt" "${LOCAL_ROOT}/tsv_ids.txt" | wc -l)
if [ "${N_MISS}" -ne 0 ]; then
    log "FATAL: TSV 에 있는데 오디오에 없는 uttid ${N_MISS}건 (예: $(comm -13 "${LOCAL_ROOT}/wav_ids.txt" "${LOCAL_ROOT}/tsv_ids.txt" | head -3 | tr '\n' ' '))"
    exit 1
fi
log "uttid 대조 통과: TSV ${N_TSV} 전건이 오디오에 존재"

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
    --lexicon_source "${LEXICON_SOURCE:-train_dict}" \
    --shards_per_speaker "${SHARDS_PER_SPEAKER}" \
    --audio_root "${AUDIO}" \
    --text_tsv "${TSV}" \
    --workdir "${WORK}"
SEC_TRAIN=$(( SECONDS - T_TRAIN ))
log "학습+정렬 완료: ${SEC_TRAIN}s"

# ── 3. 집계 ────────────────────────────────────────────────────────────────
N_ALN=$(find "${WORK}/alignments" -name '*.json' | wc -l)
log "정렬 json ${N_ALN}개 (TSV ${N_TSV} 대비 $(awk -v a="${N_ALN}" -v t="${N_TSV}" 'BEGIN{printf "%.2f%%", 100*a/t}'))"
# 정렬은 음향 실패로 소수가 빠질 수 있다(선례: 150,631 → 150,603 = 99.98%).
# 그래서 절대값이 아니라 TSV 대비 비율로 본다. 하한 99%.
MIN_ALN=$(awk -v t="${N_TSV}" 'BEGIN{printf "%d", t*0.99}')
[ "${N_ALN}" -ge "${MIN_ALN}" ] || {
    log "FATAL: 정렬 결과 부족 — ${N_ALN} < ${MIN_ALN}(TSV ${N_TSV} 의 99%)"; exit 1; }

EXTRA=$(python - <<PYEOF
import json, os
print(json.dumps({
    "nj": int(os.environ["NJ"]),
    "shards_per_speaker": int(os.environ["SHARDS_PER_SPEAKER"]),
    "lexicon_source": os.environ.get("LEXICON_SOURCE", "train_dict"),
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
# 완료 마커는 **맨 마지막**. 업로드가 중간에 끊기면 마커가 없으므로 소비자가
# 부분 산출물을 정상으로 오인하지 않는다 (Codex 검수 2026-08-31 MAJOR).
date -u +%Y-%m-%dT%H:%M:%SZ | gcloud storage cp - "${GCS_OUT}/_SUCCESS"
log "총 소요 $(( SECONDS - T0 ))s"
log "DONE"
