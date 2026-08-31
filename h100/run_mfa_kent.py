#!/usr/bin/env python
# longtou.2024
"""skt8 MFA 학습·정렬 잡 제출 (Kubeflow Pipelines).

설계 정본 = kent-kikiri `docs/wiki/mfa-kent-g2p-aligner.md`
인프라 선례 = 같은 클러스터/버킷/PVC (kent-piper `h100-kfp-job-pipeline`).

kfp 설치 환경에서 실행: ~/docker/conda/envs/docker/bin/python h100/run_mfa_kent.py --tag v1

⚠️ **CPU-only 잡인데 GPU 노드에 올라간다.** 이 클러스터에는 128 vCPU 를 담을 CPU 전용
   노드풀이 없다 (a3-highgpu-8g 208vCPU × 3대 / n4-standard-4 × 2대뿐).
   따라서 (a) GPU 는 0개 요청하고 (b) `nvidia.com/gpu=present:NoSchedule` taint 를
   **toleration 으로 통과**한다. GPU 자체는 점유하지 않으므로 8× H100 은 그대로 남는다.
   단 CPU 는 128/208 을 먹으므로, 그 노드에는 8-GPU 잡 하나가 못 들어온다
   (2026-08-03 시점 빈 노드 2대 확인).
"""

import argparse
import json
import shlex
import subprocess

from google.protobuf import json_format
from kfp import compiler, dsl, kubernetes
from kfp.client import Client
from kfp.dsl import PipelineTask
from kfp.kubernetes import common

KFP_HOST = "https://3313888af2601658-dot-us-central1.pipelines.googleusercontent.com"
PVC_NAME = "longtou-gcs-fuse-csi-static-pvc3"   # gcsfuse: prod-ai-lab-speech-bucket
MOUNT_PATH = "/home/longtou.2024/mount"
BUCKET = "gs://prod-ai-lab-speech-bucket/longtou"

# GPU 노드 taint (kubectl get nodes -o custom-columns=...:.spec.taints 로 확인)
GPU_TAINT = dict(key="nvidia.com/gpu", operator="Equal",
                 value="present", effect="NoSchedule")

# 정렬 대상 오디오 기본값 = **학습 정본 코퍼스(silnorm)**. 정렬 타임스탬프가 실제 학습
# 오디오와 어긋나면 aligner 감독도 무음 클램프도 좌표를 잃는다.
DEFAULT_AUDIO_TARS = f"{BUCKET}/h100/db/kent-kikiri/skt8_full_silnorm/audio_tars"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="이미지 태그 접미 (mfa-<tag>)")
    p.add_argument("--run-tag", default="skt8", help="산출물 태그")
    # 오디오 소스 — 기본은 **silnorm**(현 학습 정본 코퍼스). 정렬 타임스탬프가 실제 학습
    # 오디오와 일치해야 aligner 감독·무음 클램프가 성립한다. silnorm 은 tar 샤드로만
    # 영속돼 있어 tar 경로가 기본이고, 개별 wav prefix 를 쓰려면 --audio-gs 를 준다.
    # 기본값을 문자열로 박으면 --audio-gs 만 준 사용자의 의도가 조용히 무시된다
    # (Codex 검수 2026-08-31 MAJOR) → None 센티넬로 두고 아래에서 해석한다.
    p.add_argument("--audio-tars-gs", default=None,
                   help=f"tar 샤드 prefix (기본 {DEFAULT_AUDIO_TARS})")
    p.add_argument("--audio-gs", default=None,
                   help="개별 wav prefix (구 경로: …/skt8_full/audio). "
                        "--audio-tars-gs 와 동시 지정 불가")
    # 코퍼스 인벤토리 — 엔트리가 스테이징 직후 정확 대조한다(부분 코퍼스 조기 차단).
    p.add_argument("--expect-tars", type=int, default=16)
    p.add_argument("--expect-wav", type=int, default=151680)
    p.add_argument("--expect-tsv", type=int, default=150631)
    # g2p 세대 게이트 — 이 트랙의 존재 이유가 g2p 교체다. 구 세대로 13시간을 태우지 않는다.
    p.add_argument("--expect-g2p-sha", default="9c6faef",
                   help="ai-lab-tts-g2p HEAD 가 이 SHA 로 시작해야 한다 (빈 문자열=검사 생략)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="tracked 수정/unknown SHA 상태로도 발사 (재현성 포기, 명시적일 때만)")
    p.add_argument("--overwrite", action="store_true",
                   help="산출 prefix 가 비어있지 않아도 발사")
    # nj = 8(실화자) × shards. 128 = 8 × 16.
    p.add_argument("--nj", type=int, default=128)
    p.add_argument("--shards-per-speaker", type=int, default=16)
    p.add_argument("--cpu", default=None, help="기본 = nj")
    p.add_argument("--memory", default="256Gi")
    p.add_argument("--min-disk-gb", type=int, default=300,
                   help="엔트리에서 확인할 최소 로컬 여유 디스크")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def add_toleration_and_annotation(task: PipelineTask, annotations: dict) -> None:
    msg = common.get_existing_kubernetes_config_as_message(task)
    msg.pod_metadata.annotations.update(annotations)
    task.platform_config["kubernetes"] = json_format.MessageToDict(msg)


def git_state(path: str) -> dict:
    """HEAD SHA + 오염 상태. 이미지는 워킹트리를 ADD 하므로 dirty 인 채 발사하면
    MANIFEST 의 SHA 로 재현이 안 된다.

    tracked 수정과 untracked 파일을 **구분**한다 — 재현성을 실제로 깨는 것은 tracked
    수정이고, untracked 는 대부분 `.dockerignore` 로 걸러지는 잔재(outdir/·tempdir/ 등)라
    뭉뚱그리면 진짜 dirty 를 못 알아본다.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short=12", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        modified = subprocess.call(
            ["git", "-C", path, "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        untracked = subprocess.check_output(
            ["git", "-C", path, "ls-files", "--others", "--exclude-standard"],
            text=True, stderr=subprocess.DEVNULL).split()
        return {"sha": f"{sha}-dirty" if modified else sha,
                "raw_sha": sha,
                "tracked_modified": modified,
                # 개수는 자르기 **전** 값을 따로 남긴다 (Codex 검수 2026-08-31 MINOR).
                "n_untracked": len(untracked),
                "untracked": sorted(untracked)[:20]}
    except Exception:
        return {"sha": "unknown", "raw_sha": "unknown", "tracked_modified": None,
                "n_untracked": 0, "untracked": []}


def image_digest(image: str) -> str:
    """AR 에 실제로 존재하는 이미지의 digest. 없으면 빈 문자열.

    태그는 가변이라 MANIFEST 의 태그만으로는 실행 바이트를 특정할 수 없다
    (Codex 검수 2026-08-31 MAJOR). 겸사겸사 **이미지 미푸시 상태의 발사**도 여기서 잡힌다.
    """
    repo, _, tag = image.rpartition(":")
    try:
        out = subprocess.check_output(
            ["gcloud", "artifacts", "docker", "images", "describe", f"{repo}:{tag}",
             "--format=value(image_summary.digest)"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return out
    except Exception:
        return ""


def gcs_prefix_nonempty(prefix: str) -> bool:
    try:
        out = subprocess.check_output(["gcloud", "storage", "ls", f"{prefix}/"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        return bool(out)
    except Exception:
        return False


def main() -> None:
    a = parse_args()
    cpu = a.cpu or str(a.nj)
    image = f"us-central1-docker.pkg.dev/prod-ai-project/tts/kent-kikiri:mfa-{a.tag}"

    if a.nj != 8 * a.shards_per_speaker:
        print(f"⚠️  nj({a.nj}) != 8 × shards({a.shards_per_speaker}) = "
              f"{8*a.shards_per_speaker}. MFA 는 화자(=pseudo-speaker) 수보다 많은 job 을 "
              f"쓰지 못하므로 min 값으로 클램프된다.")

    # ⚠️ espnet 은 **MFA 레포의 서브모듈**을 재야 한다. 이미지에 들어가는 것은 그쪽이고,
    #    독립 클론 `~/projects/espnet` 은 이미지와 무관하다(여러 트랙이 공유하는 작업본이라
    #    항상 dirty 에 가깝다 — 처음에 그걸 재서 재현성 게이트가 오작동했다).
    MFA_DIR = "/home/longtou.2024/projects/Montreal-Forced-Aligner"
    st_mfa = git_state(MFA_DIR)
    st_espnet = git_state(f"{MFA_DIR}/espnet")
    # g2p 세대는 정렬 사전·phone 열을 통째로 바꾼다 → 산출물에 반드시 박는다
    # (kent-kikiri 메모리 `g2p-phoneme-convention-drift`).
    st_g2p = git_state(f"{MFA_DIR}/ai-lab-tts-g2p")

    # ── 오디오 소스 해석 (상호배타) ────────────────────────────────────────────
    if a.audio_gs and a.audio_tars_gs:
        raise SystemExit("--audio-gs 와 --audio-tars-gs 는 동시에 줄 수 없다")
    if a.audio_gs:
        audio_tars, audio_wav = "", a.audio_gs
    else:
        audio_tars, audio_wav = (a.audio_tars_gs or DEFAULT_AUDIO_TARS), ""

    # ── 발사 전 게이트 (전부 "13시간 뒤에 알게 되는 일"을 앞으로 당기는 장치) ──
    if a.expect_g2p_sha and not st_g2p["raw_sha"].startswith(a.expect_g2p_sha):
        raise SystemExit(
            f"g2p SHA 불일치: 기대 {a.expect_g2p_sha}…, 실제 {st_g2p['raw_sha']}. "
            f"이 트랙의 목적이 g2p 교체다 — 구 세대로 발사하지 않는다.")
    dirty = [n for n, st in (("mfa", st_mfa), ("espnet", st_espnet), ("g2p", st_g2p))
             if st["tracked_modified"] or st["sha"] == "unknown"]
    if dirty and not a.allow_dirty:
        raise SystemExit(
            f"tracked 수정/unknown SHA: {', '.join(dirty)} — 이미지가 커밋과 달라 "
            f"MANIFEST SHA 로 재현되지 않는다. 커밋 후 재빌드하거나 --allow-dirty 를 명시하라.")
    digest = image_digest(image)
    if not digest:
        raise SystemExit(
            f"이미지를 AR 에서 찾지 못했다: {image}. 빌드·푸시 후 발사하라 "
            f"(태그 오타·미푸시를 여기서 잡는다).")

    manifest_extra = json.dumps({
        "image": image,
        "image_digest": digest,
        "git_sha_mfa": st_mfa["sha"],
        "git_sha_espnet": st_espnet["sha"],
        "git_sha_g2p": st_g2p["sha"],
        "audio_source": audio_tars or audio_wav,
        "audio_kind": "tar" if audio_tars else "wav",
        # 개수만 남긴다 — 목록은 대부분 구 espeak 런의 잔재(tempdir/·outdir/·ssml_poc/)이고
        # `.dockerignore` 로 이미지에서 제외되므로 MANIFEST 에 나열할 값이 없다.
        "n_untracked_mfa": st_mfa["n_untracked"],
    })
    if dirty:  # --allow-dirty 로 통과한 경우에만 여기 온다
        print(f"⚠️  --allow-dirty 로 진행: {', '.join(dirty)} (재현 불가 상태)")
    if st_mfa["untracked"]:
        print(f"ℹ️  MFA 레포 untracked {st_mfa['n_untracked']}건 "
              f"(구 espeak 런 잔재, .dockerignore 로 이미지 제외): "
              f"{st_mfa['untracked'][:3]} …")

    env = {
        "NJ": str(a.nj),
        "SHARDS_PER_SPEAKER": str(a.shards_per_speaker),
        "GCS_AUDIO": audio_wav,
        "GCS_AUDIO_TARS": audio_tars,
        "EXPECT_TARS": str(a.expect_tars) if audio_tars else "",
        "EXPECT_WAV": str(a.expect_wav),
        "EXPECT_TSV": str(a.expect_tsv),
        "GCS_TEXT_TSV": f"{BUCKET}/h100/db/kent-kikiri/round2/aligner_ko/kent/skt8_text.tsv",
        "GCS_OUT": f"{BUCKET}/h100/db/kent-kikiri/round2/aligner_ko/kent/{a.run_tag}",
        # ⚠️ gcsfuse 마운트가 아니라 **pod 로컬 ephemeral** 이어야 한다.
        #    MFA tempdir 을 gcsfuse 에 두면 수백만 소파일 I/O 로 사실상 멈춘다.
        "LOCAL_ROOT": "/root/mfa_work",
        "RUN_TAG": a.run_tag,
        "MIN_DISK_GB": str(a.min_disk_gb),
        "MANIFEST_EXTRA": manifest_extra,
    }
    # ⚠️ shlex.quote 필수. MANIFEST_EXTRA 는 JSON 이라 값 안에 `"` 가 들어가는데
    #    f'{k}="{v}"' 로 감싸면 내부 따옴표가 문자열을 끊어 sh 가 잘못 파싱한다
    #    (실측: MANIFEST_EXTRA 가 `{image:` 로 잘려 json.loads 실패 → **13시간 학습 후**
    #    집계 단계에서 죽는다).
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    command = ["sh", "-c", f"{env_str} bash /root/Montreal-Forced-Aligner/h100/mfa_entry.sh"]

    @dsl.container_component
    def mfa_train_align() -> dsl.ContainerSpec:
        return dsl.ContainerSpec(image=image, command=command)

    @dsl.pipeline(name="kent-mfa-skt8")
    def pipe():
        task = mfa_train_align()
        task.set_cpu_request(cpu)
        task.set_memory_request(a.memory)
        # NOTE(longtou): kfp 2.9.0 에는 set_ephemeral_storage_request 가 없다.
        #   a3-highgpu-8g 노드의 ephemeral-storage 는 allocatable 5.6TB 라 우리 ~300GB
        #   수요에 여유가 충분하므로 요청 없이 가고, 대신 엔트리에서 MIN_DISK_GB 로
        #   하드 체크한다(부족하면 스테이징 전에 죽는다).
        # GPU 는 요청하지 않는다 (CPU-only 잡). taint 만 통과시킨다.
        kubernetes.add_toleration(task, **GPU_TAINT)
        # 버킷은 gcloud CLI 로 직접 읽고 쓴다 — gcsfuse 마운트는 붙이지 않는다
        # (MFA temp I/O 를 실수로 마운트에 흘리는 것을 구조적으로 막는다).
        add_toleration_and_annotation(task, {"run_tag": a.run_tag})

    # 산출 prefix 덮어쓰기 방지 — 구 런(skt8/·skt8-d2/)을 지우지 않는다
    # (Codex 검수 2026-08-31 MAJOR). 부분 업로드 판별은 엔트리가 마지막에 쓰는 _SUCCESS 로.
    if gcs_prefix_nonempty(env["GCS_OUT"]) and not a.overwrite:
        raise SystemExit(
            f"산출 prefix 가 비어있지 않다: {env['GCS_OUT']}\n"
            f"다른 --run-tag 를 쓰거나, 의도적 덮어쓰기면 --overwrite 를 명시하라.")

    pkg = "/tmp/kent_mfa_skt8.yaml"
    compiler.Compiler().compile(pipeline_func=pipe, package_path=pkg)
    print(f"compiled → {pkg}")
    print(f"image  = {image}")
    print(f"cpu    = {cpu} / mem = {a.memory} / min_disk = {a.min_disk_gb}GB")
    print(f"nj     = {a.nj} (pseudo-speakers = 8 × {a.shards_per_speaker})")
    print(f"audio  = {audio_tars or audio_wav}{' (tar)' if audio_tars else ' (wav)'}")
    print(f"expect = tars {a.expect_tars} / wav {a.expect_wav} / tsv {a.expect_tsv}")
    print(f"digest = {digest}")
    print(f"g2p    = {st_g2p['sha']}")
    print(f"out    = {env['GCS_OUT']}")
    if a.dry_run:
        print("--dry-run: 제출하지 않음")
        return
    run = Client(host=KFP_HOST).create_run_from_pipeline_package(pkg, arguments={})
    print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    main()
