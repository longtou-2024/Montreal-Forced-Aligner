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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="이미지 태그 접미 (mfa-<tag>)")
    p.add_argument("--run-tag", default="skt8", help="산출물 태그")
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
                "tracked_modified": modified,
                "untracked": sorted(untracked)[:20]}
    except Exception:
        return {"sha": "unknown", "tracked_modified": None, "untracked": []}


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
    manifest_extra = json.dumps({
        "image": image,
        "git_sha_mfa": st_mfa["sha"],
        "git_sha_espnet": st_espnet["sha"],
        # 개수만 남긴다 — 목록은 대부분 구 espeak 런의 잔재(tempdir/·outdir/·ssml_poc/)이고
        # `.dockerignore` 로 이미지에서 제외되므로 MANIFEST 에 나열할 값이 없다.
        "n_untracked_mfa": len(st_mfa["untracked"]),
    })
    if st_mfa["tracked_modified"] or st_espnet["tracked_modified"]:
        print("⚠️  tracked 수정이 남아 있다 — 이미지가 커밋과 다르다(재현 불가). "
              "커밋 후 이미지를 다시 굽고 발사할 것.")
    if st_mfa["untracked"]:
        print(f"ℹ️  MFA 레포 untracked {len(st_mfa['untracked'])}건 "
              f"(구 espeak 런 잔재, .dockerignore 로 이미지 제외): "
              f"{st_mfa['untracked'][:3]} …")

    env = {
        "NJ": str(a.nj),
        "SHARDS_PER_SPEAKER": str(a.shards_per_speaker),
        "GCS_AUDIO": f"{BUCKET}/h100/db/kent-kikiri/skt8_full/audio",
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

    pkg = "/tmp/kent_mfa_skt8.yaml"
    compiler.Compiler().compile(pipeline_func=pipe, package_path=pkg)
    print(f"compiled → {pkg}")
    print(f"image  = {image}")
    print(f"cpu    = {cpu} / mem = {a.memory} / min_disk = {a.min_disk_gb}GB")
    print(f"nj     = {a.nj} (pseudo-speakers = 8 × {a.shards_per_speaker})")
    print(f"out    = {env['GCS_OUT']}")
    if a.dry_run:
        print("--dry-run: 제출하지 않음")
        return
    run = Client(host=KFP_HOST).create_run_from_pipeline_package(pkg, arguments={})
    print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    main()
