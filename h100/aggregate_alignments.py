#!/usr/bin/env python3
# longtou.2024
"""MFA 정렬 산출물 집계 — 개별 json 15만개 → jsonl.gz + phone 인벤토리 + MANIFEST.

**개별 json 을 버킷에 그대로 올리지 않는다.** 15만 객체는 나중에 읽을 때 객체 수가
그대로 비용이 된다(선례: espeak 런도 `mfa_skt.tar.gz` 집계본 하나로 보관, 52.5MB).

출력 (--out 아래):
  mfa_kent_<tag>.jsonl.gz   — utt 당 1줄 {utt, spk, dur, phones:[[s,e,ph],…], words:[…]}
  phone_inventory_kent.json — phone 빈도 + 총계
  MANIFEST.json             — 재현 정보(코드 SHA·이미지·nj·shards·소요시간·코퍼스 해시)

`spk` 는 pseudo-speaker 라벨(`skt_F0001_p003`)에서 실화자(`skt_F0001`)를 복원해 담는다
— 샤딩은 병렬화 수단일 뿐이고 다운스트림은 실화자를 봐야 한다.
"""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PSEUDO_RE = re.compile(r"^(?P<spk>.+?)_p\d{3}$")


def real_speaker(label: str) -> str:
    m = PSEUDO_RE.match(label)
    return m.group("spk") if m else label


def one_speaker_dir(d: str):
    """화자 디렉토리 1개를 처리 (병렬 단위)."""
    d = Path(d)
    label = d.name
    spk = real_speaker(label)
    recs, inv = [], Counter()
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        tiers = data.get("tiers", {})
        phones = tiers.get("phones", {}).get("entries", [])
        words = tiers.get("words", {}).get("entries", [])
        for _s, _e, p in phones:
            inv[p] += 1
        recs.append({
            "utt": f.stem,
            "spk": spk,
            "pseudo_spk": label,
            "start": data.get("start"),
            "end": data.get("end"),
            "phones": phones,
            "words": words,
        })
    return recs, inv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alignments", required=True, help="mfa align 산출 디렉토리")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="skt8")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--manifest-extra", default=None,
                    help="MANIFEST 에 병합할 JSON 문자열 (git SHA·이미지·nj 등)")
    args = ap.parse_args()

    aln = Path(args.alignments)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spk_dirs = [str(p) for p in sorted(aln.iterdir()) if p.is_dir()]
    print(f"화자 디렉토리 {len(spk_dirs)}개", flush=True)

    inv: Counter = Counter()
    n_utt = 0
    per_spk: Counter = Counter()
    utt_hash = hashlib.sha256()
    jsonl = out / f"mfa_kent_{args.tag}.jsonl.gz"
    with gzip.open(jsonl, "wt", encoding="utf-8") as fo, \
         ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for recs, c in ex.map(one_speaker_dir, spk_dirs):
            inv.update(c)
            for r in recs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                per_spk[r["spk"]] += 1
                utt_hash.update(r["utt"].encode())
                n_utt += 1
    print(f"utt {n_utt} → {jsonl} ({jsonl.stat().st_size/1e6:.1f} MB)", flush=True)

    (out / "phone_inventory_kent.json").write_text(json.dumps({
        "n_utt": n_utt,
        "n_phone_types": len(inv),
        "n_phone_tokens": sum(inv.values()),
        "phones": dict(inv.most_common()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "tag": args.tag,
        "n_utt": n_utt,
        "n_real_speakers": len(per_spk),
        "per_speaker": dict(sorted(per_spk.items())),
        "n_pseudo_speakers": len(spk_dirs),
        "n_phone_types": len(inv),
        # uttid 집합 해시 — 나중에 "같은 코퍼스인가"를 싸게 대조하기 위함
        "uttid_set_sha256": utt_hash.hexdigest(),
    }
    if args.manifest_extra:
        manifest.update(json.loads(args.manifest_extra))
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"phone 종류 {len(inv)} / 토큰 {sum(inv.values()):,}", flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
