# longtou.2024
FROM gcr.io/deeplearning-platform-release/pytorch-cu121.2-3.py310

RUN apt update && apt install -y parallel file cmake espeak-ng # webdataset cache_dir error
WORKDIR /root
ADD Montreal-Forced-Aligner /root/Montreal-Forced-Aligner
WORKDIR /root/Montreal-Forced-Aligner
RUN ./setup_miniforge.sh conda mfa 3.11
# 인증 문제때문에 실행하기 어려우므로 직접 복사
#RUN git submodule update --init --recursive
RUN . ./activate_python.sh && pip install uv
# NOTE(longtou): MFA 는 setuptools_scm 으로 버전을 잡는데(pyproject `[tool.setuptools_scm]`),
# 빌드 context 에서 `.git`(114MB)을 빼면 버전 탐지에 실패한다
#   LookupError: setuptools-scm was unable to detect version for ...
# `.git` 을 이미지에 넣는 대신 버전을 명시한다 — 더 작고 git 상태와 무관하게 결정적이다.
# 값은 `montreal_forced_aligner/_version.py`(3.3.5.dev15+g9b0e4a199) 기준.
# ※ espnet 은 espnet/version.txt, ai-lab-tts-g2p 는 정적 version 이라 scm 무관.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=3.3.5.dev15

RUN  . ./activate_python.sh  && conda env update -f environment.yml && uv pip install -e .
RUN . ./activate_python.sh && uv pip install -r requirements.txt

# NOTE(longtou): espnet2/__init__.py 의 `from espnet import __version__` 는 이 레포
# 레이아웃(서브모듈 디렉토리명이 `espnet`)에서 깨질 수 있다. 원인은 미확정이라
# 무조건 패치하지 않고 **빌드 시 진단해서 필요할 때만** 적용한다. 둘 다 실패하면
# 빌드가 죽어야 한다 — 조용히 넘어가면 정렬 잡이 런타임에 깨진다.
RUN . ./activate_python.sh && \
    if python -c "import espnet2" 2>/dev/null; then \
        echo "[mfa] espnet2 import OK — __init__ 패치 불필요"; \
    else \
        echo "[mfa] espnet2 import 실패 → __init__ 패치 적용"; \
        sed -i 's|^from espnet import __version__|#&|' espnet/espnet2/__init__.py && \
        grep -q '^#from espnet import __version__' espnet/espnet2/__init__.py && \
        python -c "import espnet2" && echo "[mfa] 패치 후 import OK"; \
    fi

# kent_g2p (사내 ai-lab-tts-g2p) — espnet `g2p_kent2_mfa` 토크나이저가 물어 쓴다.
# [g2p] extra = python-mecab-ko (JVM 불필요; konlpy 는 kog2p-advanced 경로 전용).
RUN . ./activate_python.sh && uv pip install -e "./ai-lab-tts-g2p[g2p]"

# NOTE(longtou): kaldi `utils/validate_data_dir.sh` 는 UTF-8 로케일을
#   `locale -a | grep "C.UTF-8"` 로 찾고, 없으면 `L=en_US.UTF-8` 로 폴백한다.
#   그런데 이 이미지에는 en_US.UTF-8 이 없어서 grep 이 C 로케일로 떨어지고
#   **한글 전 줄이 non-printable 로 오탐**된다
#   ("text contains 500 lines with non-printable characters" — 데이터는 정상).
#   ※ glibc 는 `locale -a` 출력에서 이름을 `C.utf8` 로 정규화하므로 kaldi 의
#     `grep "C.UTF-8"` 는 **구조적으로 절대 매치되지 않는다**(localedef 로 C.UTF-8 을
#     만들어도 `C.utf8` 로 나온다). 그래서 폴백 대상인 en_US.UTF-8 을 실재하게 만든다.
#   실측: en_US.UTF-8 생성 후 오탐 0건. `--non-print` 로 검사를 끄지 않아도 된다.
#   ※ 일부러 늦은 레이어에 둔다 — 위 apt 레이어에 합치면 conda/kaldi 캐시가 다 날아간다.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        --no-install-recommends locales \
 && sed -i 's/^# *en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen \
 && locale-gen \
 && locale -a | grep -q "en_US" \
 && rm -rf /var/lib/apt/lists/*

# 배선 검증 — 여기서 죽으면 이미지가 잘못 구워진 것이다.
# ※ `cd /tmp` 필수. `python -c` 는 cwd 를 sys.path[0] 에 넣는데, 레포 루트에는 stdlib/
#   설치패키지를 가리는 이름들이 있다(`espnet/` 디렉토리, 루트 `__pycache__`) →
#   py3.11 의 `Blocked import of locale from current working directory` 로 죽는다.
#   같은 그림자 문제가 위 espnet2 `__init__` 패치가 필요한 이유이기도 하다.
#   실제 잡은 `python lt_scripts/...` 나 `mfa` 엔트리포인트로 돌아 sys.path[0] 이
#   레포 루트가 아니므로 영향받지 않는다.
RUN . ./activate_python.sh && cd /tmp && python -c "\
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer; \
t = PhonemeTokenizer('g2p_kent2_mfa'); \
o = t.text2tokens('한국어 음성 합성 모델.'); \
assert o == ['h0','aa','nf','k0','uu','k0','vv','|','xx','mf','s0','vv','ng','|','h0','aa','pf','ss','vv','ng','|','mm','oo','t0','ee','ll','.','|'], o; \
print('[mfa] g2p_kent2_mfa OK:', o)"

CMD ["/bin/bash"]
