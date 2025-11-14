# longtou.2024
FROM gcr.io/deeplearning-platform-release/pytorch-cu121.2-3.py310

RUN apt update && apt install -y parallel file espeak-ng # webdataset cache_dir error
WORKDIR /root
ADD Montreal-Forced-Aligner /root/Montreal-Forced-Aligner
WORKDIR /root/Montreal-Forced-Aligner
RUN ./setup_miniforge.sh conda mfa 3.11
RUN git submodule update --init --recursive
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN  . ./activate_python.sh  && conda env update -f environment.yml && uv pip install -e .
RUN . ./activate_python.sh && uv pip install -r requirements.txt

CMD ["/bin/bash"]
