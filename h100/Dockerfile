# longtou.2024
FROM gcr.io/deeplearning-platform-release/pytorch-cu121.2-3.py310

RUN apt update && apt install -y parallel file # webdataset cache_dir error
WORKDIR /root
ADD Montreal-Forced-Aligner /root/Montreal-Forced-Aligner
WORKDIR /root/Montreal-Forced-Aligner
RUN ./setup_miniforge.sh conda mfa 3.10
RUN  . ./activate_python.sh  && conda env update -f environment.yml && pip install -e .

CMD ["/bin/bash"]
