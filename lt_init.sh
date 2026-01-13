# git clone espnet
git submodule update --init --recursive
# NOTE(longtou): comment "espnet/espnet2/__init__.py: from espnet import __version__"

# install mfa
./activate_python.sh
conda env update -f environment.yaml
uv pip install -e .

# MeCab 설치 (solve error: python-mecab-ko)
# sudo apt-get update
# sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

uv pip install -r requirements.txt
sudo apt install espeak-ng
