./activate_python.sh
conda env update -f environment.yaml
uv pip install -e .
uv pip install -r requirements.txt


# MeCab 설치 (solve error: python-mecab-ko)
# sudo apt-get update
# sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
