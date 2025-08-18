#!/usr/bin/env bash

# longtou.2024

dictionary_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict
acoustic_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip
g2p_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip

sound_file_path=tmp3/kss_1_0001.wav
text_file_path=tmp3/kss_1_0001.lab

mfa align_one \
    -j 1 \
    --clean \
    --output_format "json" \
    --g2p_model_path $g2p_model_path \
    $sound_file_path \
    $text_file_path \
    $dictionary_path \
    $acoustic_model_path \
    align.json
    
