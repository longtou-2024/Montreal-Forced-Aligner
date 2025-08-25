#!/usr/bin/env bash

dictionary_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict
acoustic_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip
g2p_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip
outdir=outdir
beam=10
retry_beam=40

shard_urls=$(ls /home/longtou.2024/mount/longtou/db/literature/wds_v2/*)

parallel --line-buffer -j 24 python lt_scripts/quality_assurance.py {} \
    $dictionary_path \
    $acoustic_model_path \
    $g2p_model_path \
    $outdir \
    --temporary_directory /home/longtou.2024/Documents/{%} \
    --beam $beam \
    --retry_beam $retry_beam \
    ::: $shard_urls
    
#python lt_scripts/quality_assurance.py /home/longtou.2024/mount/longtou/db/literature/wds_v2/shard-000000.tar \
#    $dictionary_path \
#    $acoustic_model_path \
#    $g2p_model_path \
#    $outdir \
#    --temporary_directory /home/longtou.2024/Documents/1 \
#    --beam $beam \
#    --retry_beam $retry_beam
