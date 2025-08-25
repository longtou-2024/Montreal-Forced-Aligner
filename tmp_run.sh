#!/usr/bin/env bash

dictionary_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict
acoustic_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip
g2p_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip
temp_dir=tempdir
outdir=outdir
gcs_url="gs://prod-ai-lab-speech-bucket/longtou/tmp/literature/wds_v2_mfa"
beam=10
retry_beam=40

. parse_options.sh

shard_urls=$(ls /home/longtou.2024/mount/longtou/db/literature/wds_v2/* | head -n4)

parallel --line-buffer -j 4 python lt_scripts/build_wds_mfa.py {} \
    $dictionary_path \
    $acoustic_model_path \
    $g2p_model_path \
    $outdir \
    $gcs_url \
    --temporary_directory $temp_dir/{%} \
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
