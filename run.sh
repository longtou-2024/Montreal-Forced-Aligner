#!/usr/bin/env bash

dictionary_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/korean_espeak.dict
acoustic_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip
g2p_model_path=/home/longtou.2024/mount/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip

#dictionary_path=/home/longtou.2024/mount_wds/tmp/mfa/korean.dict
#acoustic_model_path=/home/longtou.2024/mount_wds/tmp/mfa/acoustic/korean.zip

#dictionary_path=/home/longtou.2024/mount_wds/tmp/kss/mfa/korean.dict
#acoustic_model_path=/home/longtou.2024/mount_wds/tmp/kss/mfa/acoustic/korean.zip
#g2p_model_path=/home/longtou.2024/mount_wds/tmp/kss/mfa/g2p/korean.zip

indir=tmp
outdir=outdir
beam=10
retry_beam=40

#python montreal_forced_aligner/lt_mfa.py \
#    $dictionary_path \
#    $acoustic_model_path \
#    $g2p_model_path \
#    $indir \
#    $outdir \
#    --beam $beam \
#    --retry_beam $retry_beam

python montreal_forced_aligner/lt_mfa2.py \
    $dictionary_path \
    $acoustic_model_path \
    $g2p_model_path \
    $indir \
    $outdir \
    --beam $beam \
    --retry_beam $retry_beam
