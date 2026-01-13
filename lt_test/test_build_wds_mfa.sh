

shard_url="/home/longtou.2024/mount/longtou/db/literature/wds_v3/shard-000000.tar"
mount_path="/home/longtou.2024/mount"
dict_path="${mount_path}/longtou/db/commbooks/mfa/espeak/korean_espeak.dict"
am_path="${mount_path}/longtou/db/commbooks/mfa/espeak/acoustic/korean_espeak.zip"
g2p_path="${mount_path}/longtou/db/commbooks/mfa/espeak/g2p/korean_espeak.zip"
temp_dir="tempdir"
gcs_url="gs://prod-ai-lab-speech-bucket/longtou/tmp/literature"
json_text_key="transcript"
outdir="wds_v3_mfa"

python lt_scripts/build_wds_mfa.py ${shard_url} $dict_path $am_path $g2p_path $outdir $gcs_url $json_text_key --temporary_directory $temp_dir
