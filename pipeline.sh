set -e
export HUQIN_DATA=${HUQIN_DATA:-data}
DATA=$HUQIN_DATA
PY=${PY:-python}
if [ -n "$ZIPS" ]; then
  echo "STAGE unzip $(date)"
  for v in left right; do
    unzip -q -n "$ZIPS/CCOM-HuQin-v2.0-videos-$v-SinglePT.zip" "*/Erhu-*" -d "$DATA/videos"
    unzip -q -n "$ZIPS/CCOM-HuQin-v2.0-videos-$v-Excerpts.zip" "*/Erhu-*" -d "$DATA/videos_excerpts"
  done
  unzip -q -n "$ZIPS/CCOM-HuQin-v2.0-videos-front-Excerpts.zip" "*/Erhu-*" -d "$DATA/videos_excerpts"
fi
echo "STAGE features $(date)"
$PY precompute_mels.py
for v in left right; do
  $PY extract_hand_keypoints.py "$DATA/videos/CCOM-HuQin-v2.0-videos-$v-SinglePT" \
    "$DATA/keypoints/CCOM-HuQin-v2.0-videos-$v-SinglePT" --workers 8 --manifest "$DATA/kp_manifest_$v.csv"
  $PY extract_hand_keypoints.py "$DATA/videos_excerpts/video-$v" \
    "$DATA/keypoints_excerpts/video-$v" --workers 8 --manifest "$DATA/kp_exc_manifest_$v.csv"
done
echo "STAGE train $(date)"
mkdir -p logs
run() {
  name=$1; shift
  [ -f "$DATA/runs/$name/report.json" ] && return
  $PY lbd_train.py "$@" > "logs/$name.log" 2>&1
  echo "DONE $name $(grep 'TEST macro' "logs/$name.log")"
}
for s in 42 43 44; do
  run audio_s$s --model audio --seed $s
  run fusion_s${s}_norm --model fusion --kp-norm --seed $s
  run kp_s$s --model kp --seed $s
  run kp_s${s}_norm --model kp --kp-norm --seed $s
  run fusion_s$s --model fusion --seed $s
done
for p in Erhu-1 Erhu-2 Erhu-3; do
  run audio_p$p --model audio --test-player $p
  run kp_p$p --model kp --test-player $p
  run kp_p${p}_norm --model kp --kp-norm --test-player $p
  run fusion_p$p --model fusion --test-player $p
  run fusion_p${p}_norm --model fusion --kp-norm --test-player $p
done
echo "STAGE demo $(date)"
$PY demo_excerpts.py --out "$DATA/demo/pred_min0.6.json"
echo "STAGE all_done $(date)"
