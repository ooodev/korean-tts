#!/usr/bin/env bash
# Build the phonemizer data from its two sources. Cloud only, CPU is enough.
#
#   build.sh <work-directory>
#
# Steps (each one is skipped when its output is already there):
#   1. fetch CMUdict and the Wikipedia Homograph Data at pinned commits
#   2. dictionary: letter-to-phoneme alignment, then CMUdict converted by rule
#   3. unknown-word model, trained on CMUdict (about an hour and a half on
#      4 CPU cores, 15 minutes on a small GPU with G2P_ARGS="--device cuda")
#   4. homograph classifiers
#
# Result: <work-directory>/built, about 6 MB. Point DIZIONE_DATA at it,
# or copy it to src/dizione/built.
set -euo pipefail
work="${1:?usage: build.sh <work-directory>}"
here="$(cd "$(dirname "$0")" && pwd)"
python="${PYTHON:-python}"
mkdir -p "$work/built"

[ -f "$work/data/cmudict/cmudict.dict" ] || bash "$here/fetch_sources.sh" "$work/data"
cp "$work/data/sources.json" "$work/built/sources.json"
# The built data is derived from both sources: their licences travel with it.
cp "$work/data/cmudict/LICENSE" "$work/built/LICENSE.cmudict"
cp "$work/data/homograph/LICENSE" "$work/built/LICENSE.homograph-data"

reuse=""
[ -f "$work/built/align.json" ] && reuse="--reuse-alignment"
"$python" "$here/build_lexicon.py" --cmudict "$work/data/cmudict/cmudict.dict" --output "$work/built" $reuse

if [ ! -f "$work/built/g2p.npz" ]; then
  # shellcheck disable=SC2086
  "$python" "$here/train_g2p.py" --cmudict "$work/data/cmudict/cmudict.dict" --output "$work/g2p" ${G2P_ARGS:-}
  cp "$work/g2p/g2p.npz" "$work/g2p/g2p.json" "$work/built/"
  cp "$work/g2p/report.json" "$work/built/g2p.report.json"
fi

"$python" "$here/train_homographs.py" --data "$work/data/homograph/data" --cmudict "$work/data/cmudict/cmudict.dict" \
  --built "$work/built" > "$work/built/homographs.log"
echo "built: $work/built"
ls -l "$work/built"
