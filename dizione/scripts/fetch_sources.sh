#!/usr/bin/env bash
# Download the two licence-clean sources of the phonemizer at pinned commits.
# Cloud only: nothing here is meant to run on the laptop.
#
#   fetch_sources.sh <output-directory>
#
# * CMUdict (Carnegie Mellon University, BSD-style licence): pronunciations.
# * Wikipedia Homograph Data (Google, Apache-2.0): labelled sentences used to
#   train the homograph rules. Only the trained weights are shipped, never the
#   sentences.
set -euo pipefail
out="${1:?usage: fetch_sources.sh <output-directory>}"
CMUDICT_COMMIT="74790861f652b15e4ac49015a90074ad62a27690"       # 2025-10-24
HOMOGRAPH_COMMIT="8f008f021e88f8b71118a27ae655f1f3121162bc"     # 2021-10-14
mkdir -p "$out/cmudict" "$out/homograph"
for name in cmudict.dict cmudict.phones cmudict.symbols cmudict.vp LICENSE README; do
  curl -fsSL "https://raw.githubusercontent.com/cmusphinx/cmudict/$CMUDICT_COMMIT/$name" -o "$out/cmudict/$name"
done
curl -fsSL "https://github.com/google-research-datasets/WikipediaHomographData/archive/$HOMOGRAPH_COMMIT.tar.gz" \
  | tar -xz -C "$out/homograph" --strip-components=1
printf '{"cmudict_commit": "%s", "cmudict_sha256": "%s", "homograph_commit": "%s"}\n' \
  "$CMUDICT_COMMIT" "$(sha256sum "$out/cmudict/cmudict.dict" | cut -d' ' -f1)" "$HOMOGRAPH_COMMIT" > "$out/sources.json"
cat "$out/sources.json"
