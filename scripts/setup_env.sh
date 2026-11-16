#!/usr/bin/env bash
# Fetch the official ADAM-U model (PNDbotics pnd_mujoco) at the commit this
# benchmark was developed against, into external/pnd_mujoco (git-ignored).
set -euo pipefail

PIN="84873eb3c4afd584d468864ba41527a1c14fb688"
URL="https://github.com/pndbotics/pnd_mujoco.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/pnd_mujoco"

if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$ROOT/external"
    git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin || true
git -C "$DEST" checkout --quiet "$PIN"

echo "pnd_mujoco at $(git -C "$DEST" rev-parse HEAD)"
echo "scene:  $DEST/pnd_robots/adam_u/scene.xml"
