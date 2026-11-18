#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

jspace_ssh <<'REMOTE'
set -euo pipefail
project_root=$1
cd "${project_root}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Refusing to pull: the remote worktree is dirty.' >&2
  git status --short >&2
  exit 3
fi
git pull --ff-only
git log -1 --oneline --decorate
REMOTE
