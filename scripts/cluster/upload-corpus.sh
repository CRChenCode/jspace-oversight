#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

local_corpus="${JSPACE_LOCAL_ROOT}/data/prompts/pilot.jsonl"
local_metadata="${JSPACE_LOCAL_ROOT}/data/prompts/pilot.metadata.json"
test -f "${local_corpus}"
test -f "${local_metadata}"

remote_directory="${JSPACE_REMOTE_SCRATCH_ROOT}/data/prompts"
ssh "${JSPACE_SSH_OPTIONS[@]}" "${JSPACE_SSH_HOST}" \
  mkdir -p "${remote_directory}"
scp "${JSPACE_SSH_OPTIONS[@]}" \
  "${local_corpus}" "${local_metadata}" \
  "${JSPACE_SSH_HOST}:${remote_directory}/"

local_corpus_sha=$(shasum -a 256 "${local_corpus}" | awk '{print $1}')
local_metadata_sha=$(shasum -a 256 "${local_metadata}" | awk '{print $1}')
jspace_ssh "${local_corpus_sha}" "${local_metadata_sha}" <<'REMOTE'
set -euo pipefail
project_root=$1
venv_path=$2
scratch_root=$3
expected_corpus_sha=$4
expected_metadata_sha=$5
remote_directory="${scratch_root}/data/prompts"
actual_corpus_sha=$(sha256sum "${remote_directory}/pilot.jsonl" | awk '{print $1}')
actual_metadata_sha=$(sha256sum "${remote_directory}/pilot.metadata.json" | awk '{print $1}')
test "${actual_corpus_sha}" = "${expected_corpus_sha}"
test "${actual_metadata_sha}" = "${expected_metadata_sha}"
printf 'corpus_sha256=%s\n' "${actual_corpus_sha}"
printf 'metadata_sha256=%s\n' "${actual_metadata_sha}"
REMOTE
