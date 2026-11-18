#!/usr/bin/env bash

set -euo pipefail

JSPACE_LOCAL_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  pwd
)"
JSPACE_CLUSTER_ENV="${JSPACE_CLUSTER_ENV:-${JSPACE_LOCAL_ROOT}/.cluster.env}"

if [[ ! -f "${JSPACE_CLUSTER_ENV}" ]]; then
  echo "Missing ${JSPACE_CLUSTER_ENV}." >&2
  echo "Copy .cluster.env.example to .cluster.env and set the SSH host." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${JSPACE_CLUSTER_ENV}"
set +a

required=(
  JSPACE_SSH_HOST
  JSPACE_REMOTE_PROJECT_ROOT
  JSPACE_REMOTE_VENV
  JSPACE_REMOTE_SCRATCH_ROOT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing ${name} in ${JSPACE_CLUSTER_ENV}." >&2
    exit 2
  fi
done

if [[ ! "${JSPACE_SSH_HOST}" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]]; then
  echo "Unsafe JSPACE_SSH_HOST value." >&2
  exit 2
fi
for path in \
  "${JSPACE_REMOTE_PROJECT_ROOT}" \
  "${JSPACE_REMOTE_VENV}" \
  "${JSPACE_REMOTE_SCRATCH_ROOT}"; do
  if [[ ! "${path}" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "Unsafe remote path: ${path}" >&2
    exit 2
  fi
done

JSPACE_SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

jspace_ssh() {
  ssh "${JSPACE_SSH_OPTIONS[@]}" "${JSPACE_SSH_HOST}" bash -s -- \
    "${JSPACE_REMOTE_PROJECT_ROOT}" \
    "${JSPACE_REMOTE_VENV}" \
    "${JSPACE_REMOTE_SCRATCH_ROOT}" \
    "$@"
}
