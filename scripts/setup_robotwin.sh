#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${PROJECT_ROOT}/third_party/RoboTwin"
PATCH_FILE="${PROJECT_ROOT}/evaluation/robotwin/patches/rewam.patch"

mkdir -p "${PROJECT_ROOT}/third_party"
exec 9>"${PROJECT_ROOT}/third_party/.robotwin-setup.lock"
flock 9

# Check out the commit recorded by the parent repository, without forcing
# checkout over local changes. Assets symlinks are managed locally.
git -C "${PROJECT_ROOT}" submodule update --init --checkout -- third_party/RoboTwin

if git -C "${ROBOTWIN_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "RoboTwin patch is already applied; no changes needed."
elif git -C "${ROBOTWIN_ROOT}" apply --check "${PATCH_FILE}"; then
  git -C "${ROBOTWIN_ROOT}" apply "${PATCH_FILE}"
  echo "RoboTwin patch applied successfully."
else
  echo "RoboTwin has changes that conflict with the patch; stopped. Review the local diff before retrying." >&2
  exit 1
fi
