#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Watch the configured Git remote and fast-forward the checked-out branch.
# Local modifications are never stashed, reset, or overwritten automatically.

readonly SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DEFAULT_REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

REPO_DIR="${DEFAULT_REPO_DIR}"
REMOTE="origin"
BRANCH="main"
POLL_INTERVAL_SECONDS="60"
RUN_ONCE="false"

log() {
  local -r level="$1"
  shift
  printf '[%s] %-5s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "$*" >&2
}

die() {
  log ERROR "$*"
  exit 1
}

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Watch a Git branch and apply remote updates using fast-forward-only merges.
The default target is origin/main in this repository:
https://github.com/khang1108/MLeCDanBGold.git

Options:
  --repo-dir PATH    Repository to update (default: repository containing script)
  --remote NAME      Git remote to watch (default: origin)
  --branch NAME      Checked-out branch to update (default: main)
  --interval SEC     Seconds between checks (default: 60)
  --once             Check once, update if safe, then exit
  -h, --help         Show this help

Safety:
  - A dirty worktree is never modified.
  - Diverged history is never merged or reset.
  - Updates must be fast-forward-only.

Examples:
  scripts/${SCRIPT_NAME}
  scripts/${SCRIPT_NAME} --interval 15
  scripts/${SCRIPT_NAME} --once

Stop continuous watching with Ctrl+C.
EOF
}

require_option_value() {
  local -r option="$1"
  local -r remaining="$2"
  [[ "${remaining}" -ge 2 ]] || die "${option} requires a value."
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --repo-dir)
        require_option_value "$1" "$#"
        REPO_DIR="$2"
        shift 2
        ;;
      --remote)
        require_option_value "$1" "$#"
        REMOTE="$2"
        shift 2
        ;;
      --branch)
        require_option_value "$1" "$#"
        BRANCH="$2"
        shift 2
        ;;
      --interval)
        require_option_value "$1" "$#"
        POLL_INTERVAL_SECONDS="$2"
        shift 2
        ;;
      --once)
        RUN_ONCE="true"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        [[ "$#" -eq 0 ]] || die "Positional arguments are not supported."
        ;;
      *)
        die "Unknown option: $1. Use --help."
        ;;
    esac
  done
}

validate_inputs() {
  command -v git >/dev/null 2>&1 || die "Required command not found: git"
  [[ "${POLL_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
    || die "--interval must be a positive integer."
  [[ -d "${REPO_DIR}" ]] || die "Repository directory not found: ${REPO_DIR}"

  REPO_DIR="$(cd -- "${REPO_DIR}" && pwd -P)"
  git -C "${REPO_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "Not a Git worktree: ${REPO_DIR}"
  git -C "${REPO_DIR}" remote get-url "${REMOTE}" >/dev/null 2>&1 \
    || die "Git remote not found: ${REMOTE}"
}

worktree_is_clean() {
  [[ -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=normal)" ]]
}

check_for_update() {
  local current_branch local_commit remote_commit remote_url

  current_branch="$(git -C "${REPO_DIR}" branch --show-current)"
  if [[ "${current_branch}" != "${BRANCH}" ]]; then
    log WARN "Checked-out branch is '${current_branch:-detached}', waiting for '${BRANCH}'."
    return 0
  fi

  remote_url="$(git -C "${REPO_DIR}" remote get-url "${REMOTE}")"
  log INFO "Checking ${REMOTE}/${BRANCH} (${remote_url})"

  # Disable interactive credential prompts so an unattended watcher cannot hang.
  if ! GIT_TERMINAL_PROMPT=0 git -C "${REPO_DIR}" fetch --quiet "${REMOTE}" "${BRANCH}"; then
    log WARN "Fetch failed; retrying on the next check."
    return 0
  fi

  local_commit="$(git -C "${REPO_DIR}" rev-parse HEAD)"
  remote_commit="$(git -C "${REPO_DIR}" rev-parse FETCH_HEAD)"

  if [[ "${local_commit}" == "${remote_commit}" ]]; then
    log INFO "Already up to date at ${local_commit:0:12}."
    return 0
  fi

  if git -C "${REPO_DIR}" merge-base --is-ancestor "${remote_commit}" "${local_commit}"; then
    log WARN "Local ${BRANCH} is ahead of ${REMOTE}/${BRANCH}; no update applied."
    return 0
  fi

  if ! git -C "${REPO_DIR}" merge-base --is-ancestor "${local_commit}" "${remote_commit}"; then
    log ERROR "Local and remote histories diverged; resolve them manually."
    return 0
  fi

  if ! worktree_is_clean; then
    log WARN "Remote update found, but the worktree has local changes; leaving it untouched."
    return 0
  fi

  log INFO "Fast-forwarding ${local_commit:0:12} -> ${remote_commit:0:12}."
  if git -C "${REPO_DIR}" merge --quiet --ff-only "${remote_commit}"; then
    log INFO "Updated ${BRANCH} to ${remote_commit:0:12}."
  else
    log ERROR "Fast-forward failed; repository was left for manual inspection."
  fi
}

handle_signal() {
  log INFO "Watcher stopped."
  exit 130
}

main() {
  parse_args "$@"
  validate_inputs

  trap handle_signal INT TERM
  log INFO "Watching ${REMOTE}/${BRANCH} every ${POLL_INTERVAL_SECONDS}s in ${REPO_DIR}."

  while true; do
    check_for_update
    [[ "${RUN_ONCE}" == "false" ]] || break
    sleep "${POLL_INTERVAL_SECONDS}"
  done
}

main "$@"
