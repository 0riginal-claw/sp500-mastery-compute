#!/usr/bin/env bash
# sensitive-path-block.sh — PreToolUse hook that re-implements the
# permissions.deny list as a hook, because under
# --permission-mode bypassPermissions the deny list is skipped.
#
# Triggered for: Read | Write | Edit | Bash (registered via matcher in settings.json).
# Input: JSON on stdin per Claude Code hook protocol.
# Block: exit 2 with reason on stderr.
# Allow: exit 0.
#
# Patterns mirror the 40 deny rules in
# AI-Tools/ClaudeCode/config/settings.json -> permissions.deny.
# Adjust here when the deny list changes; keep both in sync.

set -u
LC_ALL=C

# --- read & parse input ----------------------------------------------------
INPUT=$(cat)
if ! command -v jq >/dev/null 2>&1; then
  # jq missing → fail-open with a warning rather than break the session.
  echo "sensitive-path-block: jq not found, allowing (install jq to enforce)" >&2
  exit 0
fi

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')
TOOL_FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')
TOOL_CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')

# Expand ~ and $HOME for matching.
# NOTE: comparisons with quoted "~/..." use a glob equality that bash
# tilde-expands in the pattern position, which makes the check unreliable.
# Use substring-removal instead.
expand_path() {
  local p="$1"
  # leading "~/" or bare "~"
  if [[ "$p" = "~" ]]; then
    p="$HOME"
  elif [[ "${p:0:2}" = "~/" ]]; then
    p="$HOME/${p:2}"
  fi
  # Replace $HOME literal
  p="${p//\$HOME/$HOME}"
  printf '%s' "$p"
}

block() {
  local reason="$1"
  local path="$2"
  echo "BLOCKED by sensitive-path-block hook: $reason" >&2
  echo "Path/command: $path" >&2
  echo "Note: bypassPermissions mode skips deny rules; this hook enforces them." >&2
  exit 2
}

# --- sensitive-path matcher ------------------------------------------------
# Returns 0 (match=block) or 1 (no match). Operates on an absolute path.
matches_sensitive_path() {
  local p="$1"
  # Normalize: collapse any "/./" and trailing "/" for cleaner matching.
  p="${p//\/.\///}"

  # Read-deny equivalents (also applied to Write/Edit since exposing or
  # mutating these is equally bad):
  #   ~/.ssh/**, ~/.aws/**, ~/.gnupg/**, ~/.kube/**, ~/.azure/**,
  #   ~/.config/gcloud/**, ~/.docker/config.json
  #   ~/.npmrc, ~/.pypirc, ~/.netrc, ~/.git-credentials, ~/.electrum/**
  #   ~/.claude/settings.json, ~/.claude/settings.local.json
  #   ~/Library/Application Support/{MetaMask,Exodus,Phantom,Solflare,Electrum}/**
  #   **/.env, **/.env.*, **/.envrc, **/secrets/**, **/*.pem, **/*.key,
  #   **/*secret*, **/*credential*
  case "$p" in
    "$HOME"/.ssh/*|"$HOME"/.ssh)              return 0 ;;
    "$HOME"/.aws/*|"$HOME"/.aws)              return 0 ;;
    "$HOME"/.gnupg/*|"$HOME"/.gnupg)          return 0 ;;
    "$HOME"/.kube/*|"$HOME"/.kube)            return 0 ;;
    "$HOME"/.azure/*|"$HOME"/.azure)          return 0 ;;
    "$HOME"/.config/gcloud/*|"$HOME"/.config/gcloud) return 0 ;;
    "$HOME"/.docker/config.json)              return 0 ;;
    "$HOME"/.npmrc)                           return 0 ;;
    "$HOME"/.pypirc)                          return 0 ;;
    "$HOME"/.netrc)                           return 0 ;;
    "$HOME"/.git-credentials)                 return 0 ;;
    "$HOME"/.electrum/*|"$HOME"/.electrum)    return 0 ;;
    "$HOME"/.claude/settings.json)            return 0 ;;
    "$HOME"/.claude/settings.local.json)      return 0 ;;
    "$HOME"/Library/Application\ Support/MetaMask/*)  return 0 ;;
    "$HOME"/Library/Application\ Support/Exodus/*)    return 0 ;;
    "$HOME"/Library/Application\ Support/Phantom/*)   return 0 ;;
    "$HOME"/Library/Application\ Support/Solflare/*)  return 0 ;;
    "$HOME"/Library/Application\ Support/Electrum/*)  return 0 ;;
    # Shell rc / profile (Edit-deny equivalents — block Read/Write/Edit alike)
    "$HOME"/.bashrc|"$HOME"/.zshrc|"$HOME"/.bash_profile|"$HOME"/.zprofile|"$HOME"/.profile) return 0 ;;
  esac

  # Filename-based patterns (any directory).
  local base="${p##*/}"
  case "$base" in
    .env|.envrc)                              return 0 ;;
    .env.*)                                   return 0 ;;
    *.pem|*.key)                              return 0 ;;
  esac
  # Substring patterns on basename
  if [[ "$base" == *secret* ]] || [[ "$base" == *credential* ]]; then
    return 0
  fi
  # Directory-segment pattern: any /secrets/ directory.
  if [[ "$p" == */secrets/* ]] || [[ "$p" == */secrets ]]; then
    return 0
  fi

  return 1
}

# --- Read / Write / Edit dispatch -----------------------------------------
case "$TOOL_NAME" in
  Read|Write|Edit|NotebookEdit)
    if [[ -n "$TOOL_FILE_PATH" ]]; then
      ABS=$(expand_path "$TOOL_FILE_PATH")
      if matches_sensitive_path "$ABS"; then
        block "sensitive path access via $TOOL_NAME" "$ABS"
      fi
    fi
    ;;
  Bash)
    # Inspect the command string for read/write/exfil patterns that touch
    # the same sensitive paths. Catches the documented bypass:
    #   `bash cat ~/.ssh/id_rsa`
    # We block when a sensitive-path token appears as an argument to
    # cat/less/more/head/tail/grep/awk/sed/cp/mv/scp/rsync/tar/zip/curl/
    # base64/xxd/strings/file/openssl/gpg/age/ssh-keygen/git, OR when a
    # heredoc/pipe targets it.
    if [[ -n "$TOOL_CMD" ]]; then
      # ---- Pre-filter (tightened 2026-05-16) ------------------------------
      # Goal: only fire when a string in the command looks like a path or
      # token that points at a sensitive file/dir, not when a sensitive
      # *keyword* appears as English in a comment / echo string / unrelated
      # filename (e.g. "user.profile.json", "top secret notes").
      #
      # The pre-filter is built from three classes joined by `|`:
      #
      #   A) Sensitive directory roots — must follow `/`, `~`, or be at the
      #      start of a token; must end at a path boundary (`/`, end, quote,
      #      space, `=`):
      #        ~/.ssh, /.aws, /.gnupg, /.kube, /.azure, /.config/gcloud,
      #        /.electrum, /.claude/settings.json, /.docker/config,
      #        ~/Library/Application Support/{MetaMask,Exodus,...}
      #
      #   B) Sensitive single-file dotfiles — same token-boundary rule on
      #      both sides, so `.env` matches `.env`, `./.env`, `.env.prod`,
      #      `"$HOME/.env"` but NOT `user.profile.json`, `environment.json`,
      #      `monkey.txt`:
      #        .env, .envrc, .npmrc, .pypirc, .netrc, .git-credentials,
      #        .bashrc, .zshrc, .bash_profile, .zprofile, .profile,
      #        *.pem, *.key
      #
      #   C) `secrets/` directory segment, or `secret`/`credential` only
      #      when embedded in a path-like token (i.e. has a `/` somewhere
      #      in the same token, OR is preceded by `_`/`.`/`-` AND followed
      #      by `_`/`.`/`-`/`/` — i.e. it's the basename of a path).
      #      This prevents `echo 'secret stuff'` from firing, while still
      #      catching `/tmp/my_secret.txt` and `cat $HOME/api-credential-cache`.
      #
      # Token-boundary helpers (POSIX ERE — no \b in BSD grep):
      #   LB = lookbehind-ish boundary on the left   = (^|[/[:space:]'"\$=:])
      #   RB = lookahead-ish boundary on the right   = ($|[/[:space:]'"\$=:.])
      LB='(^|[/[:space:]'"'"'"\$=:])'
      RB='($|[[:space:]'"'"'"=:]|/|\.[^/]*$)'
      # Note: RB ends at path-boundary or "this is the last extension on
      # the filename" — so `.pem` matches `cert.pem` and `cert.pem.bak`
      # (the .pem token is bordered by `.bak`), but `.profile` does NOT
      # match `user.profile.json` because the `.profile` is followed by
      # `.json` which is NOT end-of-string AND NOT bordered as basename.
      # We instead define a separate RB_EXT for extensions so the
      # ambiguity is explicit:
      RB_END='($|[[:space:]'"'"'"=:|;)&]|/)'

      # --- Class A: sensitive directory roots ---
      RE_DIRS="${LB}(\\.ssh|\\.aws|\\.gnupg|\\.kube|\\.azure|\\.electrum|\\.config/gcloud|\\.docker/config(\\.json)?|\\.claude/settings(\\.local)?\\.json)${RB_END}"
      # GUI wallet dirs (require Library/Application Support context)
      RE_WALLETS='Library/Application Support/(MetaMask|Exodus|Phantom|Solflare|Electrum)(/|$)'

      # --- Class B: sensitive single-file dotfiles (token-bounded) ---
      # Match dotfile basenames: must start at LB, end at RB_END or
      # at a single-segment extension (e.g. ".env.prod" still flags).
      RE_DOTFILES="${LB}(\\.env(\\.[A-Za-z0-9._-]+)?|\\.envrc|\\.npmrc|\\.pypirc|\\.netrc|\\.git-credentials|\\.bashrc|\\.zshrc|\\.bash_profile|\\.zprofile|\\.profile)${RB_END}"

      # --- Class B2: .pem / .key file extensions (must be the final
      # extension of a basename — bordered by `/`, space, quote, or EOL) ---
      RE_KEYEXT='[/[:space:]'"'"'"=]*[^/[:space:]'"'"'"=]+\.(pem|key)('"$RB_END"')'
      # Simpler/cheaper: any `*.pem` or `*.key` followed by RB_END.
      RE_KEYEXT='\.(pem|key)'"$RB_END"

      # --- Class C: secrets / generic secret-credential tokens-in-paths ---
      # `secrets/` directory segment anywhere:
      RE_SECRETS_DIR='/secrets/'
      # `secret` or `credential` as part of a path basename (token must
      # contain `/`, `_`, `-`, or `.` adjacent to the keyword AND be in
      # a path-like context — i.e. not stand-alone English).
      # Heuristic: keyword is wedged on at least one side by `[/_.-]` AND
      # the surrounding word has a `/` OR a `.` ext nearby.
      RE_SECRET_TOKEN='([/_.-]secret|secret[/_.-])[A-Za-z0-9._/-]*'
      RE_CRED_TOKEN='([/_.-]credential|credential[/_.-])[A-Za-z0-9._/-]*'

      PREFILTER="(${RE_DIRS})|(${RE_WALLETS})|(${RE_DOTFILES})|(${RE_KEYEXT})|(${RE_SECRETS_DIR})|(${RE_SECRET_TOKEN})|(${RE_CRED_TOKEN})"

      if printf '%s' "$TOOL_CMD" | grep -qEi "$PREFILTER"; then
        # Now require a "reads/exposes/mutates" verb in the same command.
        if printf '%s' "$TOOL_CMD" | grep -qEi '(^|[[:space:]/;|&`(])(cat|less|more|head|tail|grep|awk|sed|cp|mv|scp|rsync|tar|zip|gzip|bzip2|xz|curl|wget|nc|ncat|base64|xxd|strings|file|openssl|gpg|age|ssh-keygen|git[[:space:]]+(add|commit|diff|log|show)|tee|dd|cmp|diff|md5sum|sha[0-9]+sum|shasum|stat|readlink|find|ls|chmod|chown|rm|touch|truncate|install|env|printf|echo|python|python3|node|ruby|perl|php|jq|source|bash|sh|zsh|exec|eval)([[:space:]]|$)'; then
          block "sensitive path access via Bash command" "$TOOL_CMD"
        fi
        # Redirection-target into a sensitive path is a block even without
        # a verb. Same tightened pattern set.
        if printf '%s' "$TOOL_CMD" | grep -qE '>>?[[:space:]]*[^[:space:]|;&)]*('"$PREFILTER"')'; then
          block "sensitive path write via Bash redirection" "$TOOL_CMD"
        fi
      fi
    fi
    ;;
esac

exit 0
