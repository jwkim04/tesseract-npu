#!/usr/bin/env bash
# apply-patches.sh — apply third_party/patches/*.patch (in name order) to the pinned llama.cpp
# submodule. Idempotent: a patch that is already applied is skipped.
#
#   third_party/apply-patches.sh             apply what is missing
#   third_party/apply-patches.sh --status    only report applied / not applied
#   third_party/apply-patches.sh --reverse   un-apply (reverse order)
#
# The patches carry our llama.cpp changes until a fork exists (spec 규칙 4, docs/llama-pin.md).
# Regenerate 0001 after editing the hook inside the submodule with:
#   git -C third_party/llama.cpp add -N <new files> && git -C third_party/llama.cpp diff > third_party/patches/0001-ggml-tesseract.patch
#   git -C third_party/llama.cpp reset -q -- <new files>
set -euo pipefail

PIN=e3797b184d428c7a019362612b76971f60af6670

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo="$here/llama.cpp"
mode=apply
case "${1:-}" in
    "")         ;;
    --status)   mode=status ;;
    --reverse)  mode=reverse ;;
    -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
    *)          echo "usage: $0 [--status|--reverse]" >&2; exit 2 ;;
esac

if [ ! -e "$repo/.git" ]; then
    echo "error: $repo is not a checkout (run: git submodule update --init third_party/llama.cpp)" >&2
    exit 1
fi

head=$(git -C "$repo" rev-parse HEAD)
if [ "$head" != "$PIN" ]; then
    echo "warning: llama.cpp HEAD is $head, the patches are made for the pin $PIN (docs/llama-pin.md)" >&2
fi

shopt -s nullglob
patches=("$here"/patches/*.patch)
if [ ${#patches[@]} -eq 0 ]; then
    echo "no patches in $here/patches"
    exit 0
fi
if [ "$mode" = reverse ]; then
    reversed=()
    for (( i=${#patches[@]}-1; i>=0; i-- )); do reversed+=("${patches[$i]}"); done
    patches=("${reversed[@]}")
fi

rc=0
for p in "${patches[@]}"; do
    name=$(basename "$p")
    if git -C "$repo" apply --check "$p" 2>/dev/null; then
        state=missing
    elif git -C "$repo" apply --reverse --check "$p" 2>/dev/null; then
        state=applied
    else
        state=conflict
    fi
    case "$mode:$state" in
        apply:missing)     git -C "$repo" apply --whitespace=nowarn "$p"; echo "applied        $name" ;;
        apply:applied)     echo "already applied $name" ;;
        reverse:applied)   git -C "$repo" apply --reverse "$p"; echo "reversed       $name" ;;
        reverse:missing)   echo "not applied    $name" ;;
        status:applied)    echo "applied        $name" ;;
        status:missing)    echo "not applied    $name" ;;
        *:conflict)
            echo "error: $name neither applies cleanly nor is fully applied to $repo:" >&2
            git -C "$repo" apply --check -v "$p" >&2 || true
            rc=1
            [ "$mode" = status ] || exit 1
            ;;
    esac
done
exit $rc
