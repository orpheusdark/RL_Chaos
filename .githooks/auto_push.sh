#!/bin/bash
# Auto-push current branch to GitHub (origin) and HuggingFace (hf) on commit.

set -u

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ -z "$branch" ] || [ "$branch" = "HEAD" ]; then
	echo "Cannot auto-push in detached HEAD state."
	exit 1
fi

echo "Auto-pushing to both remotes..."

pushed_any=0

if git remote | grep -qx "origin"; then
	echo "Pushing $branch to origin..."
	if git push origin "$branch"; then
		pushed_any=1
	else
		echo "Warning: failed to push to origin"
	fi
else
	echo "Skipping origin push (remote not configured)."
fi

if git remote | grep -qx "hf"; then
	echo "Pushing snapshot of $branch to hf/main..."
	git fetch hf main --quiet || true
	tree="$(git rev-parse HEAD^{tree})"
	parent=""
	if git rev-parse --verify hf/main >/dev/null 2>&1; then
		parent="$(git rev-parse hf/main)"
	fi

	msg="HF sync snapshot from $branch at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	if [ -n "$parent" ]; then
		snapshot_commit="$(printf "%s" "$msg" | git commit-tree "$tree" -p "$parent")"
	else
		snapshot_commit="$(printf "%s" "$msg" | git commit-tree "$tree")"
	fi

	if git push hf "$snapshot_commit:main"; then
		pushed_any=1
	else
		echo "Warning: failed to push to hf"
	fi
else
	echo "Skipping hf push (remote not configured)."
fi

if [ "$pushed_any" -eq 0 ]; then
	echo "Auto-push failed: no remote push succeeded."
	exit 1
fi

echo "Auto-push complete"
