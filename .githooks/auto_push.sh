#!/bin/bash
# Auto-push to both GitHub (origin) and HuggingFace (hf) on commit

echo "Auto-pushing to both remotes..."

# Push to GitHub
git push origin main 2>&1 | grep -v "already up to date" || true

# Push to HuggingFace Space
git push hf main 2>&1 | grep -v "already up to date" || true

echo "Auto-push complete"
