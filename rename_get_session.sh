#!/bin/bash
# Script to globally rename get_session to get_or_create_session

set -e  # Exit on any error

echo "Starting global rename: get_session -> get_or_create_session"

# Find all Python files (excluding .git and __pycache__)
find . -name "*.py" \
  -not -path "./.git/*" \
  -not -path "./__pycache__/*" \
  -not -path "*/__pycache__/*" \
  -not -path "./.venv/*" \
  -exec sed -i 's/def get_session(/def get_or_create_session(/g' {} \; \
  -exec sed -i 's/\.get_session(/\.get_or_create_session(/g' {} \; \
  -exec sed -i 's/get_session(/get_or_create_session(/g' {} \;

# Find all markdown files for documentation
find . -name "*.md" \
  -not -path "./.git/*" \
  -exec sed -i 's/get_session(/get_or_create_session(/g' {} \; \
  -exec sed -i 's/\.get_session(/\.get_or_create_session(/g' {} \;

echo "Global rename completed!"
echo ""
echo "Files modified:"
git status --porcelain | grep "^ M" | wc -l | xargs echo "  Python/MD files:"

echo ""
echo "Please review the changes before committing:"
echo "  git diff --name-only"
echo "  git diff"