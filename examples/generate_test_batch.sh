#!/bin/bash
# Example: Generate 50 diverse patients for testing

cd "$(dirname "$0")/.."

echo "Generating 50 diverse test patients..."
python3 src/utils/generate_messages.py

echo "Files generated in outputs/"
echo "Generated files:"
ls -la outputs/ | head -10