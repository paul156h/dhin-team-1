#!/bin/bash

# Build script for generate_messages executable
# This script builds a standalone executable using PyInstaller

set -e  # Exit on error

echo "Building generate_messages executable..."

# Change to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
source gan-env/bin/activate

# Clean previous builds
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build/
fi

if [ -d "dist" ]; then
    echo "Cleaning previous dist..."
    rm -rf dist/
fi

# Check if model exists, train if needed
if [ ! -f "src/models/simple_gan.pt" ]; then
    echo "Training GAN model (required for executable)..."
    python3 src/models/train_gan.py
fi

# Build executable
echo "Building with PyInstaller..."
pyinstaller --clean --noconfirm scripts/generate_messages.spec

# Test the executable
echo "Testing executable..."
./dist/generate_messages/generate_messages

echo ""
echo "Build complete! Executable location:"
echo "$(pwd)/dist/generate_messages/generate_messages"
echo ""
echo "To run the executable:"
echo "  cd $(pwd)"
echo "  ./dist/generate_messages/generate_messages"
echo ""
echo "Or copy the entire 'dist/generate_messages/' folder to another machine."