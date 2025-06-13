#!/bin/bash

# Exit if any command fails
set -e

echo "🔧 Setting up your project environment..."

# 1. Create or update the conda environment
echo "📦 Creating Conda environment from environment.yml..."
conda env create -f environment.yml || conda env update -f environment.yml

# 2. Initialize submodules (RoMa)
echo "📂 Initializing Git submodules (e.g., RoMa)..."
git submodule update --init --recursive

# 3. Show next steps
ENV_NAME=$(head -n 1 environment.yml | cut -d ' ' -f2)

echo ""
echo "✅ Setup complete!"
echo ""
echo "👉 To activate your environment, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "👉 To use RoMa in your code, ensure you include this early in your scripts:"
cat <<EOF

   from utils import setup_paths
   from roma import roma_matching
EOF

echo ""
echo "🧠 Tip: You can add a line to 'utils/setup_paths.py' to insert RoMa into sys.path:"
echo "   sys.path.insert(0, str(Path(__file__).resolve().parent.parent /'external'/'roma'))"