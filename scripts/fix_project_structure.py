#!/usr/bin/env python3
"""
Fix project structure and imports
Creates missing directories and __init__.py files
"""

import os
from pathlib import Path


def fix_project_structure():
    """Fix common project structure issues"""
    print("🔧 Fixing project structure...")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    print(f"📂 Project root: {project_root}")

    # Create missing directories
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "src",
        "src/data_collection",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/utils",
        "models/saved_models",
        "models/checkpoints",
        "results/figures",
        "results/reports",
        "results/metrics",
        "notebooks",
        "scripts",
        "docs"
    ]

    print("\n📁 Creating missing directories...")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
        else:
            print(f"✓ Exists: {dir_path}")

    # Create missing __init__.py files
    init_dirs = [
        "src",
        "src/data_collection",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/utils"
    ]

    print("\n📄 Creating missing __init__.py files...")
    for dir_path in init_dirs:
        init_file = project_root / dir_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✅ Created: {dir_path}/__init__.py")
        else:
            print(f"✓ Exists: {dir_path}/__init__.py")

    # Check for main files
    main_files = [
        "config.py",
        "requirements.txt",
        "src/data_collection/cafe_bazaar_scraper.py",
        "src/utils/gpt_labeler.py"
    ]

    print("\n📋 Checking main files...")
    missing_files = []
    for file_path in main_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n⚠️  Missing files detected:")
        for file_path in missing_files:
            print(f"   {file_path}")
        print("💡 Please make sure you've created all the necessary files.")
        return False

    print("\n✅ Project structure fixed successfully!")
    return True


def create_gitignore():
    """Create .gitignore file"""
    project_root = Path(__file__).parent.parent
    gitignore_path = project_root / ".gitignore"

    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.json
data/raw/*.csv
data/processed/*.csv
*.log

# Model files
models/saved_models/*
models/checkpoints/*
!models/saved_models/.gitkeep
!models/checkpoints/.gitkeep

# Results
results/figures/*
results/reports/*
results/metrics/*
!results/figures/.gitkeep
!results/reports/.gitkeep
!results/metrics/.gitkeep

# OS
.DS_Store
Thumbs.db

# API Keys
.env
.env.local
.env.production
"""

    if not gitignore_path.exists():
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"✅ Created .gitignore")
    else:
        print(f"✓ .gitignore already exists")


def create_gitkeep_files():
    """Create .gitkeep files for empty directories"""
    project_root = Path(__file__).parent.parent

    empty_dirs = [
        "models/saved_models",
        "models/checkpoints",
        "results/figures",
        "results/reports",
        "results/metrics"
    ]

    print("\n📌 Creating .gitkeep files...")
    for dir_path in empty_dirs:
        gitkeep_file = project_root / dir_path / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"✅ Created: {dir_path}/.gitkeep")
        else:
            print(f"✓ Exists: {dir_path}/.gitkeep")


def main():
    print("🔧 Persian Banking Sentiment Analysis - Project Structure Fix")
    print("=" * 70)

    # Fix project structure
    structure_ok = fix_project_structure()

    # Create .gitignore
    print("\n📄 Setting up .gitignore...")
    create_gitignore()

    # Create .gitkeep files
    create_gitkeep_files()

    if structure_ok:
        print("\n🎉 Project structure fix completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run: python scripts/test_imports.py")
        print("   2. If tests pass, run: python scripts/full_workflow.py --quick-test")
    else:
        print("\n⚠️  Some issues remain. Please check the missing files above.")

    print("=" * 70)


if __name__ == "__main__":
    main()
