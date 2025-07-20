#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test all required imports"""
    print("🧪 Testing Python imports for the project...")
    print("=" * 50)

    # Add the project root to the path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Also add the src directory specifically
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    print(f"📂 Project root: {project_root}")
    print(f"📂 Source path: {src_path}")
    print(f"📂 Current working directory: {os.getcwd()}")

    # Test 1: Config import
    print("\n1️⃣ Testing config import...")
    try:
        from config import FILE_PATHS, CAFE_BAZAAR_CONFIG, OPENAI_LABELING_CONFIG
        print("✅ Config imported successfully")
        print(f"   Raw comments path: {FILE_PATHS['raw_comments']}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    # Test 2: Scraper import
    print("\n2️⃣ Testing scraper import...")
    try:
        from src.data_collection.cafe_bazaar_scraper import CafeBazaarScraper
        print("✅ Scraper imported successfully")
    except ImportError as e:
        print(f"❌ Scraper import failed: {e}")
        print("💡 Make sure cafe_bazaar_scraper.py exists in src/data_collection/")

    # Test 3: GPT Labeler import
    print("\n3️⃣ Testing GPT labeler import...")
    try:
        from src.utils.gpt_labeler import GPTSentimentLabeler
        print("✅ GPT labeler imported successfully")
    except ImportError as e:
        print(f"❌ GPT labeler import failed: {e}")
        print("💡 Make sure gpt_labeler.py exists in src/utils/")

    # Test 4: Required packages
    print("\n4️⃣ Testing required packages...")

    required_packages = [
        ("pandas", "pandas"),
        ("requests", "requests"),
        ("openai", "openai"),
        ("matplotlib", "matplotlib.pyplot"),
        ("seaborn", "seaborn")
    ]

    missing_packages = []

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False

    # Test 5: File structure
    print("\n5️⃣ Testing file structure...")

    required_files = [
        project_root / "config.py",
        project_root / "src" / "data_collection" / "cafe_bazaar_scraper.py",
        project_root / "src" / "utils" / "gpt_labeler.py",
        project_root / "scripts" / "run_scraper.py",
        project_root / "scripts" / "label_comments.py",
        project_root / "scripts" / "full_workflow.py"
    ]

    missing_files = []

    for file_path in required_files:
        if file_path.exists():
            print(f"✅ {file_path.name}")
        else:
            print(f"❌ {file_path.name} - NOT FOUND")
            missing_files.append(str(file_path))

    if missing_files:
        print(f"\n📁 Missing files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False

    # Test 6: Directory structure
    print("\n6️⃣ Testing directory structure...")

    required_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "src" / "data_collection",
        project_root / "src" / "utils",
        project_root / "scripts"
    ]

    missing_dirs = []

    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path.relative_to(project_root)}")
        else:
            print(f"❌ {dir_path.relative_to(project_root)} - NOT FOUND")
            missing_dirs.append(str(dir_path))

    if missing_dirs:
        print(f"\n📁 Missing directories:")
        for dir_path in missing_dirs:
            print(f"   {dir_path}")
        print("💡 Run: python scripts/setup_project.py")
        return False

    # Test 7: Environment variables
    print("\n7️⃣ Testing environment variables...")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        if openai_key.startswith('sk-'):
            print("✅ OPENAI_API_KEY is set and looks valid")
        else:
            print("⚠️  OPENAI_API_KEY is set but format looks incorrect")
    else:
        print("⚠️  OPENAI_API_KEY not set (needed for labeling)")
        print("💡 Set with: export OPENAI_API_KEY='sk-your-key-here'")

    print("\n" + "=" * 50)
    print("🎉 Import test completed!")
    return True


def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n🔧 Testing basic functionality...")
    print("=" * 50)

    try:
        # Test config access
        from config import FILE_PATHS
        print(f"✅ Config access: {len(FILE_PATHS)} file paths defined")

        # Test scraper initialization
        from src.data_collection.cafe_bazaar_scraper import CafeBazaarScraper
        scraper = CafeBazaarScraper()
        print("✅ Scraper initialization successful")

        # Test GPT labeler initialization (without API key)
        try:
            from src.utils.gpt_labeler import GPTSentimentLabeler
            # This might fail if no API key, which is expected
            print("✅ GPT labeler class imported successfully")
        except Exception as e:
            if "API key" in str(e):
                print("⚠️  GPT labeler needs API key (expected)")
            else:
                print(f"❌ GPT labeler error: {e}")

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    print("🧪 Persian Banking Sentiment Analysis - Import Test")
    print("=" * 60)

    # Run import tests
    imports_ok = test_imports()

    if imports_ok:
        # Run functionality tests
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n🎉 All tests passed! Your environment is ready.")
            print("\n📋 Next steps:")
            print("   1. Set OPENAI_API_KEY if not already set")
            print("   2. Run: python scripts/full_workflow.py --quick-test")
            print("   3. Or run individual scripts as needed")
        else:
            print("\n⚠️  Some functionality tests failed.")
    else:
        print("\n❌ Import tests failed. Please fix the issues above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
