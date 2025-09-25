#!/usr/bin/env python3
"""
Setup script for dacapo-monai package.
Install with: pip install -e .
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def install_package():
    """Install the dacapo-monai package in development mode."""
    print("🚀 INSTALLING DACAPO-MONAI")
    print("=" * 30)

    if not check_python_version():
        return False

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    print(f"📁 Working directory: {script_dir}")

    # Install in development mode
    success = run_command(
        f'"{sys.executable}" -m pip install -e .',
        "Installing dacapo-monai in development mode...",
    )

    if not success:
        print("❌ Installation failed!")
        return False

    # Test the installation
    print("\n🧪 TESTING INSTALLATION")
    print("=" * 20)

    try:
        import dacapo_monai

        print("✅ dacapo_monai import successful")

        from dacapo_monai import PipelineDataset

        print("✅ PipelineDataset import successful")

        from dacapo_monai.transforms import SSLTransforms

        print("✅ Transform presets import successful")

        from dacapo_monai.utils import MonaiToDacapoAdapter

        print("✅ Utilities import successful")

        print("\n🎉 INSTALLATION SUCCESSFUL!")
        print("=" * 25)
        print("✅ Package installed and ready to use")
        print("✅ All core modules working")
        print("✅ Transform presets available")
        print("✅ Utilities functional")

        print("\n📚 NEXT STEPS:")
        print("1. Try the examples:")
        print("   python examples/basic_medical_imaging.py")
        print("   python examples/ssl_contrastive_learning.py")
        print("   python examples/migration_example.py")
        print("")
        print("2. Read the documentation:")
        print("   cat README.md")
        print("")
        print("3. Start using in your code:")
        print("   from dacapo_monai import iterable_dataset")
        print("   from dacapo_monai.transforms import SSLTransforms")

        return True

    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        print("💡 The package was installed but imports are failing.")
        print("   This might be due to missing dependencies.")
        return False


def install_optional_dependencies():
    """Install optional dependencies for full functionality."""
    print("\n🔧 INSTALLING OPTIONAL DEPENDENCIES")
    print("=" * 35)

    optional_packages = [
        ("pytest", "For running tests"),
        ("jupyter", "For notebook examples"),
        ("matplotlib", "For visualization"),
        ("tqdm", "For progress bars"),
    ]

    for package, description in optional_packages:
        success = run_command(
            f'"{sys.executable}" -m pip install {package}',
            f"Installing {package} - {description}",
        )
        if not success:
            print(f"⚠️  Failed to install {package} (optional)")


def main():
    """Main setup function."""
    print("🔧 DACAPO-MONAI SETUP")
    print("=" * 20)

    # Install the package
    if not install_package():
        print("\n❌ Setup failed!")
        sys.exit(1)

    # Install optional dependencies
    install_optional_dependencies()

    print("\n✨ SETUP COMPLETE!")
    print("=" * 15)
    print("DaCapo-MONAI is ready to use! 🚀")


if __name__ == "__main__":
    main()
