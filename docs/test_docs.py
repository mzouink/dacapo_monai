#!/usr/bin/env python3
"""
Documentation testing and validation script for DaCapo-MONAI.
This script validates that the documentation builds correctly and all links work.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


def run_command(cmd: str, cwd: Optional[str] = None) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_documentation_structure(docs_dir: Path) -> List[str]:
    """Check that required documentation files exist."""
    issues = []

    required_files = [
        "conf.py",
        "index.md",
        "getting_started/index.md",
        "api_reference/index.md",
        "examples/index.md",
        "user_guide/index.md",
        "developer_guide/index.md",
        "changelog.md",
        "_static/custom.css",
        "_static/custom.js",
        "Makefile",
        "requirements.txt",
    ]

    for file_path in required_files:
        full_path = docs_dir / file_path
        if not full_path.exists():
            issues.append(f"❌ Missing required file: {file_path}")
        else:
            print(f"✅ Found: {file_path}")

    return issues


def test_documentation_build(docs_dir: Path) -> Tuple[bool, List[str]]:
    """Test that documentation builds without errors."""
    issues = []

    print("🔨 Testing documentation build...")

    # Clean any existing builds
    build_dir = docs_dir / "_build"
    if build_dir.exists():
        success, _, stderr = run_command(f"rm -rf {build_dir}", cwd=str(docs_dir))
        if not success:
            issues.append(f"❌ Failed to clean build directory: {stderr}")
            return False, issues

    # Install documentation dependencies
    print("📦 Installing documentation dependencies...")
    success, stdout, stderr = run_command(
        "pip install sphinx sphinx-design furo myst-parser sphinx-copybutton sphinxcontrib-mermaid",
        cwd=str(docs_dir),
    )

    if not success:
        issues.append(f"❌ Failed to install dependencies: {stderr}")
        return False, issues

    # Build HTML documentation
    print("🔨 Building HTML documentation...")
    success, stdout, stderr = run_command(
        "sphinx-build -b html . _build/html -W", cwd=str(docs_dir)
    )

    if not success:
        issues.append(f"❌ HTML build failed: {stderr}")
        return False, issues

    # Check that HTML files were generated
    html_dir = build_dir / "html"
    if not html_dir.exists():
        issues.append("❌ HTML build directory not created")
        return False, issues

    # Check for index.html
    index_file = html_dir / "index.html"
    if not index_file.exists():
        issues.append("❌ index.html not generated")
        return False, issues

    print("✅ HTML documentation built successfully!")

    # Count generated files
    html_files = list(html_dir.rglob("*.html"))
    print(f"📄 Generated {len(html_files)} HTML files")

    return True, issues


def test_link_checking(docs_dir: Path) -> Tuple[bool, List[str]]:
    """Test that all links in documentation are valid."""
    issues = []

    print("🔗 Testing documentation links...")

    success, stdout, stderr = run_command(
        "sphinx-build -b linkcheck . _build/linkcheck", cwd=str(docs_dir)
    )

    # Link checking might have warnings but not fail completely
    if "broken" in stderr.lower() or "error" in stderr.lower():
        issues.append(f"⚠️ Link check found issues: {stderr}")

    print("✅ Link checking completed!")
    return True, issues


def validate_sphinx_config(docs_dir: Path) -> List[str]:
    """Validate Sphinx configuration."""
    issues = []

    conf_file = docs_dir / "conf.py"
    if not conf_file.exists():
        issues.append("❌ conf.py not found")
        return issues

    # Read and basic validation of conf.py
    try:
        with open(conf_file, "r") as f:
            conf_content = f.read()

        # Check for required settings
        required_settings = [
            "project",
            "extensions",
            "html_theme",
            "html_static_path",
        ]

        for setting in required_settings:
            if setting not in conf_content:
                issues.append(f"⚠️ Missing Sphinx setting: {setting}")
            else:
                print(f"✅ Found Sphinx setting: {setting}")

        # Check for common extensions
        recommended_extensions = [
            "sphinx.ext.autodoc",
            "sphinx.ext.napoleon",
            "myst_parser",
        ]

        for ext in recommended_extensions:
            if ext in conf_content:
                print(f"✅ Using extension: {ext}")
            else:
                issues.append(f"💡 Consider adding extension: {ext}")

    except Exception as e:
        issues.append(f"❌ Error reading conf.py: {e}")

    return issues


def generate_documentation_report(docs_dir: Path):
    """Generate a comprehensive documentation report."""
    print("📊 DOCUMENTATION VALIDATION REPORT")
    print("=" * 50)

    total_issues = []

    # Structure check
    print("\n🏗️ STRUCTURE CHECK")
    print("-" * 20)
    structure_issues = check_documentation_structure(docs_dir)
    total_issues.extend(structure_issues)

    if not structure_issues:
        print("✅ All required files present!")

    # Config validation
    print("\n⚙️ CONFIGURATION CHECK")
    print("-" * 25)
    config_issues = validate_sphinx_config(docs_dir)
    total_issues.extend(config_issues)

    # Build test
    print("\n🔨 BUILD TEST")
    print("-" * 12)
    build_success, build_issues = test_documentation_build(docs_dir)
    total_issues.extend(build_issues)

    # Link check (only if build succeeded)
    if build_success:
        print("\n🔗 LINK CHECK")
        print("-" * 12)
        link_success, link_issues = test_link_checking(docs_dir)
        total_issues.extend(link_issues)

    # Summary
    print("\n📋 SUMMARY")
    print("-" * 10)
    print(f"Total issues found: {len(total_issues)}")

    if total_issues:
        print("\n⚠️ ISSUES TO ADDRESS:")
        for issue in total_issues:
            print(f"  {issue}")
    else:
        print("🎉 All documentation checks passed!")

    # Build info
    build_dir = docs_dir / "_build" / "html"
    if build_dir.exists():
        html_files = list(build_dir.rglob("*.html"))
        css_files = list(build_dir.rglob("*.css"))
        js_files = list(build_dir.rglob("*.js"))

        print(f"\n📊 BUILD STATISTICS:")
        print(f"  📄 HTML files: {len(html_files)}")
        print(f"  🎨 CSS files: {len(css_files)}")
        print(f"  📜 JavaScript files: {len(js_files)}")

        # Calculate size
        try:
            total_size = sum(
                f.stat().st_size for f in build_dir.rglob("*") if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            print(f"  📦 Total size: {size_mb:.2f} MB")
        except:
            print(f"  📦 Total size: Unable to calculate")

    print(f"\n🎯 NEXT STEPS:")
    if total_issues:
        print("  1. Address the issues listed above")
        print("  2. Re-run this script to verify fixes")
        print("  3. Test documentation locally with 'make livehtml'")
    else:
        print("  1. Documentation is ready!")
        print("  2. Test locally with 'make livehtml'")
        print("  3. Deploy with CI/CD pipeline")

    return len(total_issues) == 0


def main():
    """Main function."""
    # Find docs directory
    current_dir = Path(__file__).parent
    docs_dir = current_dir

    if not (docs_dir / "conf.py").exists():
        # Try parent directory structure
        potential_docs = current_dir.parent / "docs"
        if potential_docs.exists():
            docs_dir = potential_docs
        else:
            print("❌ Could not find documentation directory with conf.py")
            sys.exit(1)

    print(f"📁 Using documentation directory: {docs_dir}")

    # Run comprehensive validation
    success = generate_documentation_report(docs_dir)

    if success:
        print("\n🎉 Documentation validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Documentation validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
