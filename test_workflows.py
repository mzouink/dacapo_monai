#!/usr/bin/env python3
"""
Test GitHub Actions workflows locally.
This script validates that the workflows are correctly configured.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any


def validate_workflow_file(workflow_path: Path) -> List[str]:
    """Validate a single workflow file."""
    issues = []

    try:
        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)
    except yaml.YAMLError as e:
        issues.append(f"❌ Invalid YAML in {workflow_path.name}: {e}")
        return issues
    except Exception as e:
        issues.append(f"❌ Error reading {workflow_path.name}: {e}")
        return issues

    # Check required fields
    required_fields = ["name", "on", "jobs"]
    for field in required_fields:
        if field not in workflow:
            issues.append(
                f"❌ Missing required field '{field}' in {workflow_path.name}"
            )

    # Check jobs structure
    if "jobs" in workflow:
        for job_name, job_config in workflow["jobs"].items():
            if "runs-on" not in job_config:
                issues.append(
                    f"❌ Job '{job_name}' missing 'runs-on' in {workflow_path.name}"
                )

            if "steps" not in job_config:
                issues.append(
                    f"❌ Job '{job_name}' missing 'steps' in {workflow_path.name}"
                )

    # Check for common best practices
    if workflow_path.name == "ci.yml":
        # CI workflow specific checks
        if "push" not in workflow.get("on", {}):
            issues.append(f"💡 Consider adding 'push' trigger to {workflow_path.name}")

        if "pull_request" not in workflow.get("on", {}):
            issues.append(
                f"💡 Consider adding 'pull_request' trigger to {workflow_path.name}"
            )

    return issues


def check_workflows_directory(workflows_dir: Path) -> Dict[str, List[str]]:
    """Check all workflow files in the .github/workflows directory."""
    results = {}

    if not workflows_dir.exists():
        return {"_error": ["❌ .github/workflows directory not found"]}

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )

    if not workflow_files:
        return {"_error": ["⚠️ No workflow files found in .github/workflows"]}

    for workflow_file in workflow_files:
        print(f"🔍 Validating {workflow_file.name}...")
        issues = validate_workflow_file(workflow_file)
        results[workflow_file.name] = issues

        if not issues:
            print(f"✅ {workflow_file.name} is valid!")
        else:
            print(f"⚠️ {workflow_file.name} has {len(issues)} issues")

    return results


def analyze_workflow_coverage(workflows_dir: Path) -> List[str]:
    """Analyze workflow coverage and suggest improvements."""
    suggestions = []

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )
    workflow_names = [f.stem for f in workflow_files]

    # Check for recommended workflows
    recommended_workflows = {
        "ci": "Continuous Integration (testing, linting)",
        "docs": "Documentation building and deployment",
        "release": "Release automation",
    }

    for workflow_type, description in recommended_workflows.items():
        if workflow_type not in workflow_names:
            suggestions.append(
                f"💡 Consider adding {workflow_type}.yml for {description}"
            )
        else:
            print(f"✅ Found {workflow_type}.yml workflow")

    # Check for common actions
    all_content = ""
    for workflow_file in workflow_files:
        try:
            with open(workflow_file, "r") as f:
                all_content += f.read()
        except:
            continue

    recommended_actions = {
        "actions/checkout": "Code checkout",
        "actions/setup-python": "Python environment setup",
        "actions/cache": "Dependency caching",
        "actions/upload-artifact": "Artifact handling",
    }

    for action, description in recommended_actions.items():
        if action in all_content:
            print(f"✅ Using {action} for {description}")
        else:
            suggestions.append(f"💡 Consider using {action} for {description}")

    return suggestions


def main():
    """Main function."""
    print("🔧 GITHUB ACTIONS WORKFLOW VALIDATOR")
    print("=" * 40)

    # Find .github/workflows directory
    current_dir = Path(__file__).parent.parent
    workflows_dir = current_dir / ".github" / "workflows"

    print(f"📁 Checking workflows in: {workflows_dir}")

    # Validate all workflows
    results = check_workflows_directory(workflows_dir)

    total_issues = 0

    print(f"\n📋 VALIDATION RESULTS")
    print("-" * 20)

    for filename, issues in results.items():
        if filename == "_error":
            for issue in issues:
                print(issue)
                total_issues += 1
        else:
            print(f"\n📄 {filename}:")
            if not issues:
                print("  ✅ No issues found!")
            else:
                for issue in issues:
                    print(f"  {issue}")
                total_issues += len(issues)

    # Analyze coverage
    if workflows_dir.exists():
        print(f"\n🎯 WORKFLOW COVERAGE ANALYSIS")
        print("-" * 30)
        suggestions = analyze_workflow_coverage(workflows_dir)

        if suggestions:
            for suggestion in suggestions:
                print(f"  {suggestion}")
        else:
            print("  🎉 Great workflow coverage!")

    # Summary
    print(f"\n📊 SUMMARY")
    print("-" * 10)
    print(f"Total issues found: {total_issues}")

    if total_issues == 0:
        print("🎉 All workflows are properly configured!")
        return True
    else:
        print("⚠️ Please address the issues above")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
