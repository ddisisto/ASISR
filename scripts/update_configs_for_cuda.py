#!/usr/bin/env python3
"""
Update Configuration Files for CUDA Auto-Detection

Changes all config files from device: "cpu" to device: "auto" to enable
automatic CUDA usage when available while maintaining CPU fallback.
"""

import re
from pathlib import Path
import yaml


def update_config_file(config_path: Path) -> bool:
    """
    Update a single configuration file to use automatic device detection.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        True if file was modified, False if no changes needed
    """
    try:
        # Read the original file
        with open(config_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: device: "cpu" (with optional comments)
        content = re.sub(
            r'^(\s*)device:\s*"cpu"(\s*#.*)?$',
            r'\1device: "auto"\2',
            content,
            flags=re.MULTILINE
        )
        
        # Pattern 2: device: cpu (without quotes)
        content = re.sub(
            r'^(\s*)device:\s*cpu(\s*#.*)?$',
            r'\1device: "auto"\2',
            content,
            flags=re.MULTILINE
        )
        
        # Check if changes were made
        if content != original_content:
            # Verify the result is still valid YAML
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"❌ YAML parsing error in {config_path}: {e}")
                return False
            
            # Write the updated content
            with open(config_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {config_path.name}")
            return True
        else:
            print(f"ℹ️  No changes needed for {config_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {config_path}: {e}")
        return False


def main():
    """Update all configuration files in the configs/ directory."""
    print("🔧 UPDATING CONFIGS FOR CUDA AUTO-DETECTION")
    print("=" * 50)
    
    configs_dir = Path("configs")
    if not configs_dir.exists():
        print(f"❌ Configs directory not found: {configs_dir}")
        return
    
    # Find all YAML files
    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    
    if not yaml_files:
        print("❌ No YAML configuration files found")
        return
    
    print(f"Found {len(yaml_files)} configuration files")
    print()
    
    # Track changes
    modified_files = []
    unchanged_files = []
    error_files = []
    
    # Process each file
    for yaml_file in sorted(yaml_files):
        try:
            if update_config_file(yaml_file):
                modified_files.append(yaml_file.name)
            else:
                unchanged_files.append(yaml_file.name)
        except Exception as e:
            print(f"❌ Failed to process {yaml_file.name}: {e}")
            error_files.append(yaml_file.name)
    
    # Summary
    print()
    print("=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    if modified_files:
        print(f"✅ Modified {len(modified_files)} files:")
        for filename in modified_files:
            print(f"   - {filename}")
        print()
    
    if unchanged_files:
        print(f"ℹ️  {len(unchanged_files)} files already correct:")
        for filename in unchanged_files[:5]:  # Show first 5
            print(f"   - {filename}")
        if len(unchanged_files) > 5:
            print(f"   ... and {len(unchanged_files) - 5} more")
        print()
    
    if error_files:
        print(f"❌ {len(error_files)} files had errors:")
        for filename in error_files:
            print(f"   - {filename}")
        print()
    
    print(f"🎯 RESULT: {len(modified_files)} files updated for automatic CUDA detection")
    
    if modified_files:
        print()
        print("📋 NEXT STEPS:")
        print("1. Test with: python scripts/validate_cuda_setup.py")
        print("2. Run experiment: python run_experiment.py single configs/phase3a_optimal_beta_8x8.yaml")
        print("3. Verify GPU usage in output logs")


if __name__ == "__main__":
    main()