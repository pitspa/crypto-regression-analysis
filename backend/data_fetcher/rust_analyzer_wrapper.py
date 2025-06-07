#!/usr/bin/env python3
"""
Python wrapper for Rust analyzer to integrate with pipeline status tracking
"""

import subprocess
import os
import json
import sys
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from pipeline_status import PipelineStatus

def run_rust_analyzer(data_dir="../data"):
    """Run the Rust analyzer and track its status"""
    status = PipelineStatus(os.path.join(data_dir, "pipeline_status.json"))
    
    # Check if data exists
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        error_msg = "No metadata.json found. Run data fetcher first."
        status.update_step("rust_analyzer", "failed", error=error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    # Check if combined data exists
    combined_data_path = os.path.join(data_dir, "combined_data.csv")
    if not os.path.exists(combined_data_path):
        error_msg = "No combined_data.csv found. Data fetcher may have failed."
        status.update_step("rust_analyzer", "failed", error=error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    # Change to Rust analyzer directory
    rust_dir = os.path.join(os.path.dirname(__file__), "../rust_analyzer")
    os.chdir(rust_dir)
    
    try:
        # Build Rust project
        print("Building Rust analyzer...")
        status.update_step("rust_build", "running")
        
        build_result = subprocess.run(
            ["cargo", "build", "--release"],
            capture_output=True,
            text=True
        )
        
        if build_result.returncode != 0:
            error_msg = f"Rust build failed: {build_result.stderr}"
            status.update_step("rust_build", "failed", error=error_msg)
            print(f"ERROR: {error_msg}")
            return False
        
        status.update_step("rust_build", "success", "Build completed")
        
        # Run analyzer
        print("Running regression analysis...")
        status.update_step("rust_analysis", "running")
        
        run_result = subprocess.run(
            ["cargo", "run", "--release"],
            capture_output=True,
            text=True
        )
        
        if run_result.returncode != 0:
            error_msg = f"Rust analyzer failed: {run_result.stderr}"
            status.update_step("rust_analysis", "failed", error=error_msg)
            print(f"ERROR: {error_msg}")
            return False
        
        # Check output files
        window_sizes = [7, 14, 30, 60, 90, 120, 180]
        generated_files = []
        missing_files = []
        
        for window_size in window_sizes:
            output_file = os.path.join(data_dir, f"regression_results_window_{window_size}.json")
            if os.path.exists(output_file):
                # Verify file is valid JSON
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                        result_count = len(data.get('results', []))
                        generated_files.append(f"window_{window_size} ({result_count} results)")
                except Exception as e:
                    missing_files.append(f"window_{window_size} (invalid JSON)")
            else:
                missing_files.append(f"window_{window_size}")
        
        if missing_files:
            status.add_warning(f"Missing output files: {', '.join(missing_files)}")
        
        status.update_step("rust_analysis", "success", 
                          f"Generated {len(generated_files)} files: {', '.join(generated_files)}")
        
        print(f"Analysis complete! Generated {len(generated_files)} result files")
        print(run_result.stdout)
        
        return True
        
    except FileNotFoundError as e:
        error_msg = "Cargo not found. Please install Rust."
        status.update_step("rust_analyzer", "failed", error=error_msg)
        print(f"ERROR: {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        status.update_step("rust_analyzer", "failed", error=error_msg)
        print(f"ERROR: {error_msg}")
        return False

if __name__ == "__main__":
    success = run_rust_analyzer()
    sys.exit(0 if success else 1)