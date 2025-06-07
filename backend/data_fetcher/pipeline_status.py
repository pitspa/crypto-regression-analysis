#!/usr/bin/env python3
"""
Pipeline status tracker - creates a status report for the entire data pipeline
"""

import json
import os
from datetime import datetime
import traceback

class PipelineStatus:
    def __init__(self, status_file="../data/pipeline_status.json"):
        self.status_file = status_file
        self.status = {
            "last_run": datetime.now().isoformat(),
            "overall_status": "running",
            "steps": {},
            "errors": [],
            "warnings": []
        }
    
    def update_step(self, step_name, status, details=None, error=None):
        """Update the status of a pipeline step"""
        self.status["steps"][step_name] = {
            "status": status,  # "success", "failed", "skipped", "running"
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "error": error
        }
        
        # Update overall status
        if any(step["status"] == "failed" for step in self.status["steps"].values()):
            self.status["overall_status"] = "failed"
        elif all(step["status"] == "success" for step in self.status["steps"].values()):
            self.status["overall_status"] = "success"
        
        self.save()
    
    def add_error(self, error_msg, step_name=None):
        """Add an error to the status"""
        self.status["errors"].append({
            "message": error_msg,
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc()
        })
        self.save()
    
    def add_warning(self, warning_msg, step_name=None):
        """Add a warning to the status"""
        self.status["warnings"].append({
            "message": warning_msg,
            "step": step_name,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def save(self):
        """Save status to file"""
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def create_summary(self):
        """Create a human-readable summary"""
        summary = []
        summary.append(f"Pipeline Status: {self.status['overall_status'].upper()}")
        summary.append(f"Last Run: {self.status['last_run']}")
        summary.append("\nSteps:")
        
        for step_name, step_data in self.status["steps"].items():
            status_icon = "✓" if step_data["status"] == "success" else "✗"
            summary.append(f"  {status_icon} {step_name}: {step_data['status']}")
            if step_data.get("details"):
                summary.append(f"    Details: {step_data['details']}")
            if step_data.get("error"):
                summary.append(f"    Error: {step_data['error']}")
        
        if self.status["errors"]:
            summary.append(f"\nErrors ({len(self.status['errors'])}):")
            for error in self.status["errors"]:
                summary.append(f"  - {error['message']} (in {error['step']})")
        
        if self.status["warnings"]:
            summary.append(f"\nWarnings ({len(self.status['warnings'])}):")
            for warning in self.status["warnings"]:
                summary.append(f"  - {warning['message']}")
        
        return "\n".join(summary)