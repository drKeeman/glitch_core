#!/usr/bin/env python3
"""
Test script to verify the improved data export functionality.
"""

import requests
import json
from datetime import datetime

def test_export():
    """Test the data export endpoint."""
    base_url = "http://localhost:8000"
    
    # Test export request
    export_request = {
        "export_format": "json",
        "include_assessments": True,
        "include_mechanistic": True,
        "include_events": True
    }
    
    print("Testing data export...")
    print(f"Request: {json.dumps(export_request, indent=2)}")
    
    try:
        # Make export request
        response = requests.post(
            f"{base_url}/api/v1/data/export",
            json=export_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Export successful!")
            print(f"Filename: {result.get('filename')}")
            print(f"Filepath: {result.get('filepath')}")
            print(f"Size: {result.get('size_bytes')} bytes")
            print(f"Data counts: {result.get('data_counts')}")
            
            # Download and examine the exported file
            download_response = requests.get(f"{base_url}/api/v1/data/download/{result.get('filename')}")
            if download_response.status_code == 200:
                export_data = json.loads(download_response.content)
                
                print(f"\n📊 Export Data Summary:")
                print(f"Assessments: {len(export_data.get('assessments', []))}")
                print(f"Mechanistic data: {len(export_data.get('mechanistic_data', []))}")
                print(f"Events: {len(export_data.get('events', []))}")
                
                # Show first few assessment records
                assessments = export_data.get('assessments', [])
                if assessments:
                    print(f"\n📋 First assessment record:")
                    print(json.dumps(assessments[0], indent=2))
                
                # Show metadata
                metadata = export_data.get('metadata', {})
                print(f"\n📋 Export metadata:")
                print(json.dumps(metadata, indent=2))
                
            else:
                print(f"❌ Failed to download export file: {download_response.status_code}")
                
        else:
            print(f"❌ Export failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing export: {e}")

if __name__ == "__main__":
    test_export() 