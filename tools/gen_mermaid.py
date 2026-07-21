#!/usr/bin/env python3
import sys
import base64
import json
import urllib.request
import os

def main():
    mmd_path = "docs/pipeline.mmd"
    svg_path = "docs/pipeline.svg"
    
    if not os.path.exists(mmd_path):
        print(f"Error: {mmd_path} not found.")
        sys.exit(1)
        
    print(f"Reading {mmd_path}...")
    with open(mmd_path, "r", encoding="utf-8") as f:
        mmd_text = f.read()
        
    # Prepare JSON structure for mermaid.ink
    data = {
        "code": mmd_text,
        "mermaid": {
            "theme": "default",
            "themeVariables": {
                "fontSize": "16px"
            }
        }
    }
    
    # Base64 encode JSON
    json_bytes = json.dumps(data).encode("utf-8")
    b64_encoded = base64.b64encode(json_bytes).decode("utf-8")
    
    url = f"https://mermaid.ink/svg/{b64_encoded}"
    print(f"Fetching precompiled SVG from {url}...")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            svg_data = response.read()
            
        print(f"Saving compiled SVG to {svg_path}...")
        with open(svg_path, "wb") as f:
            f.write(svg_data)
        print("Success!")
    except Exception as e:
        print(f"Error fetching SVG: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
