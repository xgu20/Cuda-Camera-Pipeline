#!/usr/bin/env python3
import sys
import zlib
import urllib.request
import os

def plantuml_encode(text):
    # UTF-8 encode
    utf8_bytes = text.encode('utf-8')
    
    # Raw deflate compress (zlib.compressobj with wbits=-15)
    compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    compressed = compressor.compress(utf8_bytes) + compressor.flush()
    
    # PlantUML custom base64 translation
    puml_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    
    res = []
    i = 0
    while i < len(compressed):
        b1 = compressed[i]
        b2 = compressed[i+1] if i+1 < len(compressed) else 0
        b3 = compressed[i+2] if i+2 < len(compressed) else 0
        
        c1 = b1 >> 2
        c2 = ((b1 & 0x03) << 4) | (b2 >> 4)
        c3 = ((b2 & 0x0F) << 2) | (b3 >> 6)
        c4 = b3 & 0x3F
        
        res.append(puml_chars[c1])
        res.append(puml_chars[c2])
        if i + 1 < len(compressed):
            res.append(puml_chars[c3])
        if i + 2 < len(compressed):
            res.append(puml_chars[c4])
        i += 3
        
    return "".join(res)

def main():
    puml_path = "docs/pipeline.puml"
    svg_path = "docs/pipeline.svg"
    
    if not os.path.exists(puml_path):
        print(f"Error: {puml_path} not found.")
        sys.exit(1)
        
    print(f"Reading {puml_path}...")
    with open(puml_path, "r", encoding="utf-8") as f:
        puml_text = f.read()
        
    print("Encoding PlantUML text...")
    encoded = plantuml_encode(puml_text)
    
    url = f"http://www.plantuml.com/plantuml/svg/{encoded}"
    print(f"Fetching SVG from {url}...")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            svg_data = response.read()
            
        print(f"Writing SVG to {svg_path}...")
        with open(svg_path, "wb") as f:
            f.write(svg_data)
        print("Success!")
    except Exception as e:
        print(f"Error fetching SVG: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
