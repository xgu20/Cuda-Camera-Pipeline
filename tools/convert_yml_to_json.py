import os
import yaml
import json
import glob

def convert_configs():
    # Find all yml files matching *-configs.yml in data/infinite/
    yml_files = glob.glob("data/infinite/*-configs.yml")
    for yml_path in yml_files:
        print(f"Processing {yml_path}...")
        with open(yml_path, "r") as f:
            yml_data = yaml.safe_load(f)
        
        # Build JSON sidecar config dictionary
        sensor_info = yml_data.get("sensor_info", {})
        
        # Width, height, bit_depth, bayer_pattern
        width = sensor_info.get("width", 0)
        height = sensor_info.get("height", 0)
        bit_depth = sensor_info.get("bit_depth", 16)
        bayer_pattern = sensor_info.get("bayer_pattern", "RGGB").upper()
        
        # Packing: default to unpacked_u16
        packing = "unpacked_u16"
        
        # Black level (read from black_level_correction.r_offset or fallback)
        blc = yml_data.get("black_level_correction", {})
        black_level = blc.get("r_offset", 0)
        
        # White level (r_sat or default)
        white_level = blc.get("r_sat", (1 << bit_depth) - 1)
        
        # DP thresholds
        dpc = yml_data.get("dead_pixel_correction", {})
        dp_threshold = dpc.get("dp_threshold", 8000)
        
        # White balance gains
        wb = yml_data.get("white_balance", {})
        wb_gains = {
            "r": float(wb.get("r_gain", 1.0)),
            "gr": 1.0,
            "gb": 1.0,
            "b": float(wb.get("b_gain", 1.0))
        }
        
        # Color correction matrix (scaled by 1024)
        ccm_data = yml_data.get("color_correction_matrix", {})
        ccm_array = []
        if ccm_data and ccm_data.get("is_enable", True):
            r_row = ccm_data.get("corrected_red", [1024, 0, 0])
            g_row = ccm_data.get("corrected_green", [0, 1024, 0])
            b_row = ccm_data.get("corrected_blue", [0, 0, 1024])
            # Divide each element by 1024.0
            ccm_flat = r_row + g_row + b_row
            ccm_array = [float(x) / 1024.0 for x in ccm_flat]
        
        # Construct JSON structure
        json_dict = {
            "width": int(width),
            "height": int(height),
            "bit_depth": int(bit_depth),
            "bayer_pattern": bayer_pattern,
            "packing": packing,
            "black_level": int(black_level),
            "white_level": int(white_level),
            "hot_pixel_threshold": int(dp_threshold),
            "dead_pixel_threshold": int(dp_threshold),
            "white_balance_gains": wb_gains
        }
        if ccm_array:
            json_dict["color_correction_matrix"] = ccm_array
            
        # Target JSON path: replace "-configs.yml" with ".json"
        json_path = yml_path.replace("-configs.yml", ".json")
        print(f"Writing to {json_path}...")
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)

if __name__ == "__main__":
    convert_configs()
