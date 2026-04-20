#!/usr/bin/env python3
"""
Simple script to mask out the time display in Ubuntu desktop screenshots.
Usage: python mask_time.py input_screenshot.png [output_screenshot.png]
"""

import sys
import os
from PIL import Image, ImageDraw

def mask_ubuntu_time(input_path, output_path=None, mask_color=(19, 19, 19)):
    """
    Mask the time display in Ubuntu desktop top bar.
    
    Args:
        input_path: Path to input screenshot
        output_path: Path for output (default: input_path with '_masked' suffix)
        mask_color: RGB color for the mask (default: gray)
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Load image
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    print(f"Image size: {img.width} x {img.height}")
    
    # Calculate mask area for Ubuntu time (top-right corner)
    # Ubuntu top bar is typically 24-30 pixels high
    # Time area is roughly the rightmost 100-120 pixels
    top_bar_height = max(24, int(img.height * 0.02))  # ~2% of screen height
    time_area_width = max(100, int(img.width * 0.08))   # ~8% of screen width
    
    # Define rectangle coordinates (left, top, right, bottom)
    mask_rect = (
        img.width / 2 - 50,  # left
        0,                           # top
        img.width / 2 + 150,                   # right
        top_bar_height              # bottom
    )
    
    print(f"Masking area: ({mask_rect[0]}, {mask_rect[1]}) to ({mask_rect[2]}, {mask_rect[3]})")
    print(f"Mask size: {time_area_width} x {top_bar_height} pixels")
    
    # Apply mask
    draw = ImageDraw.Draw(img)
    draw.rectangle(mask_rect, fill=mask_color)
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_masked{ext}"
    
    # Save image
    try:
        img.save(output_path)
        print(f"Masked screenshot saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python mask_time.py input_screenshot.png [output_screenshot.png]")
        print("Example: python mask_time.py screenshot.png")
        print("Example: python mask_time.py screenshot.png screenshot_no_time.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = mask_ubuntu_time(input_path, output_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()