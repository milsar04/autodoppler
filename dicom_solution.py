import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
import matplotlib.patches as patches

def get_calibration_from_metadata(ds):
    """
    Searches DICOM metadata for the Spectral Doppler region (RegionDataType == 3)
    to retrieve precise scaling factors.
    """
    delta_x = 0.01 # Default fallback (seconds/pixel)
    delta_y = 0.1  # Default fallback (cm/s/pixel)
    found = False

    if "SequenceOfUltrasoundRegions" in ds:
        for i, region in enumerate(ds.SequenceOfUltrasoundRegions):
            # RegionDataType: 1=2D, 2=M-Mode, 3=Spectral Doppler, 4=Color Flow
            r_type = getattr(region, 'RegionDataType', 0)
            
            # We specifically want Type 3 (Spectral) or if units suggest velocity (3 = cm/sec)
            units_y = getattr(region, 'PhysicalUnitsYDirection', 0)
            
            if r_type == 3 or units_y == 3:
                try:
                    # PhysicalDeltaX = seconds per pixel
                    # PhysicalDeltaY = velocity units per pixel
                    dx = region.PhysicalDeltaX
                    dy = region.PhysicalDeltaY
                    
                    # Sanity check: avoid zero divisions or zero scales
                    if dx > 0 and dy > 0:
                        delta_x = dx
                        delta_y = dy
                        print(f"Metadata Calibration Found (Region {i}): X={delta_x:.5f} s/px, Y={delta_y:.5f} cm/s/px")
                        found = True
                        break
                except AttributeError:
                    continue
    
    if not found:
        print("Warning: Spectral Doppler metadata not found. Using default calibration.")

    return delta_x, delta_y

def find_baseline_and_crop(pixel_array):
    """
    Locates the orange baseline to define the exact X and Y bounds of the spectral region.
    Uses a 'density march' from the baseline upwards to find the true top of the waveform,
    ignoring floating artifacts like text labels.
    """
    height, width, _ = pixel_array.shape
    
    # 1. Define Search Area (Bottom half only to be safe)
    search_offset_y = height // 2
    search_area = pixel_array[search_offset_y:, :, :]
    
    # 2. Create Mask for Orange/Yellow Baseline
    # R > 130, G between 80-255, B < 100 is a robust yellow/orange filter
    red_mask = search_area[:,:,0] > 130
    green_mask = (search_area[:,:,1] > 80) & (search_area[:,:,1] <= 255)
    blue_mask = search_area[:,:,2] < 100 
    
    orange_mask = red_mask & green_mask & blue_mask
    
    # 3. Detect Baseline Row (Y-Coordinate)
    # ADJUSTED HERE: Improved detection to find the TOP EDGE of the baseline
    row_counts = np.sum(orange_mask, axis=1)
    max_orange_count = np.max(row_counts)
    
    if max_orange_count < 50:
        return None, None, None, None # No valid line found
        
    # Identify all rows that contain a significant portion of the baseline (e.g., > 50% of max width)
    # This helps handle lines that might be 2-4 pixels thick.
    baseline_candidate_rows = np.where(row_counts > (max_orange_count * 0.5))[0]
    
    # Pick the smallest index (the highest row physically) to be the cut-off point.
    # This ensures we crop exactly ABOVE the orange line, excluding the line itself from the ROI.
    relative_baseline_y = np.min(baseline_candidate_rows)

    # Convert to absolute image coordinate
    baseline_y = relative_baseline_y + search_offset_y
    
    # 4. Detect Baseline Width (X-Coordinates)
    # Look at the specific row identified as the baseline (or slightly below it to be safe for width)
    # We use the row with the MAX count to determine width safely
    width_check_row = np.argmax(row_counts)
    baseline_row_pixels = orange_mask[width_check_row, :]
    orange_indices = np.where(baseline_row_pixels)[0]
    
    if len(orange_indices) == 0:
        return None, None, None, None

    min_x = orange_indices[0]
    max_x = orange_indices[-1]
    
    # 5. Detect Top Limit (Ceiling) using Density Marching
    # We start at the baseline and move UP row by row.
    # We count how many "signal" pixels are in each row.
    # If the count drops near zero for several consecutive rows, we stop.
    # This effectively "disconnects" the waveform from the floating text above it.
    
    # Crop relevant area above baseline
    # Look up to 300 pixels high
    look_ahead = 300
    roi_start_y = max(0, relative_baseline_y - look_ahead)
    density_check_region = search_area[roi_start_y:relative_baseline_y, min_x:max_x]
    
    # Convert to grayscale
    if len(density_check_region.shape) == 3:
        d_gray = np.mean(density_check_region, axis=2)
    else:
        d_gray = density_check_region
        
    # Strong threshold to see only the bright white signal
    signal_mask = d_gray > 40 
    
    # Iterate from BOTTOM (baseline) to TOP
    # 'rows' here are 0 at top, so we iterate backward from height-1 to 0
    h_roi, w_roi = d_gray.shape
    
    # We need a gap of N empty rows to declare "end of waveform"
    gap_tolerance = 10 
    current_gap = 0
    ceiling_index = 0 # Default to top of ROI if no gap found
    
    # Start from bottom (which is the baseline) and go up
    for r in range(h_roi - 1, -1, -1):
        # Count bright pixels in this row
        pixel_count = np.sum(signal_mask[r, :])
        
        # If row is empty (or just speckle noise < 5 pixels)
        if pixel_count < 5:
            current_gap += 1
        else:
            # Reset gap counter if we hit signal again
            current_gap = 0
            
        # If we found a large enough gap, this is the ceiling
        if current_gap > gap_tolerance:
            # The ceiling is the row index + gap size (going downwards)
            ceiling_index = r + gap_tolerance
            break
            
    # Convert relative ceiling index back to absolute Y coordinate
    # relative_baseline_y is the bottom of this ROI
    # The ROI starts at roi_start_y.
    # The ceiling index is relative to roi_start_y.
    
    relative_top_y = roi_start_y + ceiling_index
    
    # Add a small safety buffer (10px) above the detected signal top
    relative_top_y = max(0, relative_top_y - 10)
    
    # Convert to absolute image coordinates
    top_y = search_offset_y + relative_top_y
    
    return min_x, max_x, top_y, baseline_y

def extract_traces(roi, delta_x, delta_y):
    """
    Analyzes the ROI column by column.
    """
    height, width = roi.shape
    
    upper_line = []
    lower_line = []
    mean_velocity = []
    time_axis = []
    
    # Threshold to separate signal (white/grey) from background (black)
    threshold = 25 
    
    for col in range(width):
        column_data = roi[:, col]
        
        # Flip so index 0 is the baseline (0 velocity)
        column_flip = np.flip(column_data)
        
        # Find indices of signal pixels
        signal_indices = np.where(column_flip > threshold)[0]
        
        if len(signal_indices) > 3: 
            # 1. Upper Line (Highest velocity)
            # 98th percentile is robust against single salt-and-pepper noise pixels
            max_px = np.percentile(signal_indices, 98)
            
            # 2. Lower Line (Lowest velocity near baseline)
            # FIXED: Used 5th percentile instead of min() to ignore bottom clutter/noise
            # FIXED: Removed the check "if min_px < 5: min_px = 0" which was forcing the line flat
            min_px = np.percentile(signal_indices, 5)

            # 3. Mean Velocity (Weighted Average)
            intensities = column_flip[signal_indices]
            weighted_avg_px = np.average(signal_indices, weights=intensities)
            
            # Convert to physical units
            # delta_y is usually cm/s/px. Divide by 100 for m/s.
            v_upper = (max_px * delta_y) / 100.0
            v_lower = (min_px * delta_y) / 100.0
            v_mean  = (weighted_avg_px * delta_y) / 100.0
            
        else:
            v_upper = 0.0
            v_lower = 0.0
            v_mean  = 0.0
            
        upper_line.append(v_upper)
        lower_line.append(v_lower)
        mean_velocity.append(v_mean)
        time_axis.append(col * delta_x)
        
    return time_axis, upper_line, lower_line, mean_velocity

def analyze_dicom_velocity(file_path):
    # --- 1. CONFIGURATION ---
    # file_path: path to a spectral Doppler DICOM file
    

    # --- 2. LOAD DATA ---
    ds = pydicom.dcmread(file_path)
    pixel_data = ds.pixel_array
    
    # --- 3. GET CALIBRATION ---
    delta_x, delta_y = get_calibration_from_metadata(ds)

    # --- 4. DETECT REGION ---
    min_x, max_x, top_y, baseline_y = find_baseline_and_crop(pixel_data)
    
    if baseline_y is None:
        print("Error: Baseline not found.")
        return

    print(f"Detected Region: X[{min_x}:{max_x}], Y[{top_y}:{baseline_y}]")

    # --- 5. PREPARE ROI ---
    # Crop exactly to the detected bounds
    roi_color = pixel_data[top_y:baseline_y, min_x:max_x]
    
    if len(roi_color.shape) == 3:
        roi_gray = np.mean(roi_color, axis=2)
    else:
        roi_gray = roi_color

    # --- 6. EXTRACT TRACES ---
    t, upper, lower, mean = extract_traces(roi_gray, delta_x, delta_y)

    # Smoothing
    upper_smooth = medfilt(upper, 11)
    lower_smooth = medfilt(lower, 11)
    mean_smooth = medfilt(mean, 11)

    # Stats
    print(f"Max Velocity (Peak Systolic): {np.max(upper_smooth):.3f} m/s")
    print(f"Mean Velocity (Time Averaged): {np.mean(mean_smooth):.3f} m/s")

    # --- 7. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # === TOP: VERIFICATION ===
    ax1.imshow(pixel_data)
    ax1.set_title("Verification: Detected Region")
    
    # Draw Bounding Box
    rect = patches.Rectangle((min_x, top_y), max_x - min_x, baseline_y - top_y, 
                             linewidth=2, edgecolor='#39FF14', facecolor='none', label='Scan Region')
    ax1.add_patch(rect)
    
    # Draw Baseline
    ax1.hlines(baseline_y, min_x, max_x, colors='cyan', linewidth=2, label='Baseline')
    ax1.legend(loc='upper right')
    
    # === BOTTOM: VELOCITY GRAPH ===
    ax2.plot(t, upper_smooth, color='#4472C4', linewidth=2, label='Upper Line')
    ax2.plot(t, lower_smooth, color='#ED7D31', linewidth=2, label='Lower Line')
    ax2.plot(t, mean_smooth,  color='#A5A5A5', linewidth=2, label='Average Velocity')
    
    ax2.set_title("Blood Velocity in CCA")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    
    ax2.grid(True, which='major', axis='both', linestyle='-', color='#d9d9d9')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0, right=max(t))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("Usage: python dicom_solution.py <path_to_doppler_dicom>")
    analyze_dicom_velocity(sys.argv[1])
