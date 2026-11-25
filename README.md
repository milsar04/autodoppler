# AutoDoppler

Automated extraction of arterial blood velocity waveforms from spectral Doppler ultrasound DICOM files.

Team project (5 students), Industry 4.0 course, Lodz University of Technology.

## Problem

Spectral Doppler scans are exported from ultrasound machines as images, not as numbers. Researchers who run blood flow simulations (CFD) need velocity-time data as input, so they digitize the waveforms by hand, which can take a full day and differs between operators. AutoDoppler turns the DICOM image into numerical velocity data automatically and reproducibly.

## How it works

1. **Calibration from metadata.** Reads the DICOM `SequenceOfUltrasoundRegions` and takes the spectral Doppler region's physical scale (seconds per pixel, cm/s per pixel). Falls back to defaults with a warning if the region is missing.
2. **Region detection.** Finds the orange zero-velocity baseline by color masking, then runs a "density march" upward from the baseline, row by row, until it hits a gap of empty rows. This crops the waveform and excludes text labels and other artifacts.
3. **Trace extraction.** For each pixel column, takes the signal pixels above a threshold and computes the upper envelope (98th percentile), lower envelope (5th percentile) and intensity-weighted mean, converted to m/s on a time axis in seconds.
4. **Smoothing and cycles** (notebook). Median and Savitzky-Golay filtering, heartbeat peak detection with height and distance constraints, cycle boundary detection, and an averaged representative cardiac cycle as input for simulations.

## Files

- `dicom_solution.py`: DICOM loading, calibration, region detection, trace extraction and a verification plot (my part of the project).
- `test.ipynb`: full pipeline including peak detection, cycle splitting and the averaged cycle (team work).

## Usage

```bash
pip install -r requirements.txt
python dicom_solution.py path/to/doppler_dicom_file
```

The script prints the detected region, peak systolic and time-averaged velocity, and shows the detected region next to the extracted velocity curves.

## Data

No medical data is included in this repository. The scans used during development came from a research partner and are not public. Use your own spectral Doppler DICOM files.

## Limitations

Thresholds (color mask, signal threshold, peak and cycle percentiles) were tuned on the available scans and may need adjustment for other machines or settings. The extracted waveforms have not yet been validated against manual measurements.
