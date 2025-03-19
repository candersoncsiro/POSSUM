#!/usr/bin/env python3

"""
pseudo_3D_rm_cube_sky_rotation.py

This script rotates a 3D RA-Dec-RM (Faraday Depth) cube **in the plane of the sky**, 
preserving the Faraday Depth axis. It is designed for the output of `generate_pseudo3D.py`, 
which implements the Rudnick+2024 pseudo-3D visualization method.

### **Use Case & Assumptions**
- Works on **RA-Dec-RM (Faraday Depth) cubes**, such as those generated from polarization data.
- Assumes the WCS is valid for **small regions of the sky**, such that the WCS transforms 
  implemented by `reproject` remain accurate.
- Uses a **WCS trick**: Temporarily renaming the RM axis to a spectral axis (velocity) 
  so that `spectral-cube` and `reproject` can handle it correctly.

### **Usage**
```bash
python pseudo_3D_rm_cube_sky_rotation.py <input_fits> <rotation_angle> <output_fits>
```
- `<input_fits>`: Input 3D RA-Dec-RM cube.
- `<rotation_angle>`: Rotation in degrees (counterclockwise in the RA-Dec plane).
- `<output_fits>`: Output FITS filename.

### **Example**
```bash
python pseudo_3D_rm_cube_sky_rotation.py cutout_pseudo3d_2.fits 30 rotated_pseudo3d.fits
```

### **Dependencies**
- `astropy`
- `spectral-cube`
- `scipy`

### **Author:** Craig Anderson, 2025
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from scipy.ndimage import rotate

def rotate_rm_cube(input_fits, rotation_angle, output_fits):
    """
    Rotate a pseudo-3D RA-Dec-RM cube **in the plane of the sky**, keeping the RM axis unchanged.

    Parameters:
    - input_fits (str): Path to input RA-Dec-RM cube.
    - rotation_angle (float): Rotation angle in degrees (counterclockwise).
    - output_fits (str): Path to save the rotated cube.

    Returns:
    - None (writes the rotated FITS file to disk).
    """

    # === Step 1: Load the FITS Cube ===
    print(f"\U0001F4C2 Loading FITS cube: {input_fits}")
    with fits.open(input_fits, mode="update") as hdul:
        data = hdul[0].data  # Extract 3D data array (RA, Dec, RM)
        header = hdul[0].header  # Extract FITS header

        # === Step 2: Convert RM Axis to a "Fake Spectral Axis" ===
        print("\U0001F6E0️  Converting RM axis to a temporary spectral axis...")
        header["CTYPE3"] = "VELO-LSR"  # Fake it as a velocity axis
        header["CUNIT3"] = "m/s"  # Fake the unit
        header["RESTFRQ"] = 1.4e9  # Arbitrary rest frequency (needed for some WCS tools)
        hdul.flush()  # Save changes

    # === Step 3: Load the Modified Cube as a Spectral Cube ===
    print("\U0001F4E1 Loading modified cube with spectral-cube...")
    cube = SpectralCube.read(input_fits)

    # === Step 4: Apply Rotation in RA-Dec ===
    print(f"\U0001F504 Rotating the cube by {rotation_angle}° in the plane of the sky...")
    rotated_data = rotate(cube.filled_data[:], rotation_angle, axes=(1, 2), reshape=False, order=1)

    # === Step 5: Restore the RM Axis in the Header ===
    print("\U0001F504 Restoring RM axis labeling...")
    with fits.open(input_fits, mode="update") as hdul:
        header = hdul[0].header
        header["CTYPE3"] = "FARADAY "  # Restore RM axis
        header["CUNIT3"] = "rad/m^2"  # Restore RM units
        del header["RESTFRQ"]  # Remove fake rest frequency
        hdul.flush()  # Save changes

    # === Step 6: Save the Rotated Cube ===
    print(f"\U0001F4BE Saving rotated cube to: {output_fits}")
    hdu = fits.PrimaryHDU(rotated_data, header=header)
    hdu.writeto(output_fits, overwrite=True)

    print("✅ Rotation complete!")

# === Command-line Execution ===
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_fits> <rotation_angle> <output_fits>")
        sys.exit(1)

    input_fits = sys.argv[1]
    rotation_angle = float(sys.argv[2])
    output_fits = sys.argv[3]

    rotate_rm_cube(input_fits, rotation_angle, output_fits)