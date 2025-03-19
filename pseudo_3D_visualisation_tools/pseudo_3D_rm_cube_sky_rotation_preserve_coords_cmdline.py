#!/usr/bin/env python3

"""
pseudo_3D_rm_cube_sky_rotation.py

This script rotates a 3D RA-Dec-RM (Faraday Depth) cube **in the plane of the sky**, 
using `reproject_interp` to properly transform both the **data** and the **coordinate system (WCS)**.

### **Fixes**
- Eliminates manual WCS rotation issues.
- Uses `reproject_interp` to **automatically update the WCS correctly**.

### **Usage**
```bash
python pseudo_3D_rm_cube_sky_rotation.py <input_fits> <rotation_angle> <output_fits>
```
- `<input_fits>`: Input 3D RA-Dec-RM cube.
- `<rotation_angle>`: Rotation in degrees (counterclockwise).
- `<output_fits>`: Output FITS filename.

### **Example**
```bash
python pseudo_3D_rm_cube_sky_rotation.py cutout_pseudo3d_2.fits 30 rotated_pseudo3d.fits
```
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

def rotate_rm_cube(input_fits, rotation_angle, output_fits):
    """
    Rotate a pseudo-3D RA-Dec-RM cube **in the plane of the sky**, keeping the RM axis unchanged.
    Uses `reproject_interp` to correctly transform **both the data and WCS**.

    Parameters:
    - input_fits (str): Path to input RA-Dec-RM cube.
    - rotation_angle (float): Rotation angle in degrees (counterclockwise).
    - output_fits (str): Path to save the rotated cube.

    Returns:
    - None (writes the rotated FITS file to disk).
    """

    # === Step 1: Load the FITS Cube ===
    print(f"📂 Loading FITS cube: {input_fits}")
    with fits.open(input_fits) as hdul:
        data = hdul[0].data  # Extract 3D data array (RA, Dec, RM)
        header = hdul[0].header  # Extract FITS header
        wcs = WCS(header)  # Extract WCS object

    # === Step 2: Define the New Rotated WCS ===
    print("🛠️  Creating a rotated WCS...")
    rotated_wcs = wcs.deepcopy()

    # Ensure the PC matrix is 3x3 for a 3D cube
    if rotated_wcs.wcs.pc.shape != (3, 3):
        pc_matrix = np.eye(3)  # Identity matrix for 3D
        pc_matrix[:2, :2] = rotated_wcs.wcs.pc  # Copy existing 2D PC matrix into the new one
        rotated_wcs.wcs.pc = pc_matrix

    # Define a 3x3 rotation matrix (RA-Dec rotation, keeping RM unchanged)
    theta = np.radians(rotation_angle)  # Convert degrees to radians
    rotation_matrix = np.eye(3)  # Start with identity matrix
    rotation_matrix[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]

    # Apply the rotation to the PC matrix
    rotated_wcs.wcs.pc = np.dot(rotation_matrix, rotated_wcs.wcs.pc)

    # === Step 3: Use `reproject_interp` to Rotate Data & WCS Together ===
    print(f"🔄 Rotating cube by {rotation_angle}° using reproject_interp...")
    new_data, _ = reproject_interp((data, wcs), rotated_wcs, shape_out=data.shape)

    # === Step 4: Save the Rotated Cube ===
    print(f"💾 Saving rotated cube to: {output_fits}")
    hdu = fits.PrimaryHDU(new_data, header=rotated_wcs.to_header())
    hdu.writeto(output_fits, overwrite=True)

    print("✅ Rotation complete! Cube and WCS are now correctly aligned.")

# === Command-line Execution ===
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_fits> <rotation_angle> <output_fits>")
        sys.exit(1)

    input_fits = sys.argv[1]
    rotation_angle = float(sys.argv[2])
    output_fits = sys.argv[3]

    rotate_rm_cube(input_fits, rotation_angle, output_fits)

