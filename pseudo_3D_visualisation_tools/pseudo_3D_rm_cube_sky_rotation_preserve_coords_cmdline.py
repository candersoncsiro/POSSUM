#!/usr/bin/env python3

"""
pseudo_3D_rm_cube_sky_rotation.py

Rotate a 3D RA-Dec-RM (Faraday Depth) cube **in the plane of the sky**, using `reproject_interp`
to transform both the **data** and **WCS**, while rotating about the **geometric center** of the image.

Usage:
    python pseudo_3D_rm_cube_sky_rotation.py <input_fits> <rotation_angle> <output_fits>
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

def rotate_rm_cube(input_fits, rotation_angle, output_fits):
    print(f"📂 Loading FITS cube: {input_fits}")
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header, naxis=3)  # Force 3D WCS

    print("🛠️  Creating rotated WCS...")
    rotated_wcs = wcs.deepcopy()

    # Ensure PC matrix is 3x3
    if rotated_wcs.wcs.pc.shape != (3, 3):
        pc_matrix = np.eye(3)
        pc_matrix[:2, :2] = rotated_wcs.wcs.pc[:2, :2]
        rotated_wcs.wcs.pc = pc_matrix

    # Compute geometric center of image
    n_faraday, n_dec, n_ra = data.shape
    image_center_pix = np.array([n_ra / 2, n_dec / 2])  # (x, y)

    # Compute offset from reference pixel to center
    original_crpix = np.array([rotated_wcs.wcs.crpix[0], rotated_wcs.wcs.crpix[1]])
    offset = original_crpix - image_center_pix

    # Temporarily set CRPIX to center
    rotated_wcs.wcs.crpix[0] = image_center_pix[0]
    rotated_wcs.wcs.crpix[1] = image_center_pix[1]

    # Rotation matrix in plane of sky
    theta = np.radians(rotation_angle)
    rotation_matrix = np.eye(3)
    rotation_matrix[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ]
    rotated_wcs.wcs.pc = np.dot(rotation_matrix, rotated_wcs.wcs.pc)

    # Rotate the offset vector back to preserve WCS world alignment
    inv_rotation = np.linalg.inv(rotation_matrix[:2, :2])
    new_offset = inv_rotation @ offset

    # Restore CRPIX with rotated offset
    rotated_wcs.wcs.crpix[0] = image_center_pix[0] + new_offset[0]
    rotated_wcs.wcs.crpix[1] = image_center_pix[1] + new_offset[1]

    print(f"🔄 Rotating by {rotation_angle}° using reproject_interp...")
    new_data, _ = reproject_interp((data, wcs), rotated_wcs, shape_out=data.shape)

    print(f"💾 Saving to {output_fits}")
    hdu = fits.PrimaryHDU(new_data, header=rotated_wcs.to_header())
    hdu.writeto(output_fits, overwrite=True)
    print("✅ Rotation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_fits> <rotation_angle> <output_fits>")
        sys.exit(1)

    input_fits = sys.argv[1]
    rotation_angle = float(sys.argv[2])
    output_fits = sys.argv[3]

    rotate_rm_cube(input_fits, rotation_angle, output_fits)
