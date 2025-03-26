#!/usr/bin/env python3

"""
cutout_fits_cube.py

Extracts a rectangular spatial cutout (in pixel coordinates) from a FITS RA–Dec–Faraday cube.

USAGE:
    python cutout_fits_cube.py <input_fits> <x_min> <x_max> <y_min> <y_max> <output_fits>

ARGS:
    input_fits   Path to the input FITS file
    x_min        Minimum pixel index along RA axis (NAXIS1)
    x_max        Maximum pixel index along RA axis
    y_min        Minimum pixel index along Dec axis (NAXIS2)
    y_max        Maximum pixel index along Dec axis
    output_fits  Path to save the cropped FITS file

EXAMPLE:
    python cutout_fits_cube.py rg12_pseudo3dcube.fits 500 800 600 900 rg12_cutout.fits
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def cutout_cube(input_fits, x_min, x_max, y_min, y_max, output_fits):
    with fits.open(input_fits) as hdul:
        data = hdul[0].data  # Shape: (FARADAY, DEC, RA)
        header = hdul[0].header
        wcs = WCS(header)

        # Perform the cutout
        cutout_data = data[:, y_min:y_max, x_min:x_max]

        # Update header and CRPIX to match cutout
        new_header = header.copy()
        new_header['NAXIS1'] = x_max - x_min
        new_header['NAXIS2'] = y_max - y_min
        new_header['CRPIX1'] -= x_min
        new_header['CRPIX2'] -= y_min

        # Save cutout to new FITS
        hdu = fits.PrimaryHDU(cutout_data, header=new_header)
        hdu.writeto(output_fits, overwrite=True)
        print(f"✅ Saved cutout to: {output_fits}")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage:\n  python cutout_fits_cube.py <input_fits> <x_min> <x_max> <y_min> <y_max> <output_fits>")
        sys.exit(1)

    input_fits = sys.argv[1]
    x_min = int(sys.argv[2])
    x_max = int(sys.argv[3])
    y_min = int(sys.argv[4])
    y_max = int(sys.argv[5])
    output_fits = sys.argv[6]

    cutout_cube(input_fits, x_min, x_max, y_min, y_max, output_fits)
