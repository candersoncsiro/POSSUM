import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from matplotlib.colors import LogNorm
import argparse
import os
from matplotlib.path import Path
import xml.etree.ElementTree as ET
from matplotlib.path import Path
import re


def visualize_rm_cube(fits_file, x_min_pix, x_max_pix, y_min_pix, y_max_pix,
                      rm_min_pix=None, rm_max_pix=None, project_axis='y',
                      region_file=None, fig_width=10, fig_height=5, save_path=None):
    """
    Visualizes a rotated RA-Dec-RM cube by generating projections:
    - RM vs. X or RM vs. Y, integrated along the specified spatial axis
    - X vs. Y spatial map, integrated over RM, with optional region masking

    Parameters:
        fits_file (str): Path to the input FITS file.
        x_min_pix, x_max_pix (int): Pixel limits for the X-axis (RA).
        y_min_pix, y_max_pix (int): Pixel limits for the Y-axis (Dec).
        rm_min_pix, rm_max_pix (int): Pixel limits for the RM axis (Faraday depth).
        project_axis (str): 'x' or 'y' axis along which to project for RM slice.
        region_file (str): Path to CARTA CRTF region file in pixel coordinates.
        fig_width, fig_height (float): Size of the output figure.
        save_path (str): Path to save the figure (if desired).
    """

    # === Load the FITS Cube ===
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # 3D cube (RM, Y, X)
        header = hdul[0].header
        wcs = WCS(header)

    # Extract RM WCS info
    cdelt3 = header.get('CDELT3')
    crpix3 = header.get('CRPIX3')
    crval3 = header.get('CRVAL3')
    n_rm = data.shape[0]

    rm_values_full = crval3 + cdelt3 * (np.arange(n_rm) + 1 - crpix3)
    rm_start = rm_min_pix if rm_min_pix else 0
    rm_stop = rm_max_pix if rm_max_pix else n_rm
    rm_range = rm_values_full[rm_start:rm_stop]

    # Apply limits
    rm_slice = slice(rm_min_pix, rm_max_pix)  # Use full range unless specified
    y_slice = slice(y_min_pix, y_max_pix)
    x_slice = slice(x_min_pix, x_max_pix)
    data_cropped = data[rm_slice, y_slice, x_slice]

    # === Build mask from region file if specified ===
    if region_file:
        try:
            with open(region_file, 'r') as f:
                region_text = f.read()

            pattern = r'\[\s*([\d.]+)pix,\s*([\d.]+)pix\s*\]'
            matches = re.findall(pattern, region_text)

            if not matches:
                raise ValueError("❌ No polygon vertices found in region file.")

            polygon_coords = [(float(x), float(y)) for x, y in matches]
            adjusted_polygon = polygon_coords

            ny, nx = data_cropped.shape[1], data_cropped.shape[2]
            yy, xx = np.mgrid[:ny, :nx]
            points = np.vstack((xx.flatten(), yy.flatten())).T

            region_path = Path(adjusted_polygon)
            mask = region_path.contains_points(points).reshape(ny, nx)
            data_cropped[:, ~mask] = np.nan

        except Exception as e:
            print(e)

    if project_axis=='y':
        rm_spatial_projection = np.nansum(data_cropped, axis=1)
        spatial_spatial_projection = np.nansum(data_cropped, axis=0)

        print("NaNs in RM vs spatial projection:", np.isnan(rm_spatial_projection).sum())
        print("Max value in RM vs projection:", np.nanmax(rm_spatial_projection))

        wcs_2d = wcs.celestial

        fig = plt.figure(figsize=(fig_width, fig_height))

        ax1 = fig.add_subplot(2, 1, 1)
        cmap_top = ax1.imshow(rm_spatial_projection, aspect='auto', origin='lower',
                               extent=[x_min_pix, x_max_pix, rm_range[0], rm_range[-1]],
                               cmap='magma', norm=LogNorm(vmax=np.nanmax(rm_spatial_projection), 
                                                          vmin=np.nanmax(rm_spatial_projection) * 1e-3))
        ax1.set_ylabel("Faraday Depth (rad/m²)")
        ax1.set_xticks([])
        fig.colorbar(cmap_top, ax=ax1, label="Integrated Intensity")

        ax2 = fig.add_subplot(2, 1, 2, projection=wcs_2d, aspect='equal')
        im = ax2.imshow(spatial_spatial_projection, origin='lower', cmap='magma',
                         extent=[x_min_pix, x_max_pix, y_min_pix, y_max_pix], norm=LogNorm(vmax=np.nanmax(spatial_spatial_projection), 
                                                                  vmin=np.nanmax(spatial_spatial_projection) * 1e-3))
        ax2.coords.grid(True, color='blue', linestyle='--', alpha=0.8, linewidth=1.0)
        ax2.set_xlabel("RA")
        plt.setp(ax2.get_xticklabels(), rotation=45)
        ax2.set_ylabel("Dec")
        fig.colorbar(im, ax=ax2, label="Integrated Intensity")

        plt.subplots_adjust(hspace=0.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    elif project_axis=='x':
        rm_spatial_projection = np.nansum(data_cropped, axis=2).T
        spatial_spatial_projection = np.nansum(data_cropped, axis=0)

        print("NaNs in RM vs spatial projection:", np.isnan(rm_spatial_projection).sum())
        print("Max value in RM vs projection:", np.nanmax(rm_spatial_projection))

        wcs_2d = wcs.celestial

        fig = plt.figure(figsize=(fig_width, fig_height))

        ax1 = fig.add_subplot(1, 2, 2)
        cmap_top = ax1.imshow(rm_spatial_projection, aspect='auto', origin='lower',
                               extent=[rm_range[0], rm_range[-1], y_min_pix, y_max_pix],
                               cmap='magma', norm=LogNorm(vmax=np.nanmax(rm_spatial_projection), 
                                                          vmin=np.nanmax(rm_spatial_projection) * 1e-3))
        ax1.set_xlabel("Faraday Depth (rad/m²)")
        ax1.set_yticks([])
        fig.colorbar(cmap_top, ax=ax1, label="Integrated Intensity", location='top')

        ax2 = fig.add_subplot(1, 2, 1, projection=wcs_2d, aspect='equal')
        im = ax2.imshow(spatial_spatial_projection, origin='lower', cmap='magma',
                         extent=[x_min_pix, x_max_pix, y_min_pix, y_max_pix], norm=LogNorm(vmax=np.nanmax(spatial_spatial_projection), 
                                                                  vmin=np.nanmax(spatial_spatial_projection) * 1e-3))
        ax2.coords.grid(True, color='blue', linestyle='--', alpha=0.8, linewidth=1.0)
        ax2.set_xlabel("RA")
        plt.setp(ax2.get_xticklabels(), rotation=45)
        ax2.set_ylabel("Dec")
        fig.colorbar(im, ax=ax2, label="Integrated Intensity", location='top')

        plt.subplots_adjust(hspace=0.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    else:
        raise ValueError("Input project_axis must be either 'x' or 'y'")

# CLI Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RA-Dec-RM cube projections.")
    parser.add_argument("fits_file", type=str)
    parser.add_argument("x_min_pix", type=int)
    parser.add_argument("x_max_pix", type=int)
    parser.add_argument("y_min_pix", type=int)
    parser.add_argument("y_max_pix", type=int)
    parser.add_argument("--rm_min_pix", type=int, default=None)
    parser.add_argument("--rm_max_pix", type=int, default=None)
    parser.add_argument("--project_axis", type=str, choices=['x', 'y'], default='y')
    parser.add_argument("--region_file", type=str, default=None)
    parser.add_argument("--fig_width", type=float, default=10)
    parser.add_argument("--fig_height", type=float, default=5)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    visualize_rm_cube(args.fits_file, args.x_min_pix, args.x_max_pix,
                      args.y_min_pix, args.y_max_pix,
                      args.rm_min_pix, args.rm_max_pix,
                      args.project_axis, args.region_file,
                      args.fig_width, args.fig_height,
                      args.save_path)