import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from matplotlib.colors import LogNorm

def visualize_rm_cube(fits_file, x_min_pix, x_max_pix, y_min_pix, y_max_pix, rm_min_pix=None, rm_max_pix=None, fig_width=10, fig_height=5, save_path=None):
    """
    Visualizes a rotated RA-Dec-RM cube by generating two stacked projections:
    - Top: RM vs. X (integrated over Y)
    - Bottom: X vs. Y (integrated over RM) with WCS grid overlay.
    
    Parameters:
        fits_file (str): Path to the input FITS file.
        x_min_pix, x_max_pix (int): Pixel limits for the X-axis (RA direction).
        y_min_pix, y_max_pix (int): Pixel limits for the Y-axis (Dec direction).
        rm_min_pix, rm_max_pix (int, optional): Pixel limits for the RM axis (Faraday depth). Defaults to full range.
        fig_width, fig_height (float): Figure size in inches.
        save_path (str, optional): Path to save the output figure instead of displaying it.
    """
    # === Load the FITS Cube ===
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # 3D cube (RM, Y, X)
        header = hdul[0].header
        wcs = WCS(header)
    
    # Apply limits
    rm_slice = slice(rm_min_pix, rm_max_pix)  # Use full range unless specified
    y_slice = slice(y_min_pix, y_max_pix)
    x_slice = slice(x_min_pix, x_max_pix)
    data_cropped = data[rm_slice, y_slice, x_slice]
    
    # === Generate Integrated Maps using NaN-Sums ===
    rm_spatial_projection = np.nansum(data_cropped, axis=1)  # Integrate over Y (Top plot: RM vs X)
    spatial_spatial_projection = np.nansum(data_cropped, axis=0)  # Integrate over RM (Bottom plot: X vs Y)
    
    # === Setup WCS for the spatial-spatial projection ===
    wcs_2d = wcs.celestial
    
    # === Plot the results ===
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # --- Top Plot: RM vs. X ---
    ax1 = fig.add_subplot(2, 1, 1)
    cmap_top = ax1.imshow(rm_spatial_projection, aspect='auto', origin='lower',
                           extent=[x_min_pix, x_max_pix, rm_min_pix if rm_min_pix else 0, rm_max_pix if rm_max_pix else data.shape[0]],
                           cmap='magma', norm=LogNorm(vmax=np.nanmax(rm_spatial_projection), 
                                                      vmin=np.nanmax(rm_spatial_projection) * 1e-3))
    ax1.set_ylabel("Faraday Depth (RM)")
    ax1.set_xticks([])  # Remove x-axis tick labels
    fig.colorbar(cmap_top, ax=ax1, label="Integrated Intensity")
    
    # --- Bottom Plot: X vs. Y with WCS Grid ---
    ax2 = fig.add_subplot(2, 1, 2, projection=wcs_2d, aspect='equal')
    im = ax2.imshow(spatial_spatial_projection, origin='lower', cmap='magma',
                     extent=[x_min_pix, x_max_pix, y_min_pix, y_max_pix], norm=LogNorm(vmax=np.nanmax(spatial_spatial_projection), 
                                                                  vmin=np.nanmax(spatial_spatial_projection) * 1e-3))
    ax2.coords.grid(True, color='blue', linestyle='--', alpha=0.8, linewidth=1.0)
    ax2.set_xlabel("RA")
    ax2.set_ylabel("Dec")
    fig.colorbar(im, ax=ax2, label="Integrated Intensity")
    
    plt.subplots_adjust(hspace=0.05)  # Reduce space between plots
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a rotated RA-Dec-RM cube with integrated projections.")
    parser.add_argument("fits_file", type=str, help="Path to the input FITS file.")
    parser.add_argument("x_min_pix", type=int, help="Minimum pixel for X-axis (RA direction).")
    parser.add_argument("x_max_pix", type=int, help="Maximum pixel for X-axis (RA direction).")
    parser.add_argument("y_min_pix", type=int, help="Minimum pixel for Y-axis (Dec direction).")
    parser.add_argument("y_max_pix", type=int, help="Maximum pixel for Y-axis (Dec direction).")
    parser.add_argument("--rm_min_pix", type=int, default=None, help="Minimum pixel for RM axis (Faraday depth).")
    parser.add_argument("--rm_max_pix", type=int, default=None, help="Maximum pixel for RM axis (Faraday depth).")
    parser.add_argument("--fig_width", type=float, default=10, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=5, help="Figure height in inches.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the output figure (optional).")
    
    args = parser.parse_args()
    
    visualize_rm_cube(args.fits_file, args.x_min_pix, args.x_max_pix, args.y_min_pix, args.y_max_pix, args.rm_min_pix, args.rm_max_pix, args.fig_width, args.fig_height, args.save_path)
