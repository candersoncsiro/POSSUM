#!/usr/bin/env python3

# Standard library imports
import argparse

# Related third-party imports
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits

docstring =	"""
generate_pseudo3D.py. Written by Craig Anderson and Larry Rudnick in Feb 2024 and described in Rudnick et al., 2024, MNRAS, submitted.

Generates a FITS cube from FITS maps of polarized intensity and Faraday depth, with user-specified Gaussian smoothing in Faraday depth.  The output FITS cube can be viewed in any appropriate software package, such as ds9 or CARTA, or used as input to the visualize_pseudo3D.py script.

Usage:
	python generate_pseudo3D.py p_filename.fits rm_filename.fits output_cube.fits 
--rm_min -100 --rm_max 100 --rm_incr 1 --smooth_sigma 3

Parameters:
	p_filename (str): Path to the polarized intensity FITS file.
	rm_filename (str): Path to the Faraday depth FITS file.
	output_filename (str): Path for the output FITS cube file.
	rm_min (int, rad/m^2): Minimum Faraday depth value.
	rm_max (int, rad/m^2): Maximum Faraday depth value.
	rm_incr (float, rad/m^2): Pixel size in Faraday depth space.
	smooth_sigma (odd integer, channels): Smoothing factor (standard deviation of the Gaussian kernel) applied along  the Faraday depth axis for the pseudo 3D representation.

The file names have no defaults.
					  
NOTE: This script has been tested and verified on FITS maps that either have NAXIS=2, or NAXIS=3 with a singleton degenerate 'LINEAR' type axis (shape: <4000,<4000,1). Development work to handle other possible cases is most welcome!

Installation requirements:  
numpy, astropy, scipy

"""

def load_and_mask_fits(p_filename, rm_filename, rm_range):
	"""
	Load polarized intensity and Faraday depth FITS files, regrid the images to the beam size,
	and mask the Faraday depth data.

	Parameters:
	p_filename (str): Filename for the polarized intensity FITS file.
	rm_filename (str): Filename for the Faraday depth FITS file.
	rm_range (tuple): A tuple of (min, max) values to use as the mask for the Faraday depth.

	Returns:
	tuple: Tuple containing the regridded polarized intensity data and the masked Faraday depth data.
	"""

	# Load polarized intensity FITS
	try:
		with fits.open(p_filename) as p_hdul:
			p_data = np.squeeze(p_hdul[0].data)
			p_header = p_hdul[0].header

		# Load Faraday depth FITS
		with fits.open(rm_filename) as rm_hdul:
			rm_data = np.squeeze(rm_hdul[0].data)
			rm_header = rm_hdul[0].header
	except FileNotFoundError:
		print(f"Error: Either {p_filename} or {rm_filename} was not found, or is corrupted.")
		return

	### Check size of cubes. Warn if large.

	# Calculate the combined size of the data cubes in bytes
	combined_size_bytes = p_data.nbytes + rm_data.nbytes
	# Convert bytes to gigabytes
	combined_size_gb = combined_size_bytes / (1024**3)

	# Check if the combined size exceeds 24 GB
	if combined_size_gb > 12:
		print("\n Warning: The combined size of the input FITS cubes is %.2f GB."%combined_size_gb)
		print("This may lead to significant memory usage and potentially slow processing and output cube writing on some machines.\n")

	### Pre-filter RM map. Could think of something smarter to do here. Basically just getting rid of NaNs, forcing P to zero where RM is out of range. Should at least insert warnings. 

	# Replace NaNs with 0 before clipping
	rm_data_no_nans = np.nan_to_num(rm_data, 0.0)

	# Create a mask where RM is out of the specified range and not exactly equal to zero --- e.g. when changed from NaNs above
	RM_out_of_range_or_exactly_zero_mask = ( (rm_data_no_nans < rm_range[0]) | (rm_data_no_nans > rm_range[1]) | (rm_data_no_nans==0) )

	# Where the RM is out of range, set the P image values to zero
	p_data[RM_out_of_range_or_exactly_zero_mask] = 0

	return p_data, rm_data, p_header

def generate_3d_rm_map(p_data, rm_data, rm_range, rm_incr, smooth_sigma):
	"""
	Generates a pseudo-3D Faraday depth (FD) map from polarized intensity and Faraday depth data.

	This function rounds the Faraday depth data to the nearest increment specified by rm_incr, masks 
	out values outside the specified rm_range, and assigns polarized intensity values to their corresponding
	Faraday depth "slices" to construct a 3D representation. It then applies Gaussian smoothing along the
	Faraday depth axis to improve visual clarity.

	Parameters:
	- p_data (ndarray): A 2D numpy array of polarized intensity data.
	- rm_data (ndarray): A 2D numpy array of Faraday depth data, corresponding to p_data.
	- rm_range (tuple): A tuple specifying the minimum and maximum Faraday depth values to include.
	- rm_incr (float): The increment of Faraday depth values, defining the "thickness" of each slice.
	- smooth_sigma (float): The standard deviation for the Gaussian kernel used in smoothing along the
	  Faraday depth axis.

	Returns:
	- ndarray: A 3D numpy array representing the pseudo-3D Faraday depth map, with the third dimension
	  corresponding to Faraday depth.
	"""

	# Round the Faraday depth data to the nearest specified increment
	rm_data_rounded = np.round(rm_data / rm_incr) * rm_incr

	# Calculate the number of Faraday depth steps needed to cover the specified range
	rm_steps = int((rm_range[1] - rm_range[0]) / rm_incr) + 1

	# Initialize an empty 3D array for the pseudo-3D Faraday depth map
	pseudo_3D_RM_map_unnormalised = np.zeros((p_data.shape[0], p_data.shape[1], rm_steps))

	# Populate the 3D map with polarized intensity values, assigning them to the correct Faraday depth slice
	print('Generating 3D map...')

	# Create a mask for pixels whose peak FD lies within the specified range
	valid_fd_mask = (rm_data_rounded >= rm_range[0]) & (rm_data_rounded <= rm_range[1])

	# Calculate indices for the Faraday depth slices for all valid pixels
	rm_indices = ((rm_data_rounded[valid_fd_mask] - rm_range[0]) / rm_incr).astype(int)

	# Get the RA and Dec indices of the valid pixels
	ra_indices, dec_indices = np.where(valid_fd_mask)

	# Assign polarized intensity values to the correct Faraday depth slice in the 3D map
	# You need to use advanced indexing here. Note that for each valid pixel, you're setting a value in a specific depth slice
	pseudo_3D_RM_map_unnormalised[ra_indices, dec_indices, rm_indices] = p_data[ra_indices, dec_indices]

	# Apply Gaussian smoothing to the 3D map along the Faraday depth axis
	pseudo_3D_RM_map_unnormalised_smoothed = gaussian_filter(pseudo_3D_RM_map_unnormalised, sigma=(0, 0, smooth_sigma))

	## Re-normalise the output cube. 

	# Initialize the renormalized map with the same shape as the original, filled with NaNs or zeros
	pseudo_3D_RM_map = np.full_like(pseudo_3D_RM_map_unnormalised_smoothed, 0)

	## Re-normalise

	print('Re-normalising 3D map...')

	# Step 1: Create a mask for valid p_data values
	valid_mask = (p_data > 0) & ~np.isnan(p_data)

	# Step 2: Find the maximum value along the RM dimension for each ra_pix, dec_pix location
	# Ensure that the shape of max_values is broadcastable over pseudo_3D_RM_map_unnormalised_smoothed
	max_values = np.nanmax(pseudo_3D_RM_map_unnormalised_smoothed, axis=2, keepdims=True)

	# Step 3: Apply renormalization
	# First, prevent division by zero or NaN max_values by setting them to 1 (or another suitable value)
	# This operation ensures that you do not modify locations with max_values of zero or NaN
	max_values[max_values == 0] = 1  # Prevent division by zero
	max_values[np.isnan(max_values)] = 1  # Optional: Handle NaN max_values if they can occur

	# Perform the renormalization where valid_mask is True
	# This step multiplies by p_data, considering the broadcasting rules, p_data needs to match in dimensions
	pseudo_3D_RM_map = np.where(valid_mask[..., np.newaxis],  # Expand dims to match pseudo_3D_RM_map_unnormalised_smoothed
								pseudo_3D_RM_map_unnormalised_smoothed / max_values * p_data[..., np.newaxis],  # Apply renormalization
								pseudo_3D_RM_map_unnormalised_smoothed)  # Else, keep the original values

	return pseudo_3D_RM_map

def save_to_fits(cube, output_filename, rm_range, rm_incr, p_header):
	"""
	Saves the generated 3D Faraday depth map to a FITS file, propagating relevant header information
	from the input polarized intensity and Faraday depth data, and incorporating Faraday depth axis
	specifics into the FITS header.

	Parameters:
	- cube (ndarray): The 3D numpy array representing the pseudo-3D Faraday depth map to be saved.
	- output_filename (str): The path and filename for the output FITS file.
	- rm_range (tuple): The minimum and maximum Faraday depth values, used to set the CRVAL3 header.
	- rm_incr (float): The Faraday depth increment, defining the pixel size along the Faraday depth axis,
					   used to set the CDELT3 header.
	- p_header (fits.Header): The FITS header from the polarized intensity data file, from which the RA and
							  Dec axis information will be propagated to the new FITS header.

	This function directly modifies the header of the output FITS file to include the Faraday depth axis
	configuration and copies relevant RA and Dec information from the input data's header.
	"""

	# Transpose the cube to match FITS conventions
	cube_transposed = np.transpose(cube, (2, 0, 1))

	# Create a Primary HDU object from the transposed data cube
	hdu = fits.PrimaryHDU(cube_transposed)

	# Use P_header as a template for correct wcs coords etc, which we then update to reflect 3D cube
	hdr = p_header.copy()
	hdr_updated = update_header_for_3d(hdr, cube_transposed.shape, rm_range, rm_incr)

	# Now, hdr_updated contains the correctly ordered header for the 3D cube
	# Proceed with assigning it to your HDU and saving the FITS file
	hdu = fits.PrimaryHDU(data=cube_transposed, header=hdr_updated)
	print('Saving outfile: %s.'%output_filename)
	hdu.writeto(output_filename, overwrite=True)

def update_header_for_3d(hdr, cube_shape, rm_range, rm_incr):
    """
    Update the FITS header for a 3D cube, ensuring correct card placement. Otherwise Astropy throws an error and quits.
    
    Parameters:
    - hdr: The original 2D FITS header to be updated.
    - cube_shape: The shape of the 3D data cube (depth, height, width).
    - rm_range: The range of the Faraday depth values.
    - rm_incr: The increment of the Faraday depth values.
    
    Returns:
    - Updated FITS header for the 3D cube.
    """
    # Ensure NAXIS is correct for 3D data
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = cube_shape[2]
    hdr['NAXIS2'] = cube_shape[1]
    
    # Properly insert NAXIS3 after NAXIS2 to maintain correct order, otherwise Astropy FITS verify throws an error 
    if 'NAXIS3' in hdr:
        hdr['NAXIS3'] = cube_shape[0]  # Update existing card if it already exists
    else:
        hdr.insert('NAXIS2', ('NAXIS3', cube_shape[0]), after=True)  # Insert after NAXIS2
    
    # Update or insert the Faraday depth axis information
    hdr['CTYPE3'] = 'FARADAY'
    hdr['CDELT3'] = rm_incr
    hdr['CRVAL3'] = rm_range[0]
    hdr['CRPIX3'] = 1
    
    ## Placeholder code thus far found to be unnecessary, but which could in principle be needed to ensure FITS compatability
    # Ensure EXTEND is correctly placed if necessary
    if 'EXTEND' in hdr:
        # If EXTEND exists, ensure it's correctly positioned; this might be more complex depending on header structure
        pass  # Implementation depends on existing header structure and requirements
    
    # Remove any irrelevant keys
    # hdr.remove('INVALID_KEY', ignore_missing=True)
    
    return hdr

def main(p_filename, rm_filename, output_filename, rm_min, rm_max, rm_incr, smooth_sigma):

	# Input Validation
	if rm_min >= rm_max:
		print("Error: 'rm_min' should be less than 'rm_max'.")
		return  # Exit the function early

	if rm_incr <= 0:
		print("Error: 'rm_incr' should be greater than 0.")
		return  # Exit the function early

	if smooth_sigma <= 0:
		print("Error: 'smooth_sigma' should be greater than 0.")
		return  # Exit the function early

	# Run processes
	rm_range = (rm_min, rm_max)
	p_data, rm_data, p_header = load_and_mask_fits(p_filename, rm_filename, rm_range)
	cube = generate_3d_rm_map(p_data, rm_data, rm_range, rm_incr, smooth_sigma)
	save_to_fits(cube, output_filename, rm_range, rm_incr, p_header)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=docstring, formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('p_filename', help='Polarized intensity FITS file path. Data must have NAXIS=3 (shape: 1, 4000, 4000), or script will fail.')
	parser.add_argument('rm_filename', help='Faraday depth FITS file path. Data must have NAXIS=3 (shape: 1, 4000, 4000), or script will fail.')
	parser.add_argument('output_filename', help='Output FITS cube file path')
	parser.add_argument('--rm_min', type=int, default=-100, help='Minimum Faraday depth (rad/m^2)')
	parser.add_argument('--rm_max', type=int, default=100, help='Maximum Faraday depth (rad/m^2)')
	parser.add_argument('--rm_incr', type=float, default=1.0, help='Pixel increment size in Faraday depth space (rad/m^2)')
	parser.add_argument('--smooth_sigma', type=float, default=3.0, help='Gaussian smoothing sigma for the Faraday depth axis (number of chans)')

	args = parser.parse_args()

	main(args.p_filename, args.rm_filename, args.output_filename, args.rm_min, args.rm_max, args.rm_incr, args.smooth_sigma)

