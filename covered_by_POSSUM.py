#!/usr/bin/env python3

# Standard library imports
import argparse
import re

# Related third-party imports
import numpy as np
from numpy import ma  # ma is short for Masked Array
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, Column

docstring = """
This script processes sky coordinates provided by the user, matches them against completed POSSUM (Polarization Sky Survey of the Universe's Magnetism) survey observations, and identifies beams numbers and SBIDS that correspond to these positions. It utilizes publicly available data from Google Sheets for survey progress and beam positions, performing cross-matching to return the nearest POSSUM observations to the input coordinates.

Input: A list of coords. One per line. Can be in sexagesimal or decimal format. Can be command separated or space separated. 
Output: Currently results are just printed to terminal. 

Features:
- Reads and parses user-provided coordinates from a text file, supporting both sexagesimal and decimal formats. One coord per line.
- Fetches survey progress data and beam positions from specified CSV URLs published on Google Sheets.
- Processes the fetched data to add SkyCoord objects, format field names, and filter based on specific criteria such as SBID validation and observation status.
- Performs cross-matching between the user's coordinates and the POSSUM beam positions, identifying the nearest beams and providing details about the match.
- Outputs the cross-matching results, indicating whether the provided coordinates have been observed, and details about the closest beams.

Usage:
The script is executed from the command line, requiring the path to a text file containing the coordinates as the input argument. Coordinates can be listed one per line in either sexagesimal or decimal format. The script then processes this input, accesses the required data from online sources, performs the cross-matching, and prints the results to the console.

Example Usage:
python covered_by_POSSUM.py /path/to/coordinates.txt

Dependencies:
- astropy: For astronomical calculations and handling of coordinates.
- pandas: For processing data fetched from CSV URLs.
- numpy: For numerical operations, especially handling masked arrays.

Note:
Ensure that the Google Sheets API access and OAuth2 credentials are correctly configured for the script to fetch the necessary data.

Author: Craig Anderson
Date: March 2024
Version: 0.0.1
"""

### Functions

def read_coordinates(file_path):
	"""
	Reads coordinates from a specified file and returns them as a list.

	Each line in the file is expected to represent a single coordinate,
	which is stripped of leading and trailing whitespace.

	Parameters:
	- file_path (str): The path to the text file containing coordinates.

	Returns:
	- list: A list of strings, where each string is a coordinate read from the file.
	"""
	with open(file_path, 'r') as file:
		coordinates = [line.strip() for line in file]
	return coordinates

def load_and_parse_coordinates(file_path):
	"""
	Reads and parses coordinates from a file into SkyCoord objects.

	Supports coordinates in both sexagesimal and decimal formats.

	Parameters:
	- file_path (str): Path to the file containing the coordinates.

	Returns:
	- list: A list of SkyCoord objects representing the parsed coordinates.
	"""
	ra_dec = read_coordinates(file_path)
	candidate_coords = []

	for coord in ra_dec:
		# Check if the input is in sexagesimal format by looking for colons
		if ':' in coord:
			ra, dec = coord.split(' ')
			candidate = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='icrs')
		# Check if the input is in decimal degree format
		elif re.match(r'^[0-9.+\- ,]+$', coord):
			ra, dec = map(float, coord.replace(',',' ').split())
			candidate = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
		else:
			print(f"Could not recognize coordinates in the following string from user input: \'{coord}\'")
			continue

		candidate_coords.append(candidate)

	print(f'\nMatching against {len(candidate_coords)} successfully read-in coordinates...\n')
	return candidate_coords

def fetch_published_sheet_data(csv_url):
	"""
	Fetches data from a publicly published Google Sheet CSV URL and returns it as a pandas DataFrame.

	Parameters:
	- csv_url (str): The URL of the published Google Sheet in CSV format.

	Returns:
	- DataFrame: A pandas DataFrame containing the sheet data.
	"""
	try:
		df = pd.read_csv(csv_url)
		return df
	except Exception as e:
		print(f"Failed to fetch or parse the CSV data: {e}")
		return None

def load_and_process_survey_progress(csv_url):
	"""
	Fetches and processes the survey progress data from a CSV URL.

	The processing steps include converting the data to an Astropy Table,
	adding SkyCoord objects, formatting field names, and filtering the table
	based on specific criteria.

	Parameters:
	- csv_url (str): The URL of the CSV file containing the survey progress data.

	Returns:
	- Table: An Astropy Table containing the processed survey progress data.
	"""
	print('\nINFO: Accessing online resources...')
	sheet_data = fetch_published_sheet_data(csv_url)

	if sheet_data is not None:
		survey_progress_table = Table.from_pandas(sheet_data)

		# Add skycoord objects for 'ra' and 'dec'
		ra_sexagesimal = survey_progress_table['ra']
		dec_sexagesimal = survey_progress_table['dec']
		skycoords_sexagesimal = SkyCoord(ra=ra_sexagesimal, dec=dec_sexagesimal, unit=(u.hourangle, u.deg), frame='icrs')
		survey_progress_table['skycoords'] = skycoords_sexagesimal

		# Format Cameron's field names to match Lerato's
		new_column_data = [name.split("EMU_")[1] if "EMU_" in name else name for name in survey_progress_table['name']]
		new_column = Column(data=new_column_data, name='Lerato_field_names')
		survey_progress_table.add_column(new_column)

		# Identify observational progress by filtering rows
		sbid_is_integer_mask = [False if value is np.ma.masked else str(value).split('.')[0].isdigit() for value in survey_progress_table['sbid']]
		not_rejected_mask = ['REJECTED' not in value for value in survey_progress_table['validated']]
		not_empty_processed_mask = [False if ma.is_masked(value) else value != '' for value in survey_progress_table['processed']]		
		combined_mask = [int_mask and not_rej_mask and not_empty_proc for int_mask, not_rej_mask, not_empty_proc in zip(sbid_is_integer_mask, not_rejected_mask, not_empty_processed_mask)]
		filtered_survey_progress_table = survey_progress_table[combined_mask]

		print('INFO: Done processing survey progress data.')
		return filtered_survey_progress_table
	else:
		print("Failed to load data from the published sheet.")
		return None

def load_possum_band1_beam_positions(csv_url):
	"""
	Loads beam positions from a publicly accessible Google Sheet CSV URL, converts it to an Astropy Table,
	and enriches it with SkyCoord objects for each beam's RA and Dec.

	Parameters:
	- csv_url (str): The URL to the CSV file containing beam positions.

	Returns:
	- Astropy Table: A table containing the original data plus a 'skycoords' column with SkyCoord objects.
	"""
	print('INFO: Loading beam positions...')
	sheet_data_beams = fetch_published_sheet_data(csv_url)

	if sheet_data_beams is not None:
		try:
			# Convert DataFrame to Astropy Table for consistency with the rest of your code
			beam_positions_table = Table.from_pandas(sheet_data_beams)

			# Extract the RA and Dec columns
			ra = beam_positions_table['beam center ra [deg]'] * u.deg
			dec = beam_positions_table['beam center dec [deg]'] * u.deg

			# Create a SkyCoord object using the RA and Dec
			skycoords = SkyCoord(ra=ra, dec=dec, frame='icrs')

			# Add the SkyCoord object as a new column to the table
			beam_positions_table['skycoords'] = skycoords

			print('INFO: Beam positions loaded successfully.')
			return beam_positions_table

		except Exception as e:
			print('Beam position loading FAILED! Reason:')
			print(e)
			return None
	else:
		raise ValueError('No data available for beam positions, or the beam data was not read in successfully from online resources. The script cannot proceed.')

def perform_cross_matching(candidate_coords, observed_positions, separation_threshold=1.6):
	"""
	Perform cross-matching between candidate coordinates and observed POSSUM survey positions.

	Parameters:
	- candidate_coords (list of SkyCoord): Candidate coordinates to cross-match.
	- observed_positions (Astropy Table): Table containing observed POSSUM survey positions with 'skycoords', 'beam no', 'SB', and 'SBID' columns.
	- separation_threshold (float): Threshold in degrees to consider a candidate as observed.

	Returns:
	- list: A list containing strings with cross-matching results for each candidate.
	"""

	observed_candidates = []
	# Iterate over the candidate coordinates
	for index, candidate in enumerate(candidate_coords):
		output_string = ""
		# Convert coordinates to string format for display
		ra_str = candidate.ra.to_string(unit='hourangle', sep='', precision=0, pad=True)
		dec_str = candidate.dec.to_string(sep='', precision=0, alwayssign=True, pad=True)
		candidate_name = f"Candidate {index} at location J{ra_str}{dec_str}"
		print(f'INFO: Checking {candidate_name}...\n')

		# Calculate the on-sky separations
		separations = candidate.separation(observed_positions['skycoords'])

		# Find the indices of the three nearest separations
		nearest_indices = separations.argsort()[:3]
		nearest_separation_deg = separations[nearest_indices[0]].degree

		# Extract the 'beam no', 'SB', and 'SBID' values for the nearest three indices
		nearest_beams_candidate = observed_positions['beam no'][nearest_indices]
		nearest_SB_candidate = observed_positions['SB'][nearest_indices]
		nearest_SBID_candidate = observed_positions['SBID'][nearest_indices]
		nearest_separations_deg_candidate = separations[nearest_indices].degree

		# Determine observed status based on separation threshold
		if nearest_separation_deg < separation_threshold:
			status = 'OBSERVED'
			output_string += f"Status: {status}\n"
			for i in range(3):
				output_string += f"Tile = {nearest_SB_candidate[i]}, SBID = {nearest_SBID_candidate[i]}, Beam No. = {nearest_beams_candidate[i]}, Beam Sep. = {nearest_separations_deg_candidate[i]:.3f} degrees\n"
		else:
			status = 'NOT observed'
			output_string += f"Status: {status}\n---------- <<Nearest observation {nearest_separations_deg_candidate[0]:.1f} degrees away>> ----------\n"

		print(output_string)
		if 'OBSERVED' in output_string:
			observed_candidates.append(output_string)

	if len(candidate_coords) == 0:
		print('BIG PROBLEM -- Looks like the coord list is empty (or the regex match failed)?')
	elif len(candidate_coords) != 0 and len(observed_candidates) == 0:
		print('NO MATCHES :(')

	return observed_candidates

### MAIN

def main(file_path):

	"""
	Load in the sky coords we want to match against (i.e. those supplied by user).
	"""    
	candidate_coords = load_and_parse_coordinates(file_path)

	"""
	Pull in POSSUM survey progress for band 1, which is a publicly available and regularly updated version of Cameron's survey tracking sheet
	"""
	# Load survey progress from online resources 
	POSSUM_progress_csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSGdbCKI4BeA66fSWbiezMKlIFDkMqmscUQyLBbJkqRoNpYyTWmcgoV0cNWEy8mybY2KhNP6of8EQJG/pub?gid=711072015&single=true&output=csv"
	filtered_survey_progress_table = load_and_process_survey_progress(POSSUM_progress_csv_url)

	"""
	Pull in Lerato's positions for all beams in the survey (band 1)
	"""
	# Load beam positions for band 1 survey from online resources 
	POSSUM_band1_beams_csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS3RSYoKuPaXeKszfbmQkN5f-_mXSBcQaIDRT-7TmtzZ2mB_2iLP9ON-VCVQnnuX-hJfZ0CHstcKb79/pub?output=csv"
	POSSUM_survey_B1_all_beam_locs = load_possum_band1_beam_positions(POSSUM_band1_beams_csv_url)

	"""
	The naming conventions for tiles / SBIDs etc differ in different resources. So some jiggery-pokery is needed. 
	Here, filter the all-beam-position table (from Lerato's mapping), so that only beams that have been observed (from the survey progress spreadsheet) now appear
	"""

	# Get the values from the 'Lerato_field_names' column in the filtered_survey_progress_table
	lerato_field_names = set(filtered_survey_progress_table['Lerato_field_names'])

	# Create a mask where each value in the 'SB' column is checked if it' is in 'lerato_field_names'
	mask = [value in lerato_field_names for value in POSSUM_survey_B1_all_beam_locs['SB']]

	# Apply the mask to the POSSUM_survey_B1_all_beam_locs
	observed_POSSUM_survey_B1_all_beam_locs = POSSUM_survey_B1_all_beam_locs[mask]

	"""
	Map the actual SBIDs back from Cameron's table into Lerato's table
	"""

	# Create a dictionary that maps Lerato_field_names to SBIDs from actual observations
	lerato_to_sbid_mapping = {lerato_field_name: sbid for lerato_field_name, sbid in zip(filtered_survey_progress_table['Lerato_field_names'], filtered_survey_progress_table['sbid'])}

	# Use a list comprehension to create a new list of SBID values that correspond to the SB values in observed_POSSUM_survey_B1_all_beam_locs
	sbid_values = [lerato_to_sbid_mapping.get(sb_value, 'N/A') for sb_value in observed_POSSUM_survey_B1_all_beam_locs['SB']]

	# Create a new column with these values and add it to observed_POSSUM_survey_B1_all_beam_locs
	observed_POSSUM_survey_B1_all_beam_locs['SBID'] = sbid_values

	"""
	X-match candidate coords against beam and SBID. Return nearest 3 matches.
	"""
	observed_candidates = perform_cross_matching(candidate_coords, observed_POSSUM_survey_B1_all_beam_locs)

## Main
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=docstring, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('file_path', type=str, help='Path to the text file containing coordinates in each row.')

	args = parser.parse_args()
	main(args.file_path)
