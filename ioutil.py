import csv
import sys
import os.path

import numpy as np

from data import Star

def log(*args, **kwargs):
	kwargs["file"] = sys.stderr
	print(*args, **kwargs)

TRUE_STR = ["true", "yes", "1", "y"]
FALSE_STR = ["false", "no", "0", "n", "none"]
csv.register_dialect("my_dialect", delimiter = " ", skipinitialspace = True)
CSV_DIALECT = csv.get_dialect("my_dialect")

def path_to_star(name):
	return name + ".dat"

def path_to_photometry(base):
	return os.path.join(base, "phot", "I")

def resolve_photometry_path(base, star_name):
	path = os.path.join(base, path_to_star(star_name))
	if not os.path.isfile(path):
		new_path = os.path.join(path_to_photometry(base), path_to_star(star_name))
		if os.path.isfile(new_path):
			return new_path

	return path

def path_to_periods(base):
	return os.path.join(base, "ecl.dat")

def resolve_periods_path(path):
	if not os.path.isfile(path):
		new_path = path_to_periods(path)
		if os.path.isfile(new_path):
			return new_path
	
	return path

def path_to_primary_minimum(base):
	return os.path.join(base, "ecl.dat")

def resolve_primary_minimum_path(path):
	if not os.path.isfile(path):
		new_path = path_to_primary_minimum(path)
		if os.path.isfile(new_path):
			return new_path
	
	return path

def read_period_map(dataset_path):
	"""
	Reads, parses and returns period map

	Expected csv format:
	name ? ? period
	"""

	dataset_path = resolve_periods_path(dataset_path)

	result_map = {}

	with open(dataset_path, "r", newline = "") as file:
		reader = csv.reader(file, dialect = CSV_DIALECT)
		for row in reader:
			if len(row) == 0:
				continue

			name = row[0]
			
			period = row[3]
			try:
				period = np.float64(period)
			except ValueError as err:
				print(row)
				raise RuntimeError(f"Invalid period value \"{period}\": {err}")
			
			result_map[name] = period
	
	return result_map

def read_primary_minimum_map(dataset_path):
	"""
	Reads, parses and returns primary minimum map

	Expected csv format:
	name ? ? ? primary minimum
	"""

	dataset_path = resolve_primary_minimum_path(dataset_path)

	result_map = {}

	with open(dataset_path, "r", newline = "") as file:
		reader = csv.reader(file, dialect = CSV_DIALECT)
		for row in reader:
			if len(row) == 0:
				continue

			name = row[0]
			
			primary_minimum = row[4]
			try:
				primary_minimum = np.float64(primary_minimum)
			except ValueError as err:
				print(row)
				raise RuntimeError(f"Invalid primary minimum value \"{primary_minimum}\": {err}")
			
			result_map[name] = primary_minimum
	
	return result_map

def read_annoation_map(path):
	"""
	Reads, parses and returns annotation map

	Expected csv format:
	name ? annotation
	"""

	result_map = {}

	with open(path, "r", newline = "") as file:
		reader = csv.reader(file, dialect = CSV_DIALECT)
		for row in reader:
			name = row[0]
			
			annotation = row[2].strip().lower()
			if annotation in TRUE_STR:
				annotation = True
			elif annotation in FALSE_STR:
				annotation = False
			else:
				raise RuntimeError(f"Invalid annotation value \"{annotation}\"")
			
			result_map[name] = annotation
	
	return result_map

def read_star_data(path):
	"""
	Reads, parses and returns data of one star

	Expected csv format:
	time magnitude magnitude_error
	"""

	times = []
	magnitudes = []
	magnitude_errors = []

	with open(path, "r", newline = "") as file:
		reader = csv.reader(file, dialect = CSV_DIALECT)
		for row in reader:
			time = row[0]
			try:
				time = np.float64(time)
			except ValueError as err:
				raise RuntimeError(f"Invalid time value \"{time}\": {err}")

			magnitude = row[1]
			try:
				magnitude = np.float64(magnitude)
			except ValueError as err:
				raise RuntimeError(f"Invalid magnitude value \"{magnitude}\": {err}")

			magnitude_error = row[2]
			try:
				magnitude_error = np.float64(magnitude_error)
			except ValueError as err:
				raise RuntimeError(f"Invalid magnitude_error value \"{magnitude_error}\": {err}")

			times.append(time)
			magnitudes.append(magnitude)
			magnitude_errors.append(magnitude_error)
	
	return np.array(times, dtype = np.float64), np.array(magnitudes, dtype = np.float64), np.array(magnitude_errors, dtype = np.float64)

def read_stars_by_names(names, datset_paths):
	"""
	Reads stars from datset_paths of given name and concatenates their data into Star object
	"""
	stars = []

	for name in names:
		times = []
		magnitudes = []
		magnitude_errors = []
		for dataset_path in datset_paths:
			path = resolve_photometry_path(dataset_path, name)

			try:
				times_part, magnitudes_part, magnitude_errors_part = read_star_data(path)

			except:
				times = None
				log(f"Warning: Cannot read \"{path}\", skipping star {name}")
				break
			
			if len(magnitudes) > 0:
				current_mean = np.mean(magnitudes)
				part_mean = np.mean(magnitudes_part)
				diffrence_mean = current_mean - part_mean
				magnitudes_part += diffrence_mean

			times = np.concatenate((times, times_part))
			magnitudes = np.concatenate((magnitudes, magnitudes_part))
			magnitude_errors = np.concatenate((magnitude_errors, magnitude_errors_part))
		
		if times is not None:
			std = np.std(magnitudes)
			mean_mag = np.mean(magnitudes)
			for i in range(len(magnitudes)):
				if	magnitudes[i] < mean_mag-2*std:
					#magnitudes = np.delete(magnitudes, i)
					#times = np.delete(times, i)
					#magnitude_errors = np.delete(magnitude_errors, i)

					magnitudes[i] = None
					times[i] = None
					magnitude_errors[i] = None

			magnitudes = magnitudes[np.logical_not(np.isnan(magnitudes))]
			times = times[np.logical_not(np.isnan(times))]
			magnitude_errors = magnitude_errors[np.logical_not(np.isnan(magnitude_errors))]

			stars.append(
				Star(name, times, magnitudes, magnitude_errors)
			)


	return stars

def read_stars_for_periods(dataset_paths, annotation_paths = None):
	# load period maps for each dataset
	period_maps = []
	for dataset_path in dataset_paths:
		log(f"Reading period map from \"{dataset_path}\"")
		period_map = read_period_map(dataset_path)
		period_maps.append(period_map)

	# load annotation map
	annotation_map = None
	if annotation_paths is not None:
		annotation_map = {}
		for annotation_path in annotation_paths:
			log(f"Reading annotation map from \"{annotation_path}\"")
			anot_map = read_annoation_map(annotation_path)
			annotation_map = annotation_map | anot_map

	names = set(period_maps[0].keys())
	for period_map in period_maps[1:]:
		names = names & set(period_map.keys())
	if annotation_map is not None:
		names = names & set(annotation_map.keys())
	stars = read_stars_by_names(names, dataset_paths)

	for star in stars:
		periods = [pmap[star.name] for pmap in period_maps]
		star.period = sum(periods) / len(periods)

	return stars

def read_stars(dataset_paths, annotation_paths = None):
	"""
	Reads periods, (optionally) annotations and stars. Star names are chosen as
	the intersection of all read period and annotation maps.
	"""

	# load period maps for each dataset
	period_maps = []
	for dataset_path in dataset_paths:
		log(f"Reading period map from \"{dataset_path}\"")
		period_map = read_period_map(dataset_path)
		period_maps.append(period_map)

	# load primary minimum maps for each dataset
	primary_minimum_maps = []
	for dataset_path in dataset_paths:
		log(f"Reading primary minimum map from \"{dataset_path}\"")
		primary_minimum_map = read_primary_minimum_map(dataset_path)
		primary_minimum_maps.append(primary_minimum_map)


	# load annotation map
	annotation_map = None
	if annotation_paths is not None:
		annotation_map = {}
		for annotation_path in annotation_paths:
			log(f"Reading annotation map from \"{annotation_path}\"")
			anot_map = read_annoation_map(annotation_path)
			annotation_map = annotation_map | anot_map
	
	# select names by intersecting all available names from all maps
	names = set(period_maps[0].keys())
	for period_map in period_maps[1:]:
		names = names & set(period_map.keys())
	if annotation_map is not None:
		names = names & set(annotation_map.keys())
	stars = read_stars_by_names(names, dataset_paths)

	# calculate star phases from loaded datasets
	for star in stars:
		star.calculate_phases(primary_minimum_maps, period_maps=period_maps)
		if annotation_map is not None:
			star.annotation.truth = annotation_map[star.name]

		star.sort_by_phases()
		
	return stars

def read_name_stars(path):
	names = []
	with open(path, "r", newline = "") as file:
		reader = csv.reader(file, dialect = CSV_DIALECT)
		for row in reader:
			if len(row) == 0:
				continue

			name = row[0]
			names.append(name)

	return names

def write_names(path, names, mags, annotations):
	with open(path, "w") as file:
		for s in range(len(names)):
			print(
				f"{names[s]} {mags[s]} {annotations[s]}",
				file = file
			)
