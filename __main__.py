import argparse

import ioutil
import filters
import data
from data import Star


def build_parser():
	main_parser = argparse.ArgumentParser(description = "Program to classify, evaluate and plot data about multisystem detection.")
	subparsers = main_parser.add_subparsers(dest = "subcommand")
	subparsers.required = True

	def common_sub(subparser):
		subparser.add_argument("DATASETS", nargs = "+", help = "Paths to star datasets")
		subparser.add_argument("--annotations", "-a", dest = "ANNOTATIONS", nargs = "*", help = "Paths to annotation lists") 

	sub_run = subparsers.add_parser("run", help = "Make a list of suspicious system")
	common_sub(sub_run)
	sub_run.set_defaults(subcommand_function = run)

	return main_parser

def run(args):
	stars = ioutil.read_stars(args.DATASETS, args.ANNOTATIONS)
	stars = filters.filter_stars(stars)

	for star in stars:
		name, star.mean_magnitude, star.standard_deviation, star.magnitudes_from_std, star.phases_from_std, star.times_from_std, star.linear_model, star.average_std = data.std_wo_min(star)
		star.average_area_diffrence, star.mean_magnitude = data.area(star)
		star.area_down, star.mean_magnitude = data.under_std(star)

	mean_magnitudes_std, standard_deviations_std, fit_y_std, names_multiple_stars_std, magnitude_multiple_stars_std, annotation_std = data.std_for_plot(stars)

	mean_mag_area, area_diffrence_area, fit_y_area, names_multiple_stars_area, magnitude_multiple_stars_area, annotation_area = data.area_plot(stars)

	mean_mag_under_std, areas_down, fit_y_under_std, names_multiple_stars_under_std, magnitude_multiple_stars_under_std, annotation_under_std = data.under_std_plot(stars)

	mean_magnitudes_ratio_mag, dif_mag_ratio_mag, fit_y_ratio_mag, names_multiple_stars_ratio_mag, magnitude_multiple_stars_ratio_mag, annotation_ratio_mag = data.ratio_mean_mag_for_plot(stars)
	
	mean_magnitudes_ratio, standard_deviations_ratio, fit_y_ratio, names_multiple_stars_ratio, magnitude_multiple_stars_ratio, star.annotation_ratio = data.ratio_std_for_plot(stars)

	names_multiple = [star.name for star in stars if star.name in names_multiple_stars_area and star.name in names_multiple_stars_std and star.name in names_multiple_stars_ratio and star.name in names_multiple_stars_ratio_mag and star.name in names_multiple_stars_under_std]
	magnitude_multiple =[star.mean_magnitude for star in stars if star.name in names_multiple_stars_area and star.name in names_multiple_stars_std and star.name in names_multiple_stars_ratio and star.name in names_multiple_stars_ratio_mag and star.name in names_multiple_stars_under_std]
	annotation_multiple = []

	for star in stars:
		if star.name in names_multiple_stars_area and star.name in names_multiple_stars_std and star.name in names_multiple_stars_ratio and star.name in names_multiple_stars_ratio_mag and star.name in names_multiple_stars_under_std:
			annotation_multiple.append("True")


	ioutil.write_names("./multiple_stars.txt",  names_multiple, magnitude_multiple, annotation_multiple)

def main():
	parser = build_parser()
	args = parser.parse_args()
	ioutil.log(args)

	args.subcommand_function(args)

main()