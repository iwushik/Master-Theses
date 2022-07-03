import numpy as np

def filter_stars(stars):
	new_stars = []
	for star in stars:
		if np.mean(star.magnitudes) <= 18.5:
			new_stars.append(star)

	return new_stars
	