import math
from dataclasses import dataclass
from typing import Union
import scipy.optimize as op

import numpy as np

@dataclass
class BinaryAnnotation:
	guess: Union[bool, None] = None
	truth: Union[bool, None] = None

	def is_true_positive(self):
		return self.guess == True and self.truth == True
	
	def is_true_negative(self):
		return self.guess == False and self.truth == False

	def is_false_positive(self):
		return self.guess == True and self.truth == False 
	
	def is_false_negative(self):
		return self.guess == False and self.truth == True

	def is_annotated(self):
		return self.truth is not None
	
	def is_guessed(self):
		return self.guess is not None

@dataclass
class Star:
	name: str
	times: list
	magnitudes: list
	magnitude_errors: list
	phases: Union[list, None] = None
	annotation: BinaryAnnotation = None
	standard_deviation: float = None
	magnitudes_from_std: list = None
	phases_from_std: list = None
	times_from_std: list = None
	mean_magnitude: float = None
	linear_model: list = None
	average_area_diffrence: float = None
	area_down: float = None
	period: float = None
	ratio: float = None

	def __post_init__(self):
		if self.annotation is None:
			self.annotation = BinaryAnnotation()

	@staticmethod
	def convert_time_to_phase(period, times, primary_minimum_times):
		return np.array(
			[np.divmod((time - primary_minimum_times) / period, 1)[1] for time in times]
		)

	def calculate_phases(self, primary_minimum_maps, period = None, period_maps = None):
		"""
		Calculates self.phases by taking arithmetic mean of periods from each period map
		and convering self.times.

		Additionally sorts all held data by phase.
		"""

		if period == None: 
			periods = [pmap[self.name] for pmap in period_maps]
			star_period = sum(periods) / len(periods)
		else: star_period = period

		primary_minimums = [pmap[self.name] for pmap in primary_minimum_maps]
		star_primary_minimum = min(primary_minimums)

		self.phases = Star.convert_time_to_phase(star_period, self.times, star_primary_minimum)

	def sort_by_phases(self):
		sorted_indices = np.argsort(self.phases)

		self.phases = self.phases[sorted_indices]
		self.magnitudes = self.magnitudes[sorted_indices]
		self.magnitude_errors = self.magnitude_errors[sorted_indices]
		self.times = self.times[sorted_indices]

def windows(data, width = 100, step = 20, remaining = True):
	"""
	Yields (possibly overlapping) windows of `width` length into `data`.
	
	`remaining` parameter controls whether the last window is allowed to be yielded even if it would be smaller than the other ones.
	"""
	idx = np.arange(width)

	steps_raw = (len(data) - width + step) / step
	steps = math.floor(steps_raw)

	for x in range(steps):
		v = np.array(idx + x * step)
		yield data[v]

	if remaining:
		if steps_raw != steps:
			yield data[(x + 1) * step:]

def std_wo_min(star):
	xs = np.array([np.mean(w) for w in windows(star.phases)])
	std_ys = np.empty(shape = xs.shape)

	quadratic_model = np.polyfit(star.phases, star.magnitudes, 8)
	quadratic_model_y = np.polyval(quadratic_model, xs)

	for (i, w) in enumerate(windows(star.magnitudes)):
		std_ys[i] = np.std(w)

	average_std = np.average(std_ys)
	
	max_model = 0
	max_model_order = 0

	for j in range(len(xs)):
		if (xs[j] > 0.2 and xs[j] < 0.8):
			if max_model < quadratic_model_y[j]: 
				max_model = quadratic_model_y[j]
				max_model_order = j
		
	xs_left_max = xs[max_model_order] - 0.2
	xs_right_max = xs[max_model_order] + 0.2

	std_ys_new =[]

	for j in range(len(xs)):
		if xs_left_max > 0.2:
			if (xs[j] > 0.2 and xs[j] < xs_left_max):
					std_ys_new.append(std_ys[j])
		if xs_right_max < 0.8:
			if (xs[j] > xs_right_max and xs[j] < 0.8):
					std_ys_new.append(std_ys[j])


	magnitudes_new = []
	phases_new = []
	times_new = []

	for j in range(len(star.phases)):
		if xs_left_max > 0.2:
			if (star.phases[j] > 0.2 and star.phases[j] < xs_left_max):
					magnitudes_new.append(star.magnitudes[j])
					phases_new.append(star.phases[j])
					times_new.append(star.times[j])
		if xs_right_max < 0.8:
			if (star.phases[j] > xs_right_max and star.phases[j] < 0.8):
					magnitudes_new.append(star.magnitudes[j])
					phases_new.append(star.phases[j])
					times_new.append(star.times[j])
	
	linear_model = np.polyfit(times_new, magnitudes_new, 1)

	return star.name, np.mean(magnitudes_new), np.average(std_ys_new), magnitudes_new, phases_new, times_new,linear_model, average_std

def area_func(fx,a,b):
    return a*np.exp(b*fx)

def std_func(x, a, b):
	return a*x*x*x*x

def ratio_func(x, a, b):
	return a*x*x*x + b

def std_for_plot(stars):
	mean_magnitudes = []
	standard_deviations = []
	names = []

	for star in stars:
		mean_magnitudes.append(star.mean_magnitude)
		standard_deviations.append(star.standard_deviation - np.mean(star.magnitude_errors))
		names.append(star.name)

	norm_x = min(mean_magnitudes)
	norm_y = max(standard_deviations) 
	fx2 = mean_magnitudes - norm_x + 1
	fy2 = standard_deviations / norm_y
		
	popt,pcov=op.curve_fit(std_func,fx2,fy2,p0=(1,1))
	fit_y_ = 0.0028 #norm_y*std_func(fx2, *popt) 

	fit_y = []
	for i in range(len(mean_magnitudes)):
		fit_y.append(fit_y_)

	names_multiple_stars = []
	magnitude_multiple_stars = []
	annotation = []

	for i in range(len(mean_magnitudes)):
		if fit_y[i] < standard_deviations[i]:
			names_multiple_stars.append(names[i])
			magnitude_multiple_stars.append(mean_magnitudes[i])
			annotation.append("True")

	return mean_magnitudes, standard_deviations, fit_y, names_multiple_stars, magnitude_multiple_stars, annotation

def ratio_mean_mag_for_plot(stars):
	mean_magnitudes = []
	names = []
	mean_mag_all = []

	for star in stars:
		mean_magnitudes.append(star.mean_magnitude)
		names.append(star.name)
		mean_mag_all.append(np.average(star.magnitudes))

	mean_mag = np.asarray(mean_magnitudes)
	mean_magnitude_all = np.asarray(mean_mag_all)
	dif_mag = mean_magnitude_all / mean_mag

	norm_x = min(mean_magnitudes)
	norm_y = max(dif_mag) 
	fx2 = mean_magnitudes - norm_x + 1
	fy2 = dif_mag / norm_y
		
	popt,pcov=op.curve_fit(ratio_func,fx2,fy2,p0=(1,1))
	fit_y_ = 1.006 #norm_y*ratio_func(fx2, *popt) + 0.5  

	fit_y = []
	for i in range(len(mean_magnitudes)):
		fit_y.append(fit_y_)

	names_multiple_stars = []
	magnitude_multiple_stars = []
	annotation = []

	for i in range(len(mean_magnitudes)):
		if fit_y[i] > dif_mag[i]:
			names_multiple_stars.append(names[i])
			magnitude_multiple_stars.append(mean_magnitudes[i])
			annotation.append("True")

	return mean_magnitudes, dif_mag, fit_y, names_multiple_stars, magnitude_multiple_stars, annotation

def ratio_std_for_plot(stars):
	mean_magnitudes = []
	standard_deviations = []
	names = []

	for star in stars:
		mean_magnitudes.append(star.mean_magnitude)
		standard_deviations.append(abs(star.average_std/star.standard_deviation))
		names.append(star.name)

	norm_x = min(mean_magnitudes)
	norm_y = max(standard_deviations) 
	fx2 = mean_magnitudes - norm_x + 1
	fy2 = standard_deviations / norm_y
		
	popt,pcov=op.curve_fit(ratio_func,fx2,fy2,p0=(1,1))
	fit_y_ = 2 # norm_y*ratio_func(fx2, *popt) + 0.5  

	fit_y = []
	for i in range(len(mean_magnitudes)):
		fit_y.append(fit_y_)

	names_multiple_stars = []
	magnitude_multiple_stars = []
	annotation = []

	for i in range(len(mean_magnitudes)):
		if fit_y[i] > standard_deviations[i]:
			names_multiple_stars.append(names[i])
			magnitude_multiple_stars.append(mean_magnitudes[i])
			annotation.append("True")

	return mean_magnitudes, standard_deviations, fit_y, names_multiple_stars, magnitude_multiple_stars, annotation
	
def under_std_plot(stars):
	mean_magnitudes = []
	area_down = []
	names = []

	for star in stars:
		mean_magnitudes.append(star.mean_magnitude)
		area_down.append(star.area_down)
		names.append(star.name)

	norm_x = min(mean_magnitudes)
	norm_y = max(area_down) 
	fx2 = mean_magnitudes - norm_x + 1
	fy2 = area_down / norm_y
		
	popt,pcov=op.curve_fit(area_func,fx2,fy2,p0=(1,1))
	fit_y_ = 0.025 # norm_y*area_func(fx2, *popt) - 0.18    0.6

	fit_y = []
	for i in range(len(mean_magnitudes)):
		fit_y.append(fit_y_)

	names_multiple_stars = []
	magnitude_multiple_stars = []
	annotation = []

	for i in range(len(mean_magnitudes)):
		if fit_y[i] < area_down[i] and area_down[i] < 0.5:
			names_multiple_stars.append(names[i])
			magnitude_multiple_stars.append(mean_magnitudes[i])
			annotation.append("True")

	return mean_magnitudes, area_down, fit_y, names_multiple_stars, magnitude_multiple_stars, annotation

def area_plot(stars):
	mean_magnitudes = []
	area_diffrence = []
	names = []

	for star in stars:
		mean_magnitudes.append(star.mean_magnitude)
		area_diffrence.append(star.average_area_diffrence)
		names.append(star.name)

	norm_x = min(mean_magnitudes)
	norm_y = max(area_diffrence) 
	fx2 = mean_magnitudes - norm_x + 1
	fy2 = area_diffrence / norm_y
		
	popt,pcov=op.curve_fit(area_func,fx2,fy2,p0=(1,1))
	fit_y_ = 0.5 # norm_y*area_func(fx2, *popt) - 0.18    0.6

	fit_y = []
	for i in range(len(mean_magnitudes)):
		fit_y.append(fit_y_)

	names_multiple_stars = []
	magnitude_multiple_stars = []
	annotation = []

	for i in range(len(mean_magnitudes)):
		if fit_y[i] > area_diffrence[i]:
			names_multiple_stars.append(names[i])
			magnitude_multiple_stars.append(mean_magnitudes[i])
			annotation.append("True")

	return mean_magnitudes, area_diffrence, fit_y, names_multiple_stars, magnitude_multiple_stars, annotation

def under_std(star):
	xs = np.array([np.mean(w) for w in windows(star.phases)])
	area_down = np.empty(shape = xs.shape)
	mag_error = np.empty(shape = xs.shape)
	max_ys = np.empty(shape = xs.shape)
	std_ys = np.empty(shape = xs.shape)

	quadratic_model = np.polyfit(star.phases, star.magnitudes, 8)
	quadratic_model_y = np.polyval(quadratic_model, xs)

	for (i, w) in enumerate(windows(star.magnitude_errors)):
		mag_error[i] = np.average(w)

	for (i, w) in enumerate(windows(star.magnitudes)):
		std_ys[i] = np.std(w) - mag_error[i]
		max_ys[i] = np.max(w)
		area_down[i] = max_ys[i]


	max_model = 0
	max_model_order = 0

	for j in range(len(xs)):
		if (xs[j] > 0.2 and xs[j] < 0.8):
			if max_model < quadratic_model_y[j]: 
				max_model = quadratic_model_y[j]
				max_model_order = j
		
	xs_left_max = xs[max_model_order] - 0.2
	xs_right_max = xs[max_model_order] + 0.2

	area_down_new = []
	std_ys_new =[]

	for j in range(len(xs)):
		if xs_left_max > 0.2:
			if (xs[j] > 0.2 and xs[j] < xs_left_max):
				std_ys_new.append(std_ys[j])
				area_down_new.append(area_down[j])
		if xs_right_max < 0.8:
			if (xs[j] > xs_right_max and xs[j] < 0.8):
				std_ys_new.append(std_ys[j])
				area_down_new.append(area_down[j])

	magnitudes_new = []

	for j in range(len(star.phases)):
		if xs_left_max > 0.2:
			if (star.phases[j] > 0.2 and star.phases[j] < xs_left_max):
					magnitudes_new.append(star.magnitudes[j])
		if xs_right_max < 0.8:
			if (star.phases[j] > xs_right_max and star.phases[j] < 0.8):
					magnitudes_new.append(star.magnitudes[j])

	average_std = np.average(std_ys_new)
	average_mean = np.mean(magnitudes_new)
	average_down = np.average(area_down_new) - average_mean + average_std - 3*np.mean(star.magnitude_errors)

	return average_down, np.mean(magnitudes_new)

def area(star):
	xs = np.array([np.mean(w) for w in windows(star.phases)])
	mean_ys = np.empty(shape = xs.shape)
	std_ys = np.empty(shape = xs.shape)
	max_ys = np.empty(shape = xs.shape)
	min_ys = np.empty(shape = xs.shape)
	area_up = np.empty(shape = xs.shape)
	area_down = np.empty(shape = xs.shape)
	area_diffrence = np.empty(shape = xs.shape)
	mag_error = np.empty(shape = xs.shape)

	quadratic_model = np.polyfit(star.phases, star.magnitudes, 8)
	quadratic_model_y = np.polyval(quadratic_model, xs)

	for (i, w) in enumerate(windows(star.magnitude_errors)):
		mag_error[i] = np.average(w)

	for (i, w) in enumerate(windows(star.magnitudes)):
		mean_ys[i] = np.mean(w)
		std_ys[i] = np.std(w) - mag_error[i]
		max_ys[i] = np.max(w)
		min_ys[i] = np.min(w)
		area_up[i] = mean_ys[i] - min_ys[i] - std_ys[i]
		area_down[i] = max_ys[i] -mean_ys[i] + std_ys[i]


	max_model = 0
	max_model_order = 0

	for j in range(len(xs)):
		if (xs[j] > 0.2 and xs[j] < 0.8):
			if max_model < quadratic_model_y[j]: 
				max_model = quadratic_model_y[j]
				max_model_order = j
		
	xs_left_max = xs[max_model_order] - 0.2
	xs_right_max = xs[max_model_order] + 0.2

	std_ys_new =[]
	max_ys_new = []
	min_ys_new = []
	mean_ys_new = []
	area_up_new = []
	area_down_new = []

	for j in range(len(xs)):
		if xs_left_max > 0.2:
			if (xs[j] > 0.2 and xs[j] < xs_left_max):
					std_ys_new.append(std_ys[j])
					max_ys_new.append(max_ys[j])
					min_ys_new.append(min_ys[j])
					mean_ys_new.append(mean_ys[j])
					area_up_new.append(area_up[j])
					area_down_new.append(area_down[j])
		if xs_right_max < 0.8:
			if (xs[j] > xs_right_max and xs[j] < 0.8):
					std_ys_new.append(std_ys[j])
					max_ys_new.append(max_ys[j])
					min_ys_new.append(min_ys[j])
					mean_ys_new.append(mean_ys[j])
					area_up_new.append(area_up[j])
					area_down_new.append(area_down[j])

	magnitudes_new = []
	mag_error_new = []

	for j in range(len(star.phases)):
		if xs_left_max > 0.2:
			if (star.phases[j] > 0.2 and star.phases[j] < xs_left_max):
					magnitudes_new.append(star.magnitudes[j])
					mag_error_new.append(star.magnitude_errors[j])
		if xs_right_max < 0.8:
			if (star.phases[j] > xs_right_max and star.phases[j] < 0.8):
					magnitudes_new.append(star.magnitudes[j])
					mag_error_new.append(star.magnitude_errors[j])

	average_up = np.average(area_up_new)
	average_down = np.average(area_down_new)

	average_diffrence_area = np.abs(average_up / average_down)

	return average_diffrence_area, np.mean(magnitudes_new)