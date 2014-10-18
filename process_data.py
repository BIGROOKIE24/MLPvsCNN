import math
import pickle, gzip
import theano, numpy
from skimage.transform import PiecewiseAffineTransform, warp


def distort_data(shared_x, shared_y, dist_deg, pickle=False):
	x_orig = shared_x.get_value()
	#y_orig = shared_y.get_value()
	y_orig = shared_y
	x_distorted = []
	dim = x_orig.shape[1]
	image_dim = int(math.sqrt(dim))
	for x in x_orig:
		image = x.reshape((image_dim, image_dim))	
		rows, cols = image.shape[0], image.shape[1]

		src_cols = numpy.linspace(0, cols, 20)
		src_rows = numpy.linspace(0, rows, 10)
		src_rows, src_cols = numpy.meshgrid(src_rows, src_cols)
		src = numpy.dstack([src_cols.flat, src_rows.flat])[0]

		# add sinusoidal oscillation to row coordinates
		dst_rows = src[:, 1] - numpy.sin(numpy.linspace(0, 3 * numpy.pi, src.shape[0])) * dist_deg
		dst_cols = src[:, 0]
		dst_rows *= 1.5
		dst_rows -= 1.5 * dist_deg
		dst = numpy.vstack([dst_cols, dst_rows]).T


		tform = PiecewiseAffineTransform()
		tform.estimate(src, dst)

		out_rows = image.shape[0]
		out_cols = cols
		out = warp(image, tform, output_shape=(out_rows, out_cols))
		x_distorted.append(out.reshape(dim))
	new_x_np = numpy.concatenate([x_orig, x_distorted], axis=0)
	new_y_np = numpy.concatenate([y_orig, y_orig], axis=0)
	new_x = theano.shared(value = new_x_np, borrow=True)
	new_y = theano.shared(value = new_y_np, borrow=True)
	if pickle:
 		out_file = gzip.open("train_data_with_distortion.pkl.gz", 'wb')
        pickle.dump((new_x_np, new_y_np), out_file)
	return new_x, new_y
