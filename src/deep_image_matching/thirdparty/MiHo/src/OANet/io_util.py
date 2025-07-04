import numpy as np

def write_keypoints(path, keypoints):
	# path: path to save
	# keypoints: single-precision real matrix, N*4, x,y,scale, orientation
	assert keypoints.shape[1]==4
	write_matrix(path, keypoints, np.float32)

def write_descriptors(path, descriptors):
	write_matrix(path, descriptors, np.float32)

def write_matches(path, matches):
    if len(matches) > 0:
	    write_matrix(path, matches, np.uint32)

def read_keypoints(path):
	return read_matrix(path, np.float32)

def read_descriptors(path):
	return read_matrix(path, np.float32)

def read_matches(path):
	return read_matrix(path, np.uint32)

def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)

def write_matrix(path, data, dtype):
	with open(path, 'wb') as f:
		np.asarray(data.shape, dtype='int32').tofile(f)
		data.astype(dtype).tofile(f)
