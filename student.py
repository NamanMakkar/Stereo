import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # Get image dimensions and initialize arrays
    height, width, channel = images[0].shape
    albedo = np.zeros((height, width, channel), dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    images_np = np.array(images)

    # Calculate G_first_term using lights array (L.T @ L)^-1
    G_first_term = np.linalg.inv(lights.T @ lights)

    # Calculate intensity values for each pixel --> (N x height x width x 1)
    I = images_np.reshape(len(images), height, width, channel)

    # Compute G using G_first_term, lights, and intensity values --> (3 x height x width x channel)
    G = np.tensordot(
        G_first_term, np.tensordot(lights, I, axes=([0], [0])), axes=([1], [0])
    )

    # Calculate albedo as the norm of G along axis 0 --> (height x width x channel)
    k_d = np.linalg.norm(G, axis=0)

    # Set albedo to 0 where k_d < 1e-7
    mask = k_d < 1e-7
    albedo[mask] = 0

    # Trick to avoid division by 0 errors
    k_d[mask] = np.inf

    # Calculate normals as the normalized G (3 x height x width x channel)
    normals = G / k_d[np.newaxis, :, :, :]

    # Reversing the trick, setting normals to 0 where k_d == np.inf
    # technically not needed since val / inf = 0
    normals[:, k_d == np.inf] = 0

    # Transpose normals to get the final output shape --> (height x width x 3)
    # First we select (3 x height x width x channel[0]) with [:, :, :, 0]
    # Then we transpose it to (height x width x 3)
    normals = np.transpose(normals[:, :, :, 0], (1, 2, 0))

    # Save k_d values where k_d is not 0 to the corresponding position in the albedo
    albedo[~mask] = k_d[~mask]

    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """

    projection_matrix = K @ Rt
    height, width, _ = points.shape

    # Add an additional dimension for homogeneous coordinates with val
    homogenous_points = np.concatenate([points, np.ones((height, width, 1))], axis=-1)

    # Reshape the homogenous_points array to (height * width, 4)
    reshaped_points = homogenous_points.reshape(-1, 4)

    # print(f"points.shape: {points.shape}")
    # print(f"homogenous_points.shape: {homogenous_points.shape}")
    # print(f"reshaped_points.shape: {reshaped_points.shape}")
    # print(f"projection_matrix.shape: {projection_matrix.shape}")

    # Calculate the mapped 3D points in homogeneous coordinates by matrix multiplication
    mapped_points = projection_matrix @ reshaped_points.T

    # Get 2D points dividing by z using broadcasting
    xy_points = mapped_points[:-1] / mapped_points[-1]

    # Reshape the inhomogeneous_points array back to (height, width, 2)
    projections = xy_points.T.reshape(height, width, 2)

    return projections

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    H = image.shape[0]
    W = image.shape[1]
    C = image.shape[2]
    offset = ncc_size // 2
    normalized_channel_size = C * ncc_size**2
    normalized = np.zeros((H, W, ncc_size, ncc_size, C))
    for h in range(offset, H - offset):
        for w in range(offset, W - offset):
            normalized[h,w] = image[h-offset:h+offset+1, w-offset:w+offset+1, :]
    normalized = normalized.reshape((H,W,ncc_size**2,C)).transpose((0,1,3,2))
    normalized -= normalized.mean(axis=3,keepdims=True)
    norm = np.linalg.norm(normalized, axis=(2,3)).reshape((H,W,1,1))
    norm[norm < 1e-6] = 1
    normalized /= norm
    normalized = normalized.reshape((H,W,normalized_channel_size))
    return normalized

def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(image1*image2, axis=2)
    return ncc