import importlib
from typing import Tuple

import cv2
import numpy as np

from deep_image_matching import GeometricVerification, logger

pydegesac_default_params = {
    "laf_consistensy_coef": -1.0,
    "error_type": "sampson",
    "symmetric_error_check": True,
    "enable_degeneracy_check": True,
}
opencv_methods_mapping = {
    "LMEDS": cv2.LMEDS,
    "RANSAC": cv2.RANSAC,
    "RHO": cv2.RHO,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
}


def log_result(inlMask: np.ndarray, method: str) -> None:
    logger.debug(
        f"{method} found {inlMask.sum()} inliers ({inlMask.sum()*100/len(inlMask):.2f}%)"
    )


def log_error(err: Exception, method: str, fallback: bool = False) -> None:
    logger.warning(f"{err} - Unable to perform geometric verification with {method}.")
    if fallback:
        logger.warning("Trying using RANSAC (OpenCV) instead.")


def geometric_verification(
    kpts0: np.ndarray = None,
    kpts1: np.ndarray = None,
    method: GeometricVerification = GeometricVerification.PYDEGENSAC,
    threshold: float = 1,
    confidence: float = 0.9999,
    max_iters: int = 10000,
    quiet: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the fundamental matrix and inliers between the two images using geometric verification.

    Args:
        method (str): The method used for geometric verification. Can be one of ['pydegensac', 'opencv'].
        threshold (float): Pixel error threshold for considering a correspondence an inlier.
        confidence (float): The required confidence level in the results.
        max_iters (int): The maximum number of iterations for estimating the fundamental matrix.
        quiet (bool): If True, disables logging.
        **kwargs: Additional parameters for the selected method. Check the documentation of the selected method for more information. For pydegensac: https://github.com/ducha-aiki/pydegensac, for all the other OPENCV methods: https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f

    Returns:
        [np.ndarray, np.ndarray]: A tuple containing:
            - F: The estimated fundamental matrix.
            - inlMask: a Boolean array that masks the correspondences that were identified as inliers.

    """

    assert isinstance(
        method, GeometricVerification
    ), "Invalid method. It must be a GeometricVerification enum in GeometricVerification.PYDEGENSAC or GeometricVerification.MAGSAC."

    fallback = False
    F = None
    inlMask = np.ones(len(kpts0), dtype=bool)

    if len(kpts0) < 8:
        if not quiet:
            logger.warning("Not enough matches to perform geometric verification.")
        return F, inlMask

    if method == GeometricVerification.PYDEGENSAC:
        try:
            pydegensac = importlib.import_module("pydegensac")
        except ImportError:
            logger.warning(
                "Pydegensac not available. Using RANSAC (OpenCV) for geometric verification."
            )
            fallback = True

    if method == GeometricVerification.PYDEGENSAC and not fallback:
        try:
            params = {**pydegesac_default_params, **kwargs}
            F, inlMask = pydegensac.findFundamentalMatrix(
                kpts0,
                kpts1,
                px_th=threshold,
                conf=confidence,
                max_iters=max_iters,
                laf_consistensy_coef=params["laf_consistensy_coef"],
                error_type=params["error_type"],
                symmetric_error_check=params["symmetric_error_check"],
                enable_degeneracy_check=params["enable_degeneracy_check"],
            )
            if not quiet:
                log_result(inlMask, method.name)
        except Exception as err:
            # Fall back to RANSAC if pydegensac fails
            fallback = True
            log_error(err, method.name, fallback)

    if method == GeometricVerification.MAGSAC:
        try:
            F, inliers = cv2.findFundamentalMat(
                kpts0, kpts1, cv2.USAC_MAGSAC, threshold, confidence, max_iters
            )
            inlMask = (inliers > 0).squeeze()
            if not quiet:
                log_result(inlMask, method.name)
        except Exception as err:
            # Fall back to RANSAC if MAGSAC fails
            fallback = True
            log_error(err, method.name, fallback)

    # Use a generic OPENCV methods
    if method.name not in ["PYDEGENSAC", "MAGSAC", "RANSAC"]:
        logger.debug(f"Method was set to {method}, trying to use it from OPENCV...")
        met = opencv_methods_mapping[method.name]
        try:
            F, inliers = cv2.findFundamentalMat(
                kpts0, kpts1, met, threshold, confidence, max_iters
            )
            inlMask = (inliers > 0).squeeze()
            if not quiet:
                log_result(inlMask, method.name)
        except Exception as err:
            fallback = True
            log_error(err, method.name, fallback)
            inlMask = np.ones(len(kpts0), dtype=bool)

    # Use RANSAC as fallback
    if method == GeometricVerification.RANSAC or fallback:
        try:
            F, inliers = cv2.findFundamentalMat(
                kpts0, kpts1, cv2.RANSAC, threshold, confidence, max_iters
            )
            inlMask = (inliers > 0).squeeze()
            if not quiet:
                log_result(inlMask, method.name)
        except Exception as err:
            log_error(err, method.name)
            inlMask = np.ones(len(kpts0), dtype=bool)

    if not quiet:
        logger.debug(f"Estiamted Fundamental matrix: \n{F}")

    return F, inlMask
