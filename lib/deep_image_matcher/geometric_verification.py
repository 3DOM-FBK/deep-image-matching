import importlib
import logging

import cv2
import numpy as np

from .consts import GeometricVerification

logger = logging.getLogger(__name__)


def geometric_verification(
    mkpts0: np.ndarray = None,
    mkpts1: np.ndarray = None,
    method: GeometricVerification = GeometricVerification.PYDEGENSAC,
    threshold: float = 1,
    confidence: float = 0.9999,
    max_iters: int = 10000,
    laf_consistensy_coef: float = -1.0,
    error_type: str = "sampson",
    symmetric_error_check: bool = True,
    enable_degeneracy_check: bool = True,
) -> dict:
    """
    Computes the fundamental matrix and inliers between the two images using geometric verification.

    Args:
        method (str): The method used for geometric verification. Can be one of ['pydegensac', 'opencv'].
        threshold (float): Pixel error threshold for considering a correspondence an inlier.
        confidence (float): The required confidence level in the results.
        max_iters (int): The maximum number of iterations for estimating the fundamental matrix.
        laf_consistensy_coef (float): The weight given to Local Affine Frame (LAF) consistency term for pydegensac.
        error_type (str): The error function used for computing the residuals in the RANSAC loop.
        symmetric_error_check (bool): If True, performs an additional check on the residuals in the opposite direction.
        enable_degeneracy_check (bool): If True, enables the check for degeneracy using SVD.

    Returns:
        np.ndarray: A Boolean array that masks the correspondences that were identified as inliers.

    TODO: allow input parameters for both pydegensac and MAGSAC.

    """

    assert isinstance(
        method, GeometricVerification
    ), "Invalid method. It must be a GeometricVerification enum in GeometricVerification.PYDEGENSAC or GeometricVerification.MAGSAC."

    F = None
    inlMask = np.ones(len(mkpts0), dtype=bool)

    if len(mkpts0) < 4:
        logger.warning("Not enough matches to perform geometric verification.")
        return F, inlMask

    if method == GeometricVerification.PYDEGENSAC:
        try:
            pydegensac = importlib.import_module("pydegensac")
            fallback = False
        except:
            logger.error(
                "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
            )
            fallback = True

    if method == GeometricVerification.PYDEGENSAC and not fallback:
        try:
            F, inlMask = pydegensac.findFundamentalMatrix(
                mkpts0,
                mkpts1,
                px_th=threshold,
                conf=confidence,
                max_iters=max_iters,
                laf_consistensy_coef=laf_consistensy_coef,
                error_type=error_type,
                symmetric_error_check=symmetric_error_check,
                enable_degeneracy_check=enable_degeneracy_check,
            )
            logger.info(
                f"Pydegensac found {inlMask.sum()} inliers ({inlMask.sum()*100/len(mkpts0):.2f}%)"
            )
        except Exception as err:
            # Fall back to MAGSAC++ if pydegensac fails
            logger.error(
                f"{err}. Unable to perform geometric verification with Pydegensac. Trying using MAGSAC++ (OpenCV) instead."
            )
            fallback = True

    if method == GeometricVerification.MAGSAC or fallback:
        try:
            F, inliers = cv2.findFundamentalMat(
                mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
            )
            inlMask = (inliers > 0).squeeze()
            logger.info(
                f"MAGSAC++ found {inlMask.sum()} inliers ({inlMask.sum()*100/len(mkpts0):.2f}%)"
            )
        except Exception as err:
            logger.error(
                f"{err}. Unable to perform geometric verification with MAGSAC++."
            )
            inlMask = np.ones(len(mkpts0), dtype=bool)

    return F, inlMask
