import cv2
import numpy as np


def get_image_transform_with_border(in_res, out_res, mode="rgb", bgr_to_rgb: bool = False):
    """Pad to a square canvas and resize while preserving the image center."""
    iw, ih = in_res
    interp_method = cv2.INTER_AREA
    if mode in {"depth", "pointmap"}:
        # Avoid interpolating invalid depth or 3-D points across empty regions.
        interp_method = cv2.INTER_NEAREST

    size = max(iw, ih)
    top = (size - ih) // 2
    bottom = size - ih - top
    left = (size - iw) // 2
    right = size - iw - left

    def transform(img: np.ndarray):
        if mode == "rgb":
            assert img.shape == (ih, iw, 3)
            resized = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            resized = cv2.resize(resized, out_res, interpolation=interp_method)
            if bgr_to_rgb:
                resized = resized[:, :, ::-1]
            return resized

        if mode == "depth":
            assert img.shape == (ih, iw)
            padded = cv2.copyMakeBorder(
                img.astype(np.float32, copy=False),
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            return cv2.resize(padded, out_res, interpolation=interp_method).astype(np.float16)

        if mode == "pointmap":
            assert img.shape == (ih, iw, 3)
            padded = cv2.copyMakeBorder(
                img.astype(np.float32, copy=False),
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            return cv2.resize(padded, out_res, interpolation=interp_method).astype(np.float16)

        raise ValueError(f"Unsupported transform mode: {mode}")

    return transform
