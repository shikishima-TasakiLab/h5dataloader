from typing import Dict, Tuple
import numpy as np
import cv2
from numpy.lib.function_base import flip
from .structure import *

class Augmentation():

    @staticmethod
    def affine(rand_dict: Dict[str, float], range_angle: Tuple[float, float], range_scale: Tuple[float, float], interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0):
        if isinstance(range_angle, tuple) is False: raise TypeError('`type(range_angle)` must be tuple.')
        if len(range_angle) != 2: raise ValueError('`len(range_angle)` must be 2.')
        if range_angle[0] > range_angle[1]: raise ValueError('`range_angle[0] < range_angle[1]')

        if isinstance(range_scale, tuple) is False: raise TypeError('`type(range_scale)` must be tuple.')
        if len(range_scale) != 2: raise ValueError('`len(range_scale)` must be 2.')
        if range_scale[0] > range_scale[1]: raise ValueError('`range_scale[0] < range_scale[1]')

        rand_dict[AUG_AFFINE_ANGLE] = None
        rand_dict[AUG_AFFINE_SCALE] = None

        def _affine(src: np.ndarray, rand_dict: Dict[str, float]) -> np.ndarray:
            height, width = src.shape[:2]

            angle = rand_dict[AUG_AFFINE_ANGLE] * (range_angle[1] - range_angle[0]) + range_angle[0]
            scale = (1.0 - rand_dict[AUG_AFFINE_SCALE]) * (range_scale[1] - range_scale[0]) + range_scale[0]

            rotmat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            return cv2.warpAffine(src, rotmat, dsize=(width, height), flags=interpolation, borderMode=borderMode, borderValue=borderValue)

        return _affine

    @staticmethod
    def translation(rand_dict: Dict[str, float], range_vshift: Tuple[float, float], range_hshift: Tuple[float, float], interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0):
        if isinstance(range_vshift, tuple) is False: raise TypeError('`type(range_vshift)` must be tuple.')
        if len(range_vshift) != 2: raise ValueError('`len(range_vshift)` must be 2.')
        if range_vshift[0] > range_vshift[1]: raise ValueError('`range_vshift[0] < range_vshift[1]')

        if isinstance(range_hshift, tuple) is False: raise TypeError('`type(range_hshift)` must be tuple.')
        if len(range_hshift) != 2: raise ValueError('`len(range_hshift)` must be 2.')
        if range_hshift[0] > range_hshift[1]: raise ValueError('`range_hshift[0] < range_hshift[1]')

        rand_dict[AUG_TRANSLATION_HSHIFT] = None
        rand_dict[AUG_TRANSLATION_VSHIFT] = None

        def _translation(src: np.ndarray, rand_dict: Dict[str, float]) -> np.ndarray:
            height, width = src.shape[:2]

            hshift = rand_dict[AUG_TRANSLATION_HSHIFT] * (range_hshift[1] - range_hshift[0]) + range_hshift[0]
            vshift = rand_dict[AUG_TRANSLATION_VSHIFT] * (range_vshift[1] - range_vshift[0]) + range_vshift[0]

            rotmat = np.array([[1.0, 0.0, hshift], [0.0, 1.0, vshift]], dtype=np.float)
            return cv2.warpAffine(src, rotmat, dsize=(width, height), flags=interpolation, borderMode=borderMode, borderValue=borderValue)

        return _translation

    @staticmethod
    def flip(rand_dict: Dict[str, float], hflip_enable: bool = True, vflip_enable: bool = True):
        rand_dict[AUG_FLIP_H] = None
        rand_dict[AUG_FLIP_V] = None

        def _flip(src: np.ndarray, rand_dict: Dict[str, float]) -> np.ndarray:
            dst = np.copy(src)

            if hflip_enable and rand_dict[AUG_FLIP_H] >= 0.5:
                dst = cv2.flip(dst, 1)
            if vflip_enable and rand_dict[AUG_FLIP_V] >= 0.5:
                dst = cv2.flip(dst, 0)

            return dst

        return _flip

    @staticmethod
    def cutout(rand_dict: Dict[str, float], rate: float, n_holes: int, length: int):
        rand_dict[AUG_CUTOUT_X] = [None for _ in range(n_holes)]
        rand_dict[AUG_CUTOUT_Y] = [None for _ in range(n_holes)]
        rand_dict[AUG_CUTOUT_DO] = None

        length_half = length // 2

        def _cutout(src: np.ndarray, rand_dict: Dict[str, Union[float, List[float]]]):
            if rand_dict[AUG_CUTOUT_DO] > rate:
                return src

            height, width = src.shape[:2]
            dst = np.copy(src)

            for rand_x, rand_y in zip(rand_dict[AUG_CUTOUT_X], rand_dict[AUG_CUTOUT_Y]):
                x = int(rand_x * width)
                y = int(rand_y * height)

                y1 = np.clip(y - length_half, 0, height)
                y2 = np.clip(y + length_half, 0, height)
                x1 = np.clip(x - length_half, 0, width)
                x2 = np.clip(x + length_half, 0, width)

                dst[y1:y2, x1:x2] = 0

            return dst

        return _cutout

    @staticmethod
    def random_erasing(rand_dict: Dict[str, float], rate: float, area_ratio_range: Tuple[float, float], aspect_ratio_range: Tuple[float, float]):
        pass
