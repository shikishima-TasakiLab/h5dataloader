from typing import Dict, Tuple, NamedTuple
import numpy as np
import cv2
from .structure import *

class BoundingBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

class ValueRange(NamedTuple):
    min: Union[int, float]
    max: Union[int, float]

class Augmentation():

    @staticmethod
    def affine(
        rand_dict: Dict[str, float], range_angle: ValueRange, range_scale: ValueRange,
        interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0
    ):
        if isinstance(range_angle, tuple) is False: raise TypeError('`type(range_angle)` must be tuple.')
        if len(range_angle) != 2: raise ValueError('`len(range_angle)` must be 2.')
        if range_angle.min > range_angle.max: raise ValueError('`range_angle.min < range_angle.max')

        if isinstance(range_scale, tuple) is False: raise TypeError('`type(range_scale)` must be tuple.')
        if len(range_scale) != 2: raise ValueError('`len(range_scale)` must be 2.')
        if range_scale.min > range_scale.max: raise ValueError('`range_scale.min < range_scale.max')

        rand_dict[AUG_AFFINE_ANGLE] = None
        rand_dict[AUG_AFFINE_SCALE] = None

        def _affine(src: np.ndarray, rand_dict: Dict[str, float]) -> np.ndarray:
            height, width = src.shape[:2]

            angle = rand_dict[AUG_AFFINE_ANGLE] * (range_angle.max - range_angle.min) + range_angle.min
            scale = (1.0 - rand_dict[AUG_AFFINE_SCALE]) * (range_scale.max - range_scale.min) + range_scale.min

            rotmat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            return cv2.warpAffine(src, rotmat, dsize=(width, height), flags=interpolation, borderMode=borderMode, borderValue=borderValue)

        return _affine

    @staticmethod
    def translation(rand_dict: Dict[str, float], range_vshift: ValueRange, range_hshift: ValueRange, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0):
        if isinstance(range_vshift, tuple) is False: raise TypeError('`type(range_vshift)` must be tuple.')
        if len(range_vshift) != 2: raise ValueError('`len(range_vshift)` must be 2.')
        if range_vshift.min > range_vshift.max: raise ValueError('`range_vshift.min < range_vshift.max')

        if isinstance(range_hshift, tuple) is False: raise TypeError('`type(range_hshift)` must be tuple.')
        if len(range_hshift) != 2: raise ValueError('`len(range_hshift)` must be 2.')
        if range_hshift.min > range_hshift.max: raise ValueError('`range_hshift.min < range_hshift.max')

        rand_dict[AUG_TRANSLATION_HSHIFT] = None
        rand_dict[AUG_TRANSLATION_VSHIFT] = None

        def _translation(src: np.ndarray, rand_dict: Dict[str, float]) -> np.ndarray:
            height, width = src.shape[:2]

            hshift = rand_dict[AUG_TRANSLATION_HSHIFT] * (range_hshift.max - range_hshift.min) + range_hshift.min
            vshift = rand_dict[AUG_TRANSLATION_VSHIFT] * (range_vshift.max - range_vshift.min) + range_vshift.min

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
    def get_cutout_bbox(rate: float, n_holes: int, length: int):
        length_half = length // 2

        def _get_cutout_bbox(shape: Tuple[int, ...]) -> List[BoundingBox]:
            bboxes = []

            if np.random.rand() <= rate:
                height, width = shape[:2]

                for _ in range(n_holes):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)

                    bboxes.append(BoundingBox(
                        x1 = np.clip(x - length_half, 0, width),
                        y1 = np.clip(y - length_half, 0, height),
                        x2 = np.clip(x + length_half, 0, width),
                        y2 = np.clip(y + length_half, 0, height)
                    ))
            return bboxes

        return _get_cutout_bbox

    @staticmethod
    def random_erasing_bbox(rate: float, area_ratio_range: ValueRange, aspect_ratio_range: ValueRange):
        if rate < 0.0 or 1.0 < rate: raise ValueError('`0.0 <= rate <= 1.0`')

        def _random_erasing_bbox(shape: Tuple[int, ...]) -> List[BoundingBox]:
            bboxes = []

            if np.random.rand() <= rate:
                height, width = shape[:2]
                s = height * width

                se = (np.random.rand() * (area_ratio_range.max - area_ratio_range.min) + area_ratio_range.min) * s
                re = np.random.rand() * (aspect_ratio_range.max - aspect_ratio_range.min) + aspect_ratio_range.min

                he = int(np.sqrt(se * re))
                we = int(np.sqrt(se / re))

                xe = np.random.randint(low=0, high=width-we)
                ye = np.random.randint(low=0, high=height-he)

                bboxes.append(BoundingBox(x1=xe, y1=ye, x2=xe+we, y2=ye+he))
            return bboxes

        return _random_erasing_bbox

    @staticmethod
    def bbox_masking(masked_value: Union[int, float, ValueRange]) -> function:
        use_rand = isinstance(masked_value, ValueRange)
        if use_rand is True:
            use_int = isinstance(masked_value.min, int)

        def _bbox_masking(src: np.ndarray, bboxes: List[BoundingBox]) -> np.ndarray:
            dst = np.copy(src)

            for bbox in bboxes:
                if use_rand is True:
                    bb_shape = dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2].shape
                    if use_int is True:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.randint(masked_value.min, masked_value.max, bb_shape)
                    else:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.rand(*bb_shape) * (masked_value.max - masked_value.min) + masked_value.min
                else:
                    dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = masked_value
            return dst

        return _bbox_masking
