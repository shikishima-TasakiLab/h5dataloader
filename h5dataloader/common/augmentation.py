from typing import Dict, Tuple, NamedTuple
import numpy as np
import cv2
from .structure import *
from .convert import *

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
    def affine_2d(
        range_angle: ValueRange, range_scale: ValueRange,
        interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0
    ):
        if isinstance(range_angle, tuple) is False: raise TypeError('`type(range_angle)` must be tuple.')
        if len(range_angle) != 2: raise ValueError('`len(range_angle)` must be 2.')
        if range_angle.min > range_angle.max: raise ValueError('`range_angle.min < range_angle.max')

        if isinstance(range_scale, tuple) is False: raise TypeError('`type(range_scale)` must be tuple.')
        if len(range_scale) != 2: raise ValueError('`len(range_scale)` must be 2.')
        if range_scale.min > range_scale.max: raise ValueError('`range_scale.min < range_scale.max')

        tmp_itr: int = None
        rotmat: np.ndarray = None

        def _affine_2d(step_itr: int, src: Data) -> np.ndarray:
            nonlocal tmp_itr, rotmat
            height, width = src.data.shape[:2]

            if tmp_itr != step_itr:
                tmp_itr = step_itr

                angle = np.random.rand() * (range_angle.max - range_angle.min) + range_angle.min
                scale = (1.0 - np.random.rand()) * (range_scale.max - range_scale.min) + range_scale.min

                rotmat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            return Data(cv2.warpAffine(src.data, rotmat, dsize=(width, height), flags=interpolation, borderMode=borderMode, borderValue=borderValue), src.type)

        return _affine_2d

    @staticmethod
    def translation_2d(range_vshift: ValueRange, range_hshift: ValueRange, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0):
        if isinstance(range_vshift, tuple) is False: raise TypeError('`type(range_vshift)` must be tuple.')
        if len(range_vshift) != 2: raise ValueError('`len(range_vshift)` must be 2.')
        if range_vshift.min > range_vshift.max: raise ValueError('`range_vshift.min < range_vshift.max')

        if isinstance(range_hshift, tuple) is False: raise TypeError('`type(range_hshift)` must be tuple.')
        if len(range_hshift) != 2: raise ValueError('`len(range_hshift)` must be 2.')
        if range_hshift.min > range_hshift.max: raise ValueError('`range_hshift.min < range_hshift.max')

        tmp_itr: int = None
        rotmat: np.ndarray = None

        def _translation_2d(step_itr: int, src: Data) -> np.ndarray:
            nonlocal tmp_itr, rotmat

            height, width = src.data.shape[:2]

            if tmp_itr != step_itr:
                tmp_itr = step_itr

                hshift = np.random.rand() * (range_hshift.max - range_hshift.min) + range_hshift.min
                vshift = np.random.rand() * (range_vshift.max - range_vshift.min) + range_vshift.min

                rotmat = np.array([[1.0, 0.0, hshift], [0.0, 1.0, vshift]], dtype=np.float)
            return Data(cv2.warpAffine(src.data, rotmat, dsize=(width, height), flags=interpolation, borderMode=borderMode, borderValue=borderValue), src.type)

        return _translation_2d

    @staticmethod
    def flip_2d(hflip_rate: float = 0.5, vflip_rate: float = 0.5):
        hflip_rate = np.clip(hflip_rate, a_min=0.0, a_max=1.0)
        vflip_rate = np.clip(vflip_rate, a_min=0.0, a_max=1.0)

        tmp_itr: int = None
        tmp_hflip: bool = None
        tmp_vflip: bool = None

        def _flip_2d(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_hflip, tmp_vflip
            dst = np.copy(src.data)

            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_hflip = np.random.rand() < hflip_rate
                tmp_vflip = np.random.rand() < vflip_rate
            if tmp_hflip is True:
                dst = cv2.flip(dst, 1)
            if tmp_vflip is True:
                dst = cv2.flip(dst, 0)
            return Data(dst, src.type)

        return _flip_2d

    @staticmethod
    def cutout_2d(rate: float, n_holes: int, length: int, masked_value: Union[int, float, ValueRange]):
        rate = np.clip(rate, a_min=0.0, a_max=1.0)
        length_half = length // 2
        mask = Augmentation.bbox_masking(masked_value)

        tmp_itr: int = None
        tmp_bboxes: List[BoundingBox] = []

        def _cutout_2d(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_bboxes

            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_bboxes = []

                if np.random.rand() < rate:
                    height, width = src.data.shape[:2]
                    for _ in range(n_holes):
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)

                        tmp_bboxes.append(BoundingBox(
                            x1= np.clip(x - length_half, 0, width),
                            y1= np.clip(y - length_half, 0, height),
                            x2= np.clip(x + length_half, 0, width),
                            y2= np.clip(y + length_half, 0, height)
                        ))
            return mask(src, tmp_bboxes)

        return _cutout_2d

    @staticmethod
    def random_erasing_2d(rate: float, area_ratio_range: ValueRange, aspect_ratio_range: ValueRange, masked_value: Union[int, float, ValueRange]):
        rate = np.clip(rate, a_min=0.0, a_max=1.0)
        mask = Augmentation.bbox_masking(masked_value)

        tmp_itr: int = None
        tmp_bboxes: List[BoundingBox] = []

        def _random_erasing_2d(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_bboxes

            if tmp_itr != step_itr:
                tmp_itr = step_itr

                if np.random.rand() < rate:
                    height, width = src.data.shape[:2]
                    s = height * width

                    se = (np.random.rand() * (area_ratio_range.max - area_ratio_range.min) + area_ratio_range.min) * s
                    re = np.random.rand() * (aspect_ratio_range.max - aspect_ratio_range.min) + aspect_ratio_range.min

                    he = int(np.sqrt(se * re))
                    we = int(np.sqrt(se / re))

                    xe = np.random.randint(low=0, high=width-we)
                    ye = np.random.randint(low=0, high=height-he)

                    tmp_bboxes = [BoundingBox(x1=xe, y1=ye, x2=xe+we, y2=ye+he)]
                else:
                    tmp_bboxes = []
            return mask(src, tmp_bboxes)

        return _random_erasing_2d

    @staticmethod
    def bbox_masking(masked_value: Union[int, float, ValueRange]):
        use_rand = isinstance(masked_value, ValueRange)
        if use_rand is True:
            use_int = isinstance(masked_value.min, int)

        def _bbox_masking(src: Data, bboxes: List[BoundingBox]) -> Data:
            dst = np.copy(src.data)

            for bbox in bboxes:
                if use_rand is True:
                    bb_shape = dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2].shape
                    if use_int is True:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.randint(masked_value.min, masked_value.max, bb_shape)
                    else:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.rand(*bb_shape) * (masked_value.max - masked_value.min) + masked_value.min
                else:
                    dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = masked_value
            return Data(data=dst, type=src.type)

        return _bbox_masking

    @staticmethod
    def adjust_brightness(factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)):
        tmp_itr: int = None
        tmp_factor: float = None

        def _adjust_brightness(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_factor
            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_factor = np.random.rand() * (factor_range.max - factor_range.min) + factor_range.min

            return Data(np.clip(src.data * tmp_factor, dst_range.min, dst_range.max).astype(DTYPE_NUMPY[src.type]), src.type)

        return _adjust_brightness

    @staticmethod
    def adjust_contrast(factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)):
        tmp_itr: int = None
        tmp_factor: float = None

        def _adjust_contrast(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_factor
            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_factor = np.random.rand() * (factor_range.max - factor_range.min) + factor_range.min

            gray_mean = to_mono8(src).data.mean()

            return Data(
                data=np.clip(gray_mean * (1.0 - tmp_factor) + src.data * tmp_factor, dst_range.min, dst_range.max).astype(DTYPE_NUMPY[src.type]),
                type=src.type
            )

        return _adjust_contrast

    @staticmethod
    def adjust_saturation(factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)):
        tmp_itr: int = None
        tmp_factor: float = None

        def _adjust_saturation(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_factor
            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_factor = np.random.rand() * (factor_range.max - factor_range.min) + factor_range.min

            gray = to(to_mono8(src), src.type)

            return Data(
                data=np.clip(gray * (1.0 - tmp_factor) + src.data * tmp_factor, dst_range.min, dst_range.max).astype(DTYPE_NUMPY[src.type]),
                type=src.type
            )

        return _adjust_saturation

    @staticmethod
    def adjust_hue(factor_range: ValueRange = ValueRange(0.5, 1.5)):
        tmp_itr: int = None
        tmp_factor: float = None

        def _adjust_hue(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_factor
            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_factor = np.random.rand() * (factor_range.max - factor_range.min) + factor_range.min

            hsv = to_hsv8(src)
            hsv.data[:,:,0] += np.uint8(tmp_factor * 255)

            return to(hsv, src.type)

        return _adjust_hue

    @staticmethod
    def adjust_gamma(gamma_range: ValueRange = ValueRange(0.5, 1.5), gain_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)):
        if gain_range.min < 0.0 or gain_range.max < 0.0:
            raise ValueError('`gain_range.min >= 0.0`, `gain_range.max >= 0.0`')
        tmp_itr: int = None
        tmp_gamma: float = None
        tmp_gain: float = None

        def _adjust_gamma(step_itr: int, src: Data) -> Data:
            nonlocal tmp_itr, tmp_gamma, tmp_gain
            if tmp_itr != step_itr:
                tmp_itr = step_itr
                tmp_gamma = np.random.rand() * (gamma_range.max - gamma_range.min) + gamma_range.min
                tmp_gain = np.random.rand() * (gain_range.max - gain_range.min) + gain_range.min

            img = np.float32(src.data)
            img = dst_range.max * tmp_gain * np.power(img / dst_range.max, tmp_gamma)
            img = DTYPE_NUMPY[src.type](np.clip(img, dst_range.min, dst_range.max))

            return Data(img, src.type)

        return _adjust_gamma
