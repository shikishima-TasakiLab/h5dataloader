from typing import NamedTuple
import numpy as np
import cv2
from .structure import *
from .convert import *

class BoundingBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

class Augmentation():

    class Affine_2d():
        def __init__(
            self, range_angle: ValueRange, range_scale: ValueRange, interpolation = INTER_LINEAR, borderMode = BORDER_CONSTANT, borderValue = 0
        ) -> None:
            if isinstance(range_angle, tuple) is False: raise TypeError('`type(range_angle)` must be tuple.')
            if len(range_angle) != 2: raise ValueError('`len(range_angle)` must be 2.')
            if range_angle.min > range_angle.max: raise ValueError('`range_angle.min < range_angle.max')

            if isinstance(range_scale, tuple) is False: raise TypeError('`type(range_scale)` must be tuple.')
            if len(range_scale) != 2: raise ValueError('`len(range_scale)` must be 2.')
            if range_scale.min > range_scale.max: raise ValueError('`range_scale.min < range_scale.max')

            self.range_angle = range_angle
            self.range_scale = range_scale
            self.interpolation = interpolation
            self.borderMode = borderMode
            self.borderValue = borderValue

            self.tmp_itr: int = None
            self.rotmat: np.ndarray = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            height, width = src.data.shape[:2]

            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr

                angle = np.random.rand() * (self.range_angle.max - self.range_angle.min) + self.range_angle.min
                scale = (1.0 - np.random.rand()) * (self.range_scale.max - self.range_scale.min) + self.range_scale.min

                rotmat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            return Data(
                data=cv2.warpAffine(src.data, rotmat, dsize=(width, height), flags=self.interpolation, borderMode=self.borderMode, borderValue=self.borderValue),
                type=src.type
            )

    class Translation_2d():
        def __init__(self, range_vshift: ValueRange, range_hshift: ValueRange, interpolation = INTER_LINEAR, borderMode = BORDER_CONSTANT, borderValue = 0) -> None:
            if isinstance(range_vshift, tuple) is False: raise TypeError('`type(range_vshift)` must be tuple.')
            if len(range_vshift) != 2: raise ValueError('`len(range_vshift)` must be 2.')
            if range_vshift.min > range_vshift.max: raise ValueError('`range_vshift.min < range_vshift.max')

            if isinstance(range_hshift, tuple) is False: raise TypeError('`type(range_hshift)` must be tuple.')
            if len(range_hshift) != 2: raise ValueError('`len(range_hshift)` must be 2.')
            if range_hshift.min > range_hshift.max: raise ValueError('`range_hshift.min < range_hshift.max')

            self.range_vshift = range_vshift
            self.range_hshift = range_hshift
            self.interpolation = interpolation
            self.borderMode = borderMode
            self.borderValue = borderValue

            self.tmp_itr: int = None
            self.rotmat: np.ndarray = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            height, width = src.data.shape[:2]

            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr

                hshift = np.random.rand() * (self.range_hshift.max - self.range_hshift.min) + self.range_hshift.min
                vshift = np.random.rand() * (self.range_vshift.max - self.range_vshift.min) + self.range_vshift.min

                self.rotmat = np.array([[1.0, 0.0, hshift], [0.0, 1.0, vshift]], dtype=np.float)
            return Data(
                data=cv2.warpAffine(src.data, self.rotmat, dsize=(width, height), flags=self.interpolation, borderMode=self.borderMode, borderValue=self.borderValue),
                type=src.type
            )

    class Flip_2d():
        def __init__(self, hflip_rate: float = 0.5, vflip_rate: float = 0.5) -> None:
            self.hflip_rate = np.clip(hflip_rate, a_min=0.0, a_max=1.0)
            self.vflip_rate = np.clip(vflip_rate, a_min=0.0, a_max=1.0)

            self.tmp_itr: int = None
            self.tmp_hflip: bool = None
            self.tmp_vflip: bool = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            dst = np.copy(src.data)

            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_hflip = np.random.rand() < self.hflip_rate
                self.tmp_vflip = np.random.rand() < self.vflip_rate
            if self.tmp_hflip is True:
                dst = cv2.flip(dst, 1)
            if self.tmp_vflip is True:
                dst = cv2.flip(dst, 0)
            return Data(dst, src.type)

    class Cutout_2d():
        def __init__(self, rate: float, n_holes: int, length: int, masked_value: Union[int, float, ValueRange]) -> None:
            self.rate = np.clip(rate, a_min=0.0, a_max=1.0)
            self.n_holes = n_holes
            self.length_half = length // 2
            self.masking = Augmentation.Bbox_masking(masked_value)

            self.tmp_itr: int = None
            self.tmp_bboxes: List[BoundingBox] = []

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_bboxes = []

                if np.random.rand() < self.rate:
                    height, width = src.data.shape[:2]
                    for _ in range(self.n_holes):
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)

                        self.tmp_bboxes.append(BoundingBox(
                            x1= np.clip(x - self.length_half, 0, width),
                            y1= np.clip(y - self.length_half, 0, height),
                            x2= np.clip(x + self.length_half, 0, width),
                            y2= np.clip(y + self.length_half, 0, height)
                        ))
            return self.masking(src, self.tmp_bboxes)

    class Random_erasing_2d():
        def __init__(self, rate: float, area_ratio_range: ValueRange, aspect_ratio_range: ValueRange, masked_value: Union[int, float, ValueRange]) -> None:
            self.rate = np.clip(rate, a_min=0.0, a_max=1.0)
            self.area_ratio_range = area_ratio_range
            self.aspect_ratio_range = aspect_ratio_range
            self.masking = Augmentation.Bbox_masking(masked_value)

            self.tmp_itr: int = None
            self.tmp_bboxes: List[BoundingBox] = []

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr

                if np.random.rand() < self.rate:
                    height, width = src.data.shape[:2]
                    s = height * width

                    se = (np.random.rand() * (self.area_ratio_range.max - self.area_ratio_range.min) + self.area_ratio_range.min) * s
                    re = np.random.rand() * (self.aspect_ratio_range.max - self.aspect_ratio_range.min) + self.aspect_ratio_range.min

                    he = int(np.sqrt(se * re))
                    we = int(np.sqrt(se / re))

                    xe = np.random.randint(low=0, high=width-we)
                    ye = np.random.randint(low=0, high=height-he)

                    self.tmp_bboxes = [BoundingBox(x1=xe, y1=ye, x2=xe+we, y2=ye+he)]
                else:
                    self.tmp_bboxes = []
            return self.masking(src, self.tmp_bboxes)

    class Bbox_masking():
        def __init__(self, masked_value: Union[int, float, ValueRange]) -> None:
            self.masked_value = masked_value

            self.use_rand = isinstance(self.masked_value, ValueRange)
            if self.use_rand is True:
                self.use_int = isinstance(self.masked_value.min, int)

        def __call__(self, src: Data, bboxes: List[BoundingBox]) -> Data:
            dst: np.ndarray = np.copy(src.data)

            for bbox in bboxes:
                if self.use_rand is True:
                    bb_shape = dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2].shape
                    if self.use_int is True:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.randint(self.masked_value.min, self.masked_value.max, bb_shape)
                    else:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.rand(*bb_shape) * (self.masked_value.max - self.masked_value.min) + self.masked_value.min
                else:
                    dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = self.masked_value
            return Data(data=dst, type=src.type)

    class Adjust_brightness():
        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)) -> None:
            self.factor_range = factor_range
            self.dst_range = dst_range

            self.tmp_itr: int = None
            self.tmp_factor: float = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_factor = np.random.rand() * (self.factor_range.max - self.factor_range.min) + self.factor_range.min

            return Data(np.clip(src.data * self.tmp_factor, self.dst_range.min, self.dst_range.max).astype(DTYPE_NUMPY[src.type]), src.type)

    class Adjust_contrast():
        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)) -> None:
            self.factor_range = factor_range
            self.dst_range = dst_range

            self.tmp_itr: int = None
            self.tmp_factor: float = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_factor = np.random.rand() * (self.factor_range.max - self.factor_range.min) + self.factor_range.min

            gray_mean = Convert.to_mono8(src).data.mean()

            return Data(
                data=np.clip(gray_mean * (1.0 - self.tmp_factor) + src.data * self.tmp_factor, self.dst_range.min, self.dst_range.max).astype(DTYPE_NUMPY[src.type]),
                type=src.type
            )

    class Adjust_saturation():
        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)) -> None:
            self.factor_range = factor_range
            self.dst_range = dst_range

            self.tmp_itr: int = None
            self.tmp_factor: float = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_factor = np.random.rand() * (self.factor_range.max - self.factor_range.min) + self.factor_range.min

            gray = Convert.to(Convert.to_mono8(src), src.type)

            return Data(
                data=np.clip(gray * (1.0 - self.tmp_factor) + src.data * self.tmp_factor, self.dst_range.min, self.dst_range.max).astype(DTYPE_NUMPY[src.type]),
                type=src.type
            )

    class Adjust_hue():
        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5)) -> None:
            self.factor_range = factor_range

            self.tmp_itr: int = None
            self.tmp_factor: float = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_factor = np.random.rand() * (self.factor_range.max - self.factor_range.min) + self.factor_range.min

            hsv = Convert.to_hsv8(src)
            hsv.data[:,:,0] += np.uint8(self.tmp_factor * 255)

            return Convert.to(hsv, src.type)

    class Adjust_gamma():
        def __init__(
            self, gamma_range: ValueRange = ValueRange(0.5, 1.5), gain_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)
        ) -> None:
            if gain_range.min < 0.0 or gain_range.max < 0.0:
                raise ValueError('`gain_range.min >= 0.0`, `gain_range.max >= 0.0`')

            self.gamma_range = gamma_range
            self.gain_range = gain_range
            self.dst_range = dst_range

            self.tmp_itr: int = None
            self.tmp_gamma: float = None
            self.tmp_gain: float = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_gamma = np.random.rand() * (self.gamma_range.max - self.gamma_range.min) + self.gamma_range.min
                self.tmp_gain = np.random.rand() * (self.gain_range.max - self.gain_range.min) + self.gain_range.min

            img = np.float32(src.data)
            img = self.dst_range.max * self.tmp_gain * np.power(img / self.dst_range.max, self.tmp_gamma)
            img = DTYPE_NUMPY[src.type](np.clip(img, self.dst_range.min, self.dst_range.max))

            return Data(img, src.type)
