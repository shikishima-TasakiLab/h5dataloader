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
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_SEMANTIC2D]

        def __init__(
            self, range_angle: ValueRange, range_scale: ValueRange, interpolation = INTER_LINEAR, borderMode = BORDER_CONSTANT, borderValue = 0
        ) -> None:
            if isinstance(range_angle, ValueRange) is False: raise TypeError('`type(range_angle)` must be ValueRange.')
            if len(range_angle) != 2: raise ValueError('`len(range_angle)` must be 2.')
            if range_angle.min > range_angle.max: raise ValueError('`range_angle.min < range_angle.max')

            if isinstance(range_scale, ValueRange) is False: raise TypeError('`type(range_scale)` must be ValueRange.')
            if len(range_scale) != 2: raise ValueError('`len(range_scale)` must be 2.')
            if range_scale.min > range_scale.max: raise ValueError('`range_scale.min < range_scale.max')

            self._range_angle = range_angle
            self._range_scale = range_scale
            self._interpolation = interpolation
            self._borderMode = borderMode
            self._borderValue = borderValue

            self._tmp_itr: int = None
            self._rotmat: np.ndarray = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            height, width = src.data.shape[:2]

            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr

                angle = np.random.rand() * (self._range_angle.max - self._range_angle.min) + self._range_angle.min
                scale = (1.0 - np.random.rand()) * (self._range_scale.max - self._range_scale.min) + self._range_scale.min

                self._rotmat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            return Data(
                data=cv2.warpAffine(src.data, self._rotmat, dsize=(width, height), flags=self._interpolation, borderMode=self._borderMode, borderValue=self._borderValue),
                type=src.type
            )

    class Translation_2d():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_DISPARITY, TYPE_SEMANTIC2D]

        def __init__(self, range_vshift: ValueRange, range_hshift: ValueRange, interpolation = INTER_LINEAR, borderMode = BORDER_CONSTANT, borderValue = 0) -> None:
            self.set_range_vshift(range_vshift)
            self.set_range_hshift(range_hshift)
            self.set_interpolation(interpolation)
            self.set_borderMode(borderMode)
            self.set_borderValue(borderValue)

            self._tmp_itr: int = None
            self._rotmat: np.ndarray = None

        def set_range_vshift(self, range_vshift: ValueRange) -> None:
            if isinstance(range_vshift, ValueRange) is False: raise TypeError('`type(range_vshift)` must be ValueRange.')
            if len(range_vshift) != 2: raise ValueError('`len(range_vshift)` must be 2.')
            if range_vshift.min > range_vshift.max: raise ValueError('`range_vshift.min < range_vshift.max')
            self._range_vshift = range_vshift

        def set_range_hshift(self, range_hshift: ValueRange) -> None:
            if isinstance(range_hshift, ValueRange) is False: raise TypeError('`type(range_hshift)` must be ValueRange.')
            if len(range_hshift) != 2: raise ValueError('`len(range_hshift)` must be 2.')
            if range_hshift.min > range_hshift.max: raise ValueError('`range_hshift.min < range_hshift.max')
            self._range_hshift = range_hshift

        def set_interpolation(self, interpolation) -> None:
            self._interpolation = interpolation

        def set_borderMode(self, borderMode) -> None:
            self._borderMode = borderMode

        def set_borderValue(self, borderValue) -> None:
            self._borderValue = borderValue

        def __call__(self, step_itr: int, src: Data) -> Data:
            height, width = src.data.shape[:2]

            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr

                hshift = np.random.rand() * (self._range_hshift.max - self._range_hshift.min) + self._range_hshift.min
                vshift = np.random.rand() * (self._range_vshift.max - self._range_vshift.min) + self._range_vshift.min

                self._rotmat = np.array([[1.0, 0.0, hshift], [0.0, 1.0, vshift]], dtype=np.float)
            return Data(
                data=cv2.warpAffine(src.data, self._rotmat, dsize=(width, height), flags=self._interpolation, borderMode=self._borderMode, borderValue=self._borderValue),
                type=src.type
            )

    class Flip_2d():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_DISPARITY, TYPE_SEMANTIC2D]

        def __init__(self, hflip_rate: float = 0.5, vflip_rate: float = 0.5) -> None:
            self.set_hflip_rate(hflip_rate)
            self.set_vflip_rate(vflip_rate)

            self._tmp_itr: int = None
            self._tmp_hflip: bool = None
            self._tmp_vflip: bool = None

        def set_hflip_rate(self, hflip_rate: float) -> None:
            self._hflip_rate = np.clip(hflip_rate, a_min=0.0, a_max=1.0)

        def set_vflip_rate(self, vflip_rate: float) -> None:
            self._vflip_rate = np.clip(vflip_rate, a_min=0.0, a_max=1.0)

        def __call__(self, step_itr: int, src: Data) -> Data:
            dst = np.copy(src.data)

            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_hflip = np.random.rand() < self._hflip_rate
                self._tmp_vflip = np.random.rand() < self._vflip_rate
            if self._tmp_hflip is True:
                dst = cv2.flip(dst, 1)
            if self._tmp_vflip is True:
                dst = cv2.flip(dst, 0)
            return Data(dst, src.type)

    class Cutout_2d():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_SEMANTIC2D]

        def __init__(self, rate: float, n_holes: int, length: int, masked_value: Union[int, float, ValueRange]) -> None:
            self._rate = np.clip(rate, a_min=0.0, a_max=1.0)
            self._n_holes = n_holes
            self._length_half = length // 2
            self._masking = Augmentation.Bbox_masking(masked_value)

            self._tmp_itr: int = None
            self._tmp_bboxes: List[BoundingBox] = []

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_bboxes = []

                if np.random.rand() < self._rate:
                    height, width = src.data.shape[:2]
                    for _ in range(self._n_holes):
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)

                        self._tmp_bboxes.append(BoundingBox(
                            x1= np.clip(x - self._length_half, 0, width),
                            y1= np.clip(y - self._length_half, 0, height),
                            x2= np.clip(x + self._length_half, 0, width),
                            y2= np.clip(y + self._length_half, 0, height)
                        ))
            return self._masking(src, self._tmp_bboxes)

    class Random_erasing_2d():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_SEMANTIC2D]

        def __init__(self, rate: float, area_ratio_range: ValueRange, aspect_ratio_range: ValueRange, masked_value: Union[int, float, ValueRange]) -> None:
            self._rate = np.clip(rate, a_min=0.0, a_max=1.0)
            self._area_ratio_range = area_ratio_range
            self._aspect_ratio_range = aspect_ratio_range
            self._masking = Augmentation._Bbox_masking(masked_value)

            self._tmp_itr: int = None
            self._tmp_bboxes: List[BoundingBox] = []

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr

                if np.random.rand() < self._rate:
                    height, width = src.data.shape[:2]
                    s = height * width

                    se = (np.random.rand() * (self._area_ratio_range.max - self._area_ratio_range.min) + self._area_ratio_range.min) * s
                    re = np.random.rand() * (self._aspect_ratio_range.max - self._aspect_ratio_range.min) + self._aspect_ratio_range.min

                    he = int(np.sqrt(se * re))
                    we = int(np.sqrt(se / re))

                    xe = np.random.randint(low=0, high=width-we)
                    ye = np.random.randint(low=0, high=height-he)

                    self._tmp_bboxes = [BoundingBox(x1=xe, y1=ye, x2=xe+we, y2=ye+he)]
                else:
                    self._tmp_bboxes = []
            return self._masking(src, self._tmp_bboxes)

    class _Bbox_masking():
        def __init__(self, masked_value: Union[int, float, ValueRange]) -> None:
            self.set_masked_value(masked_value)

        def set_masked_value(self, masked_value: Union[int, float, ValueRange]) -> None:
            self._masked_value = masked_value

            self._use_rand = isinstance(self._masked_value, ValueRange)
            if self._use_rand is True:
                self._use_int = isinstance(self._masked_value.min, int)

        def __call__(self, src: Data, bboxes: List[BoundingBox]) -> Data:
            dst: np.ndarray = np.copy(src.data)

            for bbox in bboxes:
                if self._use_rand is True:
                    bb_shape = dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2].shape
                    if self._use_int is True:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.randint(self._masked_value.min, self._masked_value.max, bb_shape)
                    else:
                        dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = np.random.rand(*bb_shape) * (self._masked_value.max - self._masked_value.min) + self._masked_value.min
                else:
                    dst[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = self._masked_value
            return Data(data=dst, type=src.type)

    class Adjust_brightness():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8]

        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)) -> None:
            self.set_factor_range(factor_range)
            self.set_dst_range(dst_range)

            self._tmp_itr: int = None
            self._tmp_factor: float = None

        def set_factor_range(self, factor_range: ValueRange) -> None:
            self._factor_range = factor_range

        def set_dst_range(self, dst_range: ValueRange) -> None:
            self._dst_range = dst_range

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_factor = np.random.rand() * (self._factor_range.max - self._factor_range.min) + self._factor_range.min

            return Data(np.clip(src.data * self._tmp_factor, self._dst_range.min, self._dst_range.max).astype(DTYPE_NUMPY[src.type]), src.type)

    class Adjust_contrast():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8]

        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)) -> None:
            self.set_factor_range(factor_range)
            self.set_dst_range(dst_range)

            self._tmp_itr: int = None
            self._tmp_factor: float = None

        def set_factor_range(self, factor_range: ValueRange) -> None:
            self._factor_range = factor_range

        def set_dst_range(self, dst_range: ValueRange) -> None:
            self._dst_range = dst_range

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_factor = np.random.rand() * (self._factor_range.max - self._factor_range.min) + self._factor_range.min

            gray_mean = Convert.to_mono8(src).data.mean()

            return Data(
                data=np.clip(gray_mean * (1.0 - self._tmp_factor) + src.data * self._tmp_factor, self._dst_range.min, self._dst_range.max).astype(DTYPE_NUMPY[src.type]),
                type=src.type
            )

    class Adjust_saturation():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8]

        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)) -> None:
            self.set_factor_range(factor_range)
            self.set_dst_range(dst_range)

            self._tmp_itr: int = None
            self._tmp_factor: float = None

        def set_factor_range(self, factor_range: ValueRange) -> None:
            self._factor_range = factor_range

        def set_dst_range(self, dst_range: ValueRange) -> None:
            self._dst_range = dst_range

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_factor = np.random.rand() * (self._factor_range.max - self._factor_range.min) + self._factor_range.min

            gray = Convert.to(Convert.to_mono8(src), src.type)

            return Data(
                data=np.clip(gray * (1.0 - self._tmp_factor) + src.data * self._tmp_factor, self._dst_range.min, self._dst_range.max).astype(DTYPE_NUMPY[src.type]),
                type=src.type
            )

    class Adjust_hue():
        supported = [TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8]

        def __init__(self, factor_range: ValueRange = ValueRange(0.5, 1.5)) -> None:
            self.set_factor_range(factor_range)

            self._tmp_itr: int = None
            self._tmp_factor: float = None

        def set_factor_range(self, factor_range: ValueRange) -> None:
            self._factor_range = factor_range

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_factor = np.random.rand() * (self._factor_range.max - self._factor_range.min) + self._factor_range.min

            hsv = Convert.to_hsv8(src)
            hsv.data[:,:,0] += np.uint8(self._tmp_factor * 255)

            return Convert.to(hsv, src.type)

    class Adjust_gamma():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8]

        def __init__(
            self, gamma_range: ValueRange = ValueRange(0.5, 1.5), gain_range: ValueRange = ValueRange(0.5, 1.5), dst_range: ValueRange = ValueRange(0, 255)
        ) -> None:
            self.set_gamma_range(gamma_range)
            self.set_gain_range(gain_range)
            self.set_dst_range(dst_range)

            self._tmp_itr: int = None
            self._tmp_gamma: float = None
            self._tmp_gain: float = None

        def set_gamma_range(self, gamma_range: ValueRange) -> None:
            self._gamma_range = gamma_range

        def set_gain_range(self, gain_range: ValueRange) -> None:
            if gain_range.min < 0.0 or gain_range.max < 0.0:
                raise ValueError('`gain_range.min >= 0.0`, `gain_range.max >= 0.0`')
            self._gain_range = gain_range

        def set_dst_range(self, dst_range: ValueRange) -> None:
            self._dst_range = dst_range

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_gamma = np.random.rand() * (self._gamma_range.max - self._gamma_range.min) + self._gamma_range.min
                self._tmp_gain = np.random.rand() * (self._gain_range.max - self._gain_range.min) + self._gain_range.min

            img = np.float32(src.data)
            img = self._dst_range.max * self._tmp_gain * np.power(img / self._dst_range.max, self._tmp_gamma)
            img = DTYPE_NUMPY[src.type](np.clip(img, self._dst_range.min, self._dst_range.max))

            return Data(img, src.type)

    class BilateralBlur():
        supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8]

        def __init__(self, sigma_spatial: Union[float, ValueRange], sigma_color: float = None, win_size: int = None) -> None:
            self._tmp_itr: int = None
            self._tmp_sigma_spatial: float = None
            self._tmp_sigma_color: float = None

            self.set_sigma_spatial(sigma_spatial)
            self.set_sigma_color(sigma_color)
            self.set_win_size(win_size)

        def set_sigma_spatial(self, sigma_spatial: Union[float, ValueRange]) -> None:
            self._sigma_spatial = sigma_spatial
            if isinstance(sigma_spatial, float):
                self._get_sigma_spatial = self._get_sigma_spatial_const
            elif isinstance(sigma_spatial, ValueRange):
                self._get_sigma_spatial = self._get_sigma_spatial_rand
            else:
                raise NotImplementedError

        def set_sigma_color(self, sigma_color: float) -> None:
            self._sigma_color = sigma_color
            if self._sigma_color is not None:
                self._get_sigma_color = self._get_sigma_color_const
            else:
                self._get_sigma_color = self._get_sigma_color_std

        def set_win_size(self, win_size: int) -> None:
            self._d: int = -1 if win_size is None else win_size

        def _get_sigma_spatial_rand(self) -> float:
            return np.random.rand() * (self._sigma_spatial.max - self._sigma_spatial.min) + self._sigma_spatial.min

        def _get_sigma_spatial_const(self) -> float:
            return self._sigma_spatial

        def _get_sigma_color_std(self, src: np.ndarray) -> float:
            return src.std()

        def _get_sigma_color_const(self, src: np.ndarray) -> float:
            return self._sigma_color

        def __call__(self, step_itr: int, src: Data) -> Data:
            if self._tmp_itr != step_itr:
                self._tmp_itr = step_itr
                self._tmp_sigma_spatial = self._get_sigma_spatial()

            self._tmp_sigma_color = self._get_sigma_color(src)

            return Data(
                data=cv2.bilateralFilter(src.data, self._d, self._sigma_color, self._sigma_spatial),
                type=src.type
            )

    # class RandomPose():
    #     supported = [TYPE_POSE]

    #     def __init__(self, range_tr: Tuple[float, float, float], range_rot: float) -> None:
    #         self.set_range_tr(range_tr)
    #         self.set_range_rot(range_rot)

    #         self._tmp_itr = None

    #     def set_range_tr(self, range_tr: Tuple[float, float, float]) -> None:
    #         self._range_tr = np.array(range_tr, dtype=np.float32)

    #     def set_range_rot(self, range_rot: float) -> None:
    #         self._range_rot = np.deg2rad(range_rot)

    #     def __call__(self, step_itr: int, src: Data) -> Data:
    #         if self._tmp_itr != step_itr:
    #             self._tmp_itr = step_itr
