from .structure import *

DEPTH_NORMALIZE_INF = 2.0

class Normalization():
    supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_POSE]

    def __init__(self, type: str, norm_range: ValueRange, inf: float = DEPTH_NORMALIZE_INF, *args, **kwargs) -> None:
        self.set_norm_range(norm_range)
        self.set_inf(inf)

        if type in [
            TYPE_UINT8, TYPE_INT8, TYPE_INT16, TYPE_INT32, TYPE_INT64, TYPE_FLOAT16, TYPE_FLOAT32, TYPE_FLOAT64,
            TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8
        ]:
            self.__call__ = self._norm_default
        elif type in [TYPE_DEPTH]:
            self.__call__ = self._norm_depth
        elif type in [TYPE_POSE]:
            self.__call__ = self._norm_pose
        else:
            raise NotImplementedError

    def set_norm_range(self, norm_range: ValueRange) -> None:
        self._norm_range = norm_range

    def set_inf(self, inf) -> None:
        self._inf = inf

    def __call__(self, step_itr: int, src: Data) -> Data:
        return src

    def _norm_default(self, step_itr: int, src: Data) -> Data:
        dst = np.float32(src.data)
        return Data(
            data=(dst - self._norm_range.min) / (self._norm_range.max - self._norm_range.min),
            type=src.type,
            normalized=True
        )

    def _norm_depth(self, step_itr: int, src: Data) -> Data:
        in_range = (self._norm_range.min <= src.data) & (src.data <= self._norm_range.max)
        dst = np.float32(src.data)
        return Data(
            data=np.where(in_range, (dst - self._norm_range.min) / (self._norm_range.max - self._norm_range.min), self._inf),
            type=src.type,
            normalized=True
        )

    def _norm_pose(self, step_itr: int, src: Data) -> Data:
        dst = np.float32(src.data)
        dst[0:3] = (src.data[0:3] - self._norm_range.min) / (self._norm_range.max - self._norm_range.min)
        return Data(data=dst, type=src.type, normalized=True)
