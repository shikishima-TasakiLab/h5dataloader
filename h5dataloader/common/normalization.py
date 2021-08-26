from .structure import *

DEPTH_NORMALIZE_INF = 2.0

class Normalization():
    def __init__(self, type: str, norm_range: ValueRange, inf: float = DEPTH_NORMALIZE_INF, *args, **kwargs):
        self.norm_range = norm_range
        self.inf = inf

        if type in [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8]:
            self.__call__ = self.norm_default
        elif type in [TYPE_DEPTH]:
            self.__call__ = self.norm_depth
        elif type in [TYPE_POSE]:
            self.__call__ = self.norm_pose
        else:
            raise NotImplementedError

    def __call__(self, step_itr: int, src: Data) -> Data:
        return src

    def norm_default(self, step_itr: int, src: Data) -> Data:
        dst = np.float32(src.data)
        return Data(
            data=(dst - self.norm_range.min) / (self.norm_range.max - self.norm_range.min),
            type=src.type,
            normalized=True
        )

    def norm_depth(self, step_itr: int, src: Data) -> Data:
        in_range = (self.norm_range.min <= src.data) & (src.data <= self.norm_range.max)
        dst = np.float32(src.data)
        return Data(
            data=np.where(in_range, (dst - self.norm_range.min) / (self.norm_range.max - self.norm_range.min), self.inf),
            type=src.type,
            normalized=True
        )

    def norm_pose(self, step_itr: int, src: Data) -> Data:
        dst = np.float32(src.data)
        dst[0:3] = (src.data[0:3] - self.norm_range.min) / (self.norm_range.max - self.norm_range.min)
        return Data(data=dst, type=src.type, normalized=True)
