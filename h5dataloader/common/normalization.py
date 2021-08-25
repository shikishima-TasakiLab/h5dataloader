from .structure import *

DEPTH_NORMALIZE_INF = 2.0

class Normalization():

    @staticmethod
    def norm_default(norm_range: ValueRange, *args, **kwargs):
        def _norm_image(step_itr: int, src: Data) -> Data:
            dst = np.float32(src.data)
            return Data(
                data=(dst - norm_range.min) / (norm_range.max - norm_range.min),
                type=src.type,
                normalized=True
            )
        return _norm_image

    @staticmethod
    def norm_passthrough(*args, **kwargs):
        def _norm_passthrough(step_itr: int, src: Data) -> Data:
            return src
        return _norm_passthrough

    @staticmethod
    def norm_depth(norm_range: ValueRange, inf: float = DEPTH_NORMALIZE_INF, *args, **kwargs):
        def _norm_depth(step_itr: int, src: Data) -> Data:
            in_range = (norm_range.min <= src.data) & (src.data <= norm_range.max)
            dst = np.float32(src.data)
            return Data(
                data=np.where(in_range, (dst - norm_range.min) / (norm_range.max - norm_range.min), inf),
                type=src.type,
                normalized=True
            )
        return _norm_depth

    @staticmethod
    def norm_pose(norm_range: ValueRange, *args, **kwargs):
        def _norm_pose(step_itr: int, src: Data) -> Data:
            dst = np.float32(src.data)
            dst[0:3] = (src.data[0:3] - norm_range.min) / (norm_range.max - norm_range.min)
            return Data(data=dst, type=src.type, normalized=True)
        return _norm_pose

    @staticmethod
    def norm(type: str, norm_range: ValueRange, inf: float = DEPTH_NORMALIZE_INF, *args, **kwargs):
        if type in [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8]:
            return Normalization.norm_default(norm_range)
        elif type in [TYPE_DEPTH]:
            return Normalization.norm_depth(norm_range, inf)
        elif type in [TYPE_POSE]:
            return Normalization.norm_pose(norm_range)
        else:
            raise NotImplementedError
