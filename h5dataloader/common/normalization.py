from .structure import *

DEPTH_NORMALIZE_INF = 2.0

def norm_default(norm_range: ValueRange, *args, **kwargs):
    def _norm_image(src: Data) -> Data:
        dst = np.float32(src.data)
        return Data(
            data=(dst - norm_range.min) / (norm_range.max - norm_range.min),
            type=src.type,
            normalized=True
        )
    return _norm_image

def norm_passthrough(*args, **kwargs):
    def _norm_passthrough(src: Data) -> Data:
        return src
    return _norm_passthrough

def norm_depth(norm_range: ValueRange, inf: float = DEPTH_NORMALIZE_INF, *args, **kwargs):
    def _norm_depth(src: Data) -> Data:
        in_range = (norm_range.min <= src.data) & (src.data <= norm_range.max)
        dst = np.float32(src.data)
        return Data(
            data=np.where(in_range, (dst - norm_range.min) / (norm_range.max - norm_range.min), inf),
            type=src.type,
            normalized=True
        )
    return _norm_depth

def norm(type: str, norm_range: ValueRange, inf: float = DEPTH_NORMALIZE_INF, *args, **kwargs):
    if type in [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8]:
        return norm_default(norm_range)
    elif type in [TYPE_DEPTH]:
        return norm_depth(norm_range, inf)
    else:
        raise NotImplementedError
