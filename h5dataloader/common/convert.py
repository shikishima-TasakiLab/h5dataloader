from .structure import *

class Convert():

    @staticmethod
    def to_passthrough(src: Data) -> Data:
        return src

    @staticmethod
    def to_mono8(src: Data) -> Data:
        if src.type == TYPE_MONO8: return src
        elif src.type == TYPE_MONO16:
            return Data(np.uint8(src.data / 257.), TYPE_MONO8)
        elif src.type == TYPE_BGR8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGR2GRAY), TYPE_MONO8)
        elif src.type == TYPE_RGB8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGB2GRAY), TYPE_MONO8)
        elif src.type == TYPE_BGRA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGRA2GRAY), TYPE_MONO8)
        elif src.type == TYPE_RGBA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGBA2GRAY), TYPE_MONO8)
        elif src.type == TYPE_HSV8:
            return Data(src.data[:, :, 2], TYPE_MONO8)
        else:
            raise NotImplementedError

    @staticmethod
    def to_mono16(src: Data) -> Data:
        if src.type == TYPE_MONO16: return src
        elif src.type == TYPE_MONO8:
            dst = np.copy(src.data)
        elif src.type == TYPE_BGR8:
            dst = cv2.cvtColor(src.data, cv2.COLOR_BGR2GRAY)
        elif src.type == TYPE_RGB8:
            dst = cv2.cvtColor(src.data, cv2.COLOR_RGB2GRAY)
        elif src.type == TYPE_BGRA8:
            dst = cv2.cvtColor(src.data, cv2.COLOR_BGRA2GRAY)
        elif src.type == TYPE_RGBA8:
            dst = cv2.cvtColor(src.data, cv2.COLOR_RGBA2GRAY)
        elif src.type == TYPE_HSV8:
            dst = np.copy(src.data[:,:,2])
        else:
            raise NotImplementedError
        return Data(np.uint16(dst) * 257, TYPE_MONO16)

    @staticmethod
    def to_bgr8(src: Data) -> Data:
        if src.type == TYPE_BGR8: return src
        elif src.type == TYPE_MONO8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_GRAY2BGR), TYPE_BGR8)
        elif src.type == TYPE_MONO16:
            return Data(cv2.cvtColor(np.uint8(src.data / 257.), cv2.COLOR_GRAY2BGR), TYPE_BGR8)
        elif src.type == TYPE_RGB8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGB2BGR), TYPE_BGR8)
        elif src.type == TYPE_BGRA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGRA2BGR), TYPE_BGR8)
        elif src.type == TYPE_RGBA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGBA2BGR), TYPE_BGR8)
        elif src.type == TYPE_HSV8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_HSV2BGR_FULL), TYPE_BGR8)
        else:
            raise NotImplementedError

    @staticmethod
    def to_rgb8(src: Data) -> Data:
        if src.type == TYPE_RGB8: return src
        elif src.type == TYPE_MONO8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_GRAY2RGB), TYPE_RGB8)
        elif src.type == TYPE_MONO16:
            return Data(cv2.cvtColor(np.uint8(src.data / 257.), cv2.COLOR_GRAY2RGB), TYPE_RGB8)
        elif src.type == TYPE_BGR8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGR2RGB), TYPE_RGB8)
        elif src.type == TYPE_BGRA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGRA2RGB), TYPE_RGB8)
        elif src.type == TYPE_RGBA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGBA2RGB), TYPE_RGB8)
        elif src.type == TYPE_HSV8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_HSV2RGB_FULL), TYPE_RGB8)
        else:
            raise NotImplementedError

    @staticmethod
    def to_bgra8(src: Data) -> Data:
        if src.type == TYPE_BGRA8: return src
        elif src.type == TYPE_MONO8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_GRAY2BGRA), TYPE_BGRA8)
        elif src.type == TYPE_MONO16:
            return Data(cv2.cvtColor(np.uint8(src.data / 257.), cv2.COLOR_GRAY2BGRA), TYPE_BGRA8)
        elif src.type == TYPE_BGR8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGR2BGRA), TYPE_BGRA8)
        elif src.type == TYPE_RGB8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGB2BGRA), TYPE_BGRA8)
        elif src.type == TYPE_RGBA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGBA2BGRA), TYPE_BGRA8)
        elif src.type == TYPE_HSV8:
            h, w = src.data.shape[:2]
            bgr = cv2.cvtColor(src.data, cv2.COLOR_HSV2BGR_FULL)
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            return Data(np.concatenate([bgr, alpha], axis=2), TYPE_BGRA8)
        else:
            raise NotImplementedError

    @staticmethod
    def to_rgba8(src: Data) -> Data:
        if src.type == TYPE_RGBA8: return src
        elif src.type == TYPE_MONO8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_GRAY2RGBA), TYPE_RGBA8)
        elif src.type == TYPE_MONO16:
            return Data(cv2.cvtColor(np.uint8(src.data / 257.), cv2.COLOR_GRAY2RGBA), TYPE_RGBA8)
        elif src.type == TYPE_BGR8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGR2RGBA), TYPE_RGBA8)
        elif src.type == TYPE_RGB8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_RGB2RGBA), TYPE_RGBA8)
        elif src.type == TYPE_BGRA8:
            return Data(cv2.cvtColor(src.data, cv2.COLOR_BGRA2RGBA), TYPE_RGBA8)
        elif src.type == TYPE_HSV8:
            h, w = src.data.shape[:2]
            rgb = cv2.cvtColor(src.data, cv2.COLOR_HSV2RGB_FULL)
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            return Data(np.concatenate([rgb, alpha], axis=2), TYPE_RGBA8)
        else:
            raise NotImplementedError

    @staticmethod
    def to_hsv8(src: Data) -> Data:
        if src.type == TYPE_HSV8: return src
        elif src.type == TYPE_MONO8:
            h, w = src.data.shape[:2]
            dst = np.zeros((h, w, 3), dtype=np.uint8)
            dst[:, :, 2] = src.data
            return Data(dst, TYPE_HSV8)
        elif src.type == TYPE_MONO16:
            h, w = src.data.shape[:2]
            dst = np.zeros((h, w, 3), dtype=np.uint8)
            dst[:, :, 2] = np.uint8(np.copy(src.data) / 257.)
            return Data(dst, TYPE_HSV8)
        elif src.type in [TYPE_BGR8, TYPE_BGRA8]:
            return Data(cv2.cvtColor(src.data[:,:,:3], cv2.COLOR_BGR2HSV_FULL), TYPE_HSV8)
        elif src.type in [TYPE_RGB8, TYPE_RGBA8]:
            return Data(cv2.cvtColor(src.data[:,:,:3], cv2.COLOR_RGB2HSV_FULL), TYPE_HSV8)
        else:
            raise NotImplementedError

    TO_FUNCTIONS: dict = {
        TYPE_MONO8: to_mono8,
        TYPE_MONO16: to_mono16,
        TYPE_BGR8: to_bgr8,
        TYPE_RGB8: to_rgb8,
        TYPE_BGRA8: to_bgra8,
        TYPE_RGBA8: to_rgba8,
        TYPE_HSV8: to_hsv8
    }

    @staticmethod
    def to(src: Data, type: str) -> Data:
        to_func = Convert.TO_FUNCTIONS.get(type, None)
        if to_func is None:
            raise NotImplementedError
        return to_func(src)
