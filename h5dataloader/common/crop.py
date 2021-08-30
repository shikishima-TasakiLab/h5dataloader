from .structure import *

def _init_left_top(out_h_w: int, in_h_w: int) -> int:
    return 0

def _init_center(out_h_w: int, in_h_w: int) -> int:
    return (out_h_w - in_h_w) // 2

def _init_right_bottom(out_h_w: int, in_h_w: int) -> int:
    return out_h_w - in_h_w

_CROP_INIT: Dict[int, _init_center] = {
    CROP_TOP: _init_left_top,
    CROP_CENTER: _init_center,
    CROP_BOTTOM: _init_right_bottom
}

class Crop():
    supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_SEMANTIC2D]

    def __init__(self, output_size: Tuple[int, int], vertical: int = CROP_CENTER, horizontal: int = CROP_CENTER) -> None:
        """Setting up the crop function.

        Args:
            output_size (Tuple[int, int]): Output size of the image.
            vertical (int): CROP_TOP, CROP_CENTER or CROP_BOTTOM.
            horizontal (int): CROP_LEFT, CROP_CENTER or CROP_RIGHT.

        Return:
            function (step_itr: int, src: Data) -> Data
        """
        self.set_output_size(output_size)
        self.set_vertical(vertical)
        self.set_horizontal(horizontal)

    def set_output_size(self, output_size: Tuple[int, int]) -> None:
        self._out_height, self._out_width = output_size

    def set_vertical(self, vertical: int) -> None:
        if vertical < CROP_TOP or CROP_BOTTOM < vertical:
            raise ValueError('`vertical` must be CROP_TOP, CROP_CENTER or CROP_BOTTOM.')
        self._y_init = _CROP_INIT[vertical]

    def set_horizontal(self, horizontal: int) -> None:
        if horizontal < CROP_LEFT or CROP_RIGHT < horizontal:
            raise ValueError('`horizontal` must be CROP_LEFT, CROP_CENTER or CROP_RIGHT.')
        self._x_init = _CROP_INIT[horizontal]

    def __call__(self, step_itr: int, src: Data) -> Data:
        """Crop the image.

        Args:
            step_itr (int): Number of steps.
            src (Data): Input data.

        Returns:
            Data: Cropped data.
        """
        height, width = src.data.shape[:2]
        if height > self._out_height or width > self._out_width:
            raise ValueError("`height > out_height or width > out_width`")
        y = self._y_init(self._out_height, height)
        x = self._x_init(self._out_width, width)
        return Data(data=src.data[y:y+self._out_height, x:x+self._out_width], type=src.type)

class Random_crop():
    supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_SEMANTIC2D]

    def __init__(self, output_size: Tuple[int, int]) -> None:
        """Setting up the random_crop function.

        Args:
            output_size (Tuple[int, int]): Output size of the image.

        Return:
            function (step_itr: int, src: Data) -> Data
        """
        self.set_output_size(output_size)

        self._tmp_itr: int = None
        self._tmp_x: int = None
        self._tmp_y: int = None

    def set_output_size(self, output_size: Tuple[int, int]) -> None:
        self._out_height, self._out_width = output_size

    def __call__(self, step_itr: int, src: Data) -> Data:
        """Crop the image.

        Args:
            step_itr (int): Number of steps.
            src (Data): Input data.

        Returns:
            Data: Cropped data.
        """
        height, width = src.data.shape[:2]
        if height > self._out_height or width > self._out_width:
            raise ValueError("`height > out_height or width > out_width`")
        if self._tmp_itr != step_itr:
            self._tmp_itr = step_itr
            self._tmp_y = np.random.randint(0, height - self._out_height + 1)
            self._tmp_x = np.random.randint(0, width - self._out_width + 1)
        return Data(data=src.data[self._tmp_y:self._tmp_y+self._out_height, self._tmp_x:self._tmp_x+self._out_width], type=src.type)
