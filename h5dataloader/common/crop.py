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

    class Crop_2d():
        def __init__(self, output_size: Tuple[int, int], vertical: int = CROP_CENTER, horizontal: int = CROP_CENTER) -> None:
            """Setting up the crop_2d function.

            Args:
                output_size (Tuple[int, int]): Output size of the image.
                vertical (int): CROP_TOP, CROP_CENTER or CROP_BOTTOM.
                horizontal (int): CROP_LEFT, CROP_CENTER or CROP_RIGHT.

            Return:
                function (step_itr: int, src: Data) -> Data
            """
            self.out_height, self.out_width = output_size

            if vertical < CROP_TOP or CROP_BOTTOM < vertical:
                raise ValueError('`vertical` must be CROP_TOP, CROP_CENTER or CROP_BOTTOM.')
            if horizontal < CROP_LEFT or CROP_RIGHT < horizontal:
                raise ValueError('`horizontal` must be CROP_LEFT, CROP_CENTER or CROP_RIGHT.')

            self.y_init = _CROP_INIT[vertical]
            self.x_init = _CROP_INIT[horizontal]

        def __call__(self, step_itr: int, src: Data) -> Data:
            """Crop the image.

            Args:
                step_itr (int): Number of steps.
                src (Data): Input data.

            Returns:
                Data: Cropped data.
            """
            height, width = src.data.shape[:2]
            if height > self.out_height or width > self.out_width:
                raise ValueError("`height > out_height or width > out_width`")
            y = self.y_init(self.out_height, height)
            x = self.x_init(self.out_width, width)
            return Data(data=src.data[y:y+self.out_height, x:x+self.out_width], type=src.type)

    class Random_crop_2d():
        def __init__(self, output_size: Tuple[int, int]) -> None:
            """Setting up the random_crop_2d function.

            Args:
                output_size (Tuple[int, int]): Output size of the image.

            Return:
                function (step_itr: int, src: Data) -> Data
            """
            self.out_height, self.out_width = output_size

            self.tmp_itr: int = None
            self.tmp_x: int = None
            self.tmp_y: int = None

        def __call__(self, step_itr: int, src: Data) -> Data:
            """Crop the image.

            Args:
                step_itr (int): Number of steps.
                src (Data): Input data.

            Returns:
                Data: Cropped data.
            """
            height, width = src.data.shape[:2]
            if height > self.out_height or width > self.out_width:
                raise ValueError("`height > out_height or width > out_width`")
            if self.tmp_itr != step_itr:
                self.tmp_itr = step_itr
                self.tmp_y = np.random.randint(0, height - self.out_height + 1)
                self.tmp_x = np.random.randint(0, width - self.out_width + 1)
            return Data(data=src.data[self.tmp_y:self.tmp_y+self.out_height, self.tmp_x:self.tmp_x+self.out_width], type=src.type)
