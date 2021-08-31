from .structure import *

class Resize():
    supported = [TYPE_MONO8, TYPE_MONO16, TYPE_BGR8, TYPE_RGB8, TYPE_BGRA8, TYPE_RGBA8, TYPE_HSV8, TYPE_DEPTH, TYPE_SEMANTIC2D]

    def __init__(self, output_size: Tuple[int, int], interpolation: int = INTER_LINEAR) -> None:
        """Setting up the resize_2d function.

        Args:
            output_size (Tuple[int, int]): Output size of the image.
            interpolation (int, optional): Image interpolation method. [INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4]. Defaults to INTER_LINEAR.

        Return:
            function (step_itr: int, src: Data) -> Data
        """
        self.set_output_size(output_size)
        self.set_interpolation(interpolation)

    def set_output_size(self, output_size: Tuple[int, int]) -> None:
        self._output_size = output_size

    def set_interpolation(self, interpolation: int) -> None:
        self._interpolation = interpolation

    def __call__(self, step_itr: int, src: Data) -> Data:
        """Resize an image.

        Args:
            step_itr (int): Number of steps.
            src (Data): Input data.

        Returns:
            Data: Resized data.
        """
        dst = src.data if src.data.shape[:2] == self._output_size else cv2.resize(src.data, dsize=self._output_size, interpolation=self._interpolation)
        return Data(data=dst, type=src.type)
