from .structure import *

class Resize():
    def __init__(self, output_size: Tuple[int, int], interpolation: int = INTER_LINEAR) -> None:
        """Setting up the resize_2d function.

        Args:
            output_size (Tuple[int, int]): Output size of the image.
            interpolation (int, optional): Image interpolation method. [INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4]. Defaults to INTER_LINEAR.

        Return:
            function (step_itr: int, src: Data) -> Data
        """
        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self, step_itr: int, src: Data) -> Data:
        """Resize an image.

        Args:
            step_itr (int): Number of steps.
            src (Data): Input data.

        Returns:
            Data: Resized data.
        """
        return Data(data=cv2.resize(src.data, dsize=self.output_size, interpolation=self.interpolation), type=src.type)
