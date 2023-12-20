import numpy as np
from scipy import interpolate


class Nanoparticle:
    """
        @todo: documentation
        """

    def __init__(self):
        self.centroid = []
        self.local_reference = []
        self.color_values = []
        self.normalized_spectrum = []
        self.lambda_offset = []
        self.true_colour = []
        self.np_class = []
        self.false_color = []
        self.interp_function = []
        self.interp_spectrum = []
        self.lambda_range = []
        self.is_discarded = False

    def extract_colors(self, zcolor_slice, box_type='star', box_size=3):
        """

            @param zcolor_slice:
            @param box_type:
            @param box_size:
            @return:
            """
        # Calculate spectrum
        # slice_position = self.centroid[2] - 1
        if box_type == 'star':
            try:
                signal = np.nanmean([zcolor_slice[self.centroid[0], self.centroid[1], :],
                                  zcolor_slice[self.centroid[0] + 1, self.centroid[1], :],
                                  zcolor_slice[self.centroid[0] - 1, self.centroid[1], :],
                                  zcolor_slice[self.centroid[0] - 1, self.centroid[1] - 1, :],
                                  zcolor_slice[self.centroid[0], self.centroid[1] - 1, :]], axis=(0))
                self.color_values = np.uint8(signal)
            except:
                self.is_discarded = True
                signal = [0, 0, 0]
                print('oops some NPs were excluded from the dataset')
        elif box_type == 'square':
            x_np = self.centroid[0]
            y_np = self.centroid[1]
            x_ref = self.local_reference[0]
            y_ref = self.local_reference[1]
            signal = np.nanmean(zcolor_slice[x_np - 2:x_np + 2, y_np - 2:y_np + 2, :], axis=(0, 1))
            signal = np.nanmean(signal, axis=0)
        else:
            return "Invalid parameter for box type"
        return signal

    def classify_np(self):

        # TODO:
        return None
