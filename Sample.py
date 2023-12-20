import cv2
import matplotlib.pyplot as plt
import pandas as pd
import skimage.feature as skifeat
from find_maxima import find_maxima
from scipy import ndimage
import copy

from Nanoparticle import *


class Sample:
    """
    The class Sample instantiate a Sample containing a certain number of different markers, raw data acquired for a
    certain number of hyperspectral channels, an external reference, a number of cells and a list of instances
    'Nanoparticle'. In the setup used, there is a variation of lambda along the x-axis within the same spectral slice
    of a hyperstack. The image processing is performed accordingly.
    
    Attributes:
        num_nps_classes: number of classes of NPs in the sample
        num_channels: number of channels (i.e. wavelengths) in the sample = 3 bc RGB
        rgb_stack: RGB stack with shape [z (number of slices), y (height), x (width), 3 (RGB)]
        nps_list: list of Nanoparticles per slice with shape
            [z (number of slices), n (number of nanoparticles in that slice)]
        total_detected: total number of detected Nanoparticles
        real_nps_number: number of non-discarded Nanoparticles
    """

    def __init__(
            self,
            num_nps_class: int,
            file_path: str):
        """
        @param num_nps_class:
        @type num_nps_class:
        @param file_path: filepath to load picture
        @type file_path: str
        """
        self.num_nps_classes = num_nps_class
        self.file_path = file_path
        self.num_channels = 3

        self.zbgr_stack = []
        self.color_dst = []
        self.current_color_space = 'BGR'
        self.zxy_projected_stack = []
        self.num_z_slices = 0
        self.rgb_stack = []
        self.nps_list = []
        self.total_detected = 0
        self.real_nps_number = 0
        self.classifier_model = []
        # Optical config

        self.mag = 40
        self.z_space = 1

    def create_zbgr_stack(self):
        """
        From attribute file_path, creates a list of BGR images [x, y, c],
        with c = 3 because we are dealing with BGR pictures.

        The length of the list corresponds to the z dimension.

        NB: the images are kept with the BGR format to avoid losses in multiple conversions.
        The conversion is done only when the displaying of the images occurs.

        @return: returns the list of images that was created
        @rtype: tuple
        """
        self.zbgr_stack = cv2.imreadmulti(self.file_path, flags=cv2.IMREAD_COLOR)[1]
        self.num_z_slices = len(self.zbgr_stack)
        self.color_dst = copy.deepcopy(self.zbgr_stack)
        self.current_color_space = 'BGR'
        return self.zbgr_stack

    def update_color_dst(self, new_color_space='HSV'):
        dst = np.array(copy.deepcopy(self.zbgr_stack))
        for i in range(len(dst)):
            dst[i] = cv2.cvtColor(dst[i], choose_conversion_flag(new_color_space))
        self.color_dst = dst
        self.current_color_space = new_color_space
        return self.color_dst

    def find_nps_in_sample(self, relative_thres=15):
        """
        Create the list of NPs in the sample organized as a list of list of nps per slice.

        @param relative_thres: relative threshold for the filtering of detected NPs below the noise.
        @return: None
        """
        for i in range(0, self.num_z_slices):
            xy_slice = self.zbgr_stack[i]
            img_data = xy_slice.astype(np.float64)
            if img_data.shape.__len__() > 2:
                img_data = (np.sum(img_data, 2) / 3.0)
            if np.max(img_data) > 255 or np.min(img_data) < 0:
                print('warning: your image should be scaled between 0 and 255 (8-bit).')
            np_in_slice = []
            res = self.find_maxima_refined(img_data, noise_tolerance=relative_thres)
            print('so far so good')
            for centroid in res:
                # Verify if the NP is really in the current Z-slide
                # Step 1 : extract signal from box around (x,y) and 5 adjacent slides if possible
                nanopart = Nanoparticle()
                nanopart.centroid = centroid
                scope_z_search = 2
                z_min = max(0, i - scope_z_search)
                z_max = min(i + scope_z_search, self.num_z_slices)

                signals = [np.sum(nanopart.extract_colors(zcolor_slice=self.zbgr_stack[j],
                                                   box_type="star")) for j in range(z_min, z_max)]

                # Step 2 : find max signal
                max_idx = np.argmax(signals)
                print(max_idx)
                # Step 3 : verify if max signal is located in i
                if z_min + max_idx == i:
                    # Step 4: if true then create the NP.
                    nanopart.centroid.append(i)
                    np_in_slice.append(nanopart)
                else:
                    del nanopart
            self.nps_list.append(np_in_slice)
        return self.nps_list

    def extract_color_values(self, box_type='star', std_dev_thres=3.4):
        for i in range(0, self.num_z_slices):
            for nanoparticle in self.nps_list[i]:
                nanoparticle.extract_colors(zcolor_slice=self.color_dst[i], box_type=box_type)

    def find_maxima_via_Harris(self, xy_slice, relative_thres: float):
        local_maxima = cv2.cornerHarris(xy_slice, 3, 3, relative_thres)
        local_maxima = cv2.dilate(local_maxima, None)
        ret, local_maxima = cv2.threshold(local_maxima, relative_thres * local_maxima.max(), 255, 0)
        local_maxima = np.uint8(local_maxima)
        # Find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(local_maxima)
        res = centroids.T
        res = res.tolist()
        return res

    def find_maxima_via_skifeat(self, xy_slice, relative_thres: float):
        peaks = skifeat.peak_local_max(xy_slice, min_distance=5, threshold_rel=relative_thres, indices=True)
        res = np.int0(peaks).tolist()
        return res

    def find_local_maxima_np(self, img_data):
        # This is the numpy/scipy version of the above function (find local maxima).
        # Its a bit faster, and more compact code.
        # Filter data with maximum filter to find maximum filter response in each neighbourhood
        max_out = ndimage.filters.maximum_filter(img_data, size=3)
        # Find local maxima.
        local_max = np.zeros(img_data.shape)
        local_max[max_out == img_data] = 1
        local_max[img_data == np.min(img_data)] = 0
        return local_max.astype(np.bool)

    def find_maxima_refined(self, xy_slice, noise_tolerance: float = 10):
        img = np.copy(xy_slice)
        local_max = self.find_local_maxima_np(img)
        y, x, _ = find_maxima(img, local_max.astype(np.uint8), noise_tolerance)
        res = np.array([y, x])
        res = res.T
        res = res.tolist()
        return res

    def white_balancing(self, img):
        """
        Performs a white balance on a given RGB image.

        @param img:
        @type img:
        @return:
        @rtype:
        """
        result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                l, a, b = result[x, y, :]
                # fix for CV correction
                l *= 100 / 255.0
                result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
                result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result

    def display_rgb_slice(self, zslice_number=0, save_file=False):
        """
        @deprecated
        Display the selected rgb slice in a matplotlib
        @param zslice_number:
        @type zslice_number:
        @param save_file:
        @type save_file:
        @return:
        @rtype:
        """
        plt.subplots()
        plt.axis('off')
        chosen_slice = self.rgb_stack[zslice_number]
        chosen_slice_rgb = cv2.cvtColor(chosen_slice, cv2.COLOR_BGR2RGB)
        plt.imshow(chosen_slice_rgb)
        plt.show()
        if save_file:
            cv2.imwrite('reconstructedRgbImage.tif', chosen_slice)
        return

    def display_nps_on_slice(self, zslice_number=0, np_filter='real_nps', save_file=False):
        # TODO: CORRECTING THIS FUNCTION THAT DOES NOT SEEM TO WORK. NB: the display of a red square
        # around the NPs seems to work in the GUI though.
        plt.subplots()
        plt.axis('off')
        # pick up slice
        color_slice = self.zbgr_stack[zslice_number]
        # color_slice = (chosen_slice - np.min(chosen_slice)) / np.max(chosen_slice - np.min(chosen_slice)) * 255
        color_slice = cv2.cvtColor(color_slice, code=cv2.COLOR_BGR2RGB)
        # color_slice = np.int0(color_slice)
        centroids_list = []
        # create dst
        # overlay red squares around centroid

        nps_on_slice = self.nps_list[zslice_number]
        for nanoparticle in nps_on_slice:
            if np_filter == 'real_nps':
                if not nanoparticle.is_discarded:
                    centroids_list.append(nanoparticle.centroid[0:2])
            elif np_filter == 'discarded_aggregates':
                if nanoparticle.is_discarded:
                    centroids_list.append(nanoparticle.centroid[0:2])
            elif np_filter == 'all':
                centroids_list.append(nanoparticle.centroid[0:2])
            else:
                return "Invalid parameter for np_filter"
        centroids_array = np.array(centroids_list)
        print(np.shape(color_slice))
        color_slice[centroids_array[:, 0], centroids_array[:, 1], :] = [255, 0, 0]
        plt.imshow(color_slice)
        plt.show()

        if save_file:
            color_slice[centroids_array[:, 0], centroids_array[:, 1]] = [0, 0, 255]
            cv2.imwrite("Z{}_-{}-overlay.tif".format(zslice_number, np_filter), color_slice)
        return

    def plot_real_np_spectra(self):
        """
        Plot all valid spectra in the sample i.e. NPs with parameter is_discarded == 0
        @return:
        """
        return


def choose_conversion_flag(new_color_space: str) -> int:
    return {
        'RGB': eval('cv2.COLOR_BGR2RGB'),
        'HSV': eval('cv2.COLOR_BGR2HSV'),
        'LAB': eval('cv2.COLOR_BGR2LAB'),
        'LUV': eval('cv2.COLOR_BGR2LUV'),
        'XYZ': eval('cv2.COLOR_BGR2XYZ'),
    }[new_color_space]
