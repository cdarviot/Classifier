import sys
import pickle
from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtCore import Qt
# from PyQt6 import QtCore, QtWidgets, uic
# from PyQt6.QtCore import Qt
import pyqtgraph as pg
# Import opengl was fixed following this thread: https://github.com/PixarAnimationStudios/USD/issues/1372
import pyqtgraph.opengl as gl
from Sample import *

# Set parameters for plots
plt.rcParams["font.size"] = 13
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

qtCreatorFile = "myGui.ui"  # Enter file here.
Ui_MainWindow, QMainWindow = uic.loadUiType(qtCreatorFile)


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()

        # Initialisation for data storage
        self.echantillon: Sample
        self.img_overlay = []
        self.color_dst = []
        self.ref_plot = []

        # Initialisation of Gui elements
        self.setupUi(self)
        self.img_roi = pg.RectROI([20, 20], [20, 20], pen=(0, 9))  # Create an interactive roi in the image

        # Initialisation classifier
        self.classification_result = []
        self.classes = {}


        # Interaction with the sample description box in the Gui
        self.hyperstack_path.returnPressed.connect(self.get_file_path)  # get the hyperstack path
        self.load_data_button.clicked.connect(self.load_data)  # load the data

        # Interaction with the Display settings box in the Gui
        self.combo_color_space.currentTextChanged.connect(
            self.update_color_space)  # change the display mode (rgb, mono)
        self.find_nps_button.clicked.connect(self.find_nps_auto)  # automatically find nps list
        self.clear_nps_list_button.clicked.connect(self.clear_nps_list)  # clear the np list
        # self.show_mean_spectrum_cb.ValueChanged.connect()

        # Interaction with the classifier:
        self.classifier_path.returnPressed.connect(self.get_model_path)
        self.load_model_button.clicked.connect(self.load_classifier_model)
        self.classify_button.clicked.connect(self.classify_nps)
        self.classes_list.itemSelectionChanged.connect(self.display_nps_from_selected_class)

        # Interaction with th Saving files box in the Gui
        self.save_image_button.clicked.connect(self.save_img)
        self.save_nps_list_button.clicked.connect(self.save_nps_list)

    def get_file_path(self):
        """
        Opens a dialog to choose the data path. The dataset has to be a .tif file.
        @return: insert the datapath 
        @rtype: .tif
        """
        self.hyperstack_path.clear()
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                          'c:\\', "Image files (*.tif)")
        self.hyperstack_path.setText(file_name[0])

    # def get_reference_path(self):
    #     """
    #     Opens a dialog to choose the reference path. The reference has to be a txt file.
    #     @return:
    #     @rtype:
    #     """
    #     self.ref_path.clear()
    #     file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
    #                                                       'c:\\', "Image files (*.txt)")
    #     self.ref_path.setText(file_name[0])

    def get_model_path(self):
        """
        Opens a dialog to  choose the model path. The model has to be a sav file.
        Note the model is composed of two elements: a scaler and a classifier
        @return:
        @rtype:
        """
        self.classifier_path.clear()
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", 'c:\\', "Model (*.sav)" )
        self.classifier_path.setText(file_name[0])

    def load_data(self):
        # TODO: add condition sample path not empty
        """
        Load the data. Creates a RGB dst of the image and displays it in the bottom left window. Data could be 3D stack.
        @return:
        @rtype:
        """
        print('Data is loading')
        self.echantillon = Sample(num_nps_class=1, file_path=self.hyperstack_path.text())
        self.echantillon.create_zbgr_stack()
        self.color_dst = self.echantillon.update_color_dst('RGB')
        self.img_overlay = []
        self.add_img()
        print('Data is loaded')

    def load_classifier_model(self):
        """
        Load the model located at path classifier_path.
        Note that the model is composed of two elements: a scaler and a classifier. Associate the loaded model to the sample.
        @return:
        @rtype:
        """
        model_path = self.classifier_path.text()
        loaded_model = pickle.load(open(model_path, "rb"))
        self.echantillon.classifier_model = loaded_model
        print(self.echantillon.classifier_model)
        return None


    def add_img(self):
        """
        Displays the dst of the raw data that were loaded. If the original data is a 3d stack,
        a cursor appears and allows to scroll across the stack.
        #
        @return:
        @rtype:
        """
        self.img_window.clear()
        if self.echantillon:
            # As OpenCV considers float only when values range from 0-1. If it finds a float value larger than 1 it
            # clips off the value thinking floats only exists between 0-1.
            image_array = np.copy(self.color_dst)
            img = image_array.astype(np.uint8)
            self.img_window.setImage(img=img)
            self.img_window.getView().addItem(self.img_roi)

    def find_nps_auto(self):
        """

        """
        if self.echantillon:
            if self.echantillon.nps_list:
                self.display_nps_auto()
            else:
                rel_thres = float(self.relative_threshold_tb.text())
                self.echantillon.find_nps_in_sample(relative_thres=rel_thres)
                self.echantillon.extract_color_values()
                self.display_nps_auto()
                # self.plot_np_locations()


    def clear_nps_list(self):
        print('I came here')
        self.echantillon.nps_list = []
        self.echantillon.real_nps_number = 0
        self.add_img()
        self.interactive_plot_window.clear()
        # for it in reversed(self.interactive_plot_window.items):
        #     self.interactive_plot_window.removeItem(it)
        # del it

    def update_color_space(self):
        self.interactive_plot_window.clear()
        new_space = self.combo_color_space.currentText()
        self.color_dst = self.echantillon.update_color_dst(new_space)
        if self.echantillon.nps_list:
            self.display_nps_auto()
            self.echantillon.extract_color_values()
            self.plot_np_locations()
            # self.plot_color_values()
        else:
            self.add_img()
        return

    def plot_np_locations(self):
        i = 0
        pos_tot = []
        color_tot = []
        [d, w, h, c] = np.shape(np.array(self.echantillon.zbgr_stack))
        for slice in range(self.echantillon.num_z_slices):
            for nanoparticle in self.echantillon.nps_list[slice]:
                if not nanoparticle.is_discarded:
                    x_pos = nanoparticle.centroid[1] - 4
                    y_pos = nanoparticle.centroid[0] - 4
                    z_pos = i*10
                    color = np.append(np.array(nanoparticle.color_values/255.0), 1.00)
                    color = tuple(color)
                    color_tot.append(color)
                    pos = tuple([x_pos, y_pos, z_pos])
                    pos_tot.append(pos)
            i+=1
        color_tot = np.array(color_tot)
        pos_tot = np.array(pos_tot)
        print(pos_tot.shape)
        print(np.shape(color_tot))
        plot_3dscatter = gl.GLScatterPlotItem(pos=pos_tot,  color=color_tot, size=5, pxMode=False)
        # self.interactive_plot_window est un GLViewWidget.
        plot_3dscatter.translate(-h//2, -w//2, 0)
        self.interactive_plot_window.addItem(plot_3dscatter)


    def display_nps_auto(self):
        self.img_window.clear()
        self.img_overlay = copy.deepcopy(self.color_dst)  # The deepcopy is needed to produce the dst
        # because we are dealing with a list of objects, and not a simple list.
        for slice in range(self.echantillon.num_z_slices):
            for nanoparticle in self.echantillon.nps_list[slice]:
                if not nanoparticle.is_discarded:
                    x_pos = nanoparticle.centroid[1] - 4
                    y_pos = nanoparticle.centroid[0] - 4
                    cv2.rectangle(self.img_overlay[slice], (x_pos, y_pos), (x_pos + 8, y_pos + 8), (255, 0, 0), 1)
        print(np.shape(self.img_overlay)[-1])
        image_array = np.copy(self.img_overlay)
        img = image_array.astype(np.uint8)
        self.img_window.setImage(img=img, levelMode='mono')

    def save_img(self):
        current_slice = self.img_window.currentIndex
        # display_mode = self.combo_disp_mode.currentText()

        img = self.img_overlay.copy()

        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = self.image_filename.text() + '.tif'
        cv2.imwrite(filename, img)

    def save_nps_list(self):
        np.set_printoptions(suppress=True)
        matrix = []
        label = self.class_label_tb.text()
        name = self.nps_list_filename.text() + '.csv'
        column_names_deb = ['ID', 'x0', 'y0', 'Label', 'Num channels']

        if self.echantillon:
            col_features = ["Feat{}".format(i) for i in range(self.echantillon.num_channels)]
            column_names = column_names_deb + col_features
            if self.echantillon.nps_list:
                for i in range(len(self.echantillon.nps_list)):
                    selected_z_list = self.echantillon.nps_list[i]
                    for k in range(len(selected_z_list)):
                        selected_np = selected_z_list[k]
                        selected_np.ID = k  # TODO: change this one !!! This ID will be overwritten for every slice iteration
                        if not np.isnan(selected_np.color_values).any():
                            col = [k, selected_np.centroid[0], selected_np.centroid[1], label,
                                   self.echantillon.num_channels] + list(selected_np.color_values)
                            matrix.append(col)
                            if selected_np.is_discarded:
                                label = "Aggregate"
                df = pd.DataFrame(matrix)
                df.dropna()
                df.to_csv(name, index=False, header=column_names)
        return None

    def classify_nps(self):
        self.classes_list.clear()

        self.save_nps_list()
        filename = self.nps_list_filename.text() + '.csv'
        dataset = pd.read_csv(filename)
        dataset = pd.read_csv(filename).dropna()
        data_set_size = len(dataset.ID)
        features = dataset[["Feat{}".format(i) for i in range(self.echantillon.num_channels)]]
        labels = self.echantillon.classifier_model.classes_
        result = self.echantillon.classifier_model.predict(features.values)
        counts_per_class = [np.sum([result == label]) for label in labels]
        percents = [counts/np.sum(counts_per_class)*100 for counts in counts_per_class]
        text_result = '\n'.join(["Counts for class {} : {} ({:.2f}%)".format(labels[i], counts_per_class[i], percents[i])
                                 for i in range(len(labels))])
        self.results_txtbox.setText(text_result)
        [self.classes_list.addItem(label) for label in labels]
        self.classification_result = result
        for label in labels:
            color = list(np.random.choice(range(256), size=3))
            if label == "B":
                def_color = (0, 0, 255)
            elif label == "G":
                def_color = (0, 255, 0)
            elif label == "Y":
                def_color = (255, 0, 0)
            else:
                def_color = tuple([int(c) for c in color]) # TODO: cleaning this part
            self.classes[label] = def_color
        print("Came out of there")

    def display_nps_from_selected_class(self, selected_label="None"):
        # TODO: implementation
        currentSlice = self.img_window.currentIndex
        # if self.combo_disp_mode.currentText() == 'Projection mono':
        #     img = self.echantillon.zxy_projected_stack[currentSlice, :, :].copy()
        #     img = img / np.max(img) * 255
        #     self.img_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # else:
        self.img_overlay = self.color_dst[currentSlice].copy() # The deepcopy is needed to produce the dst
        filename = self.nps_list_filename.text() + '.csv'
        dataset = pd.read_csv(filename).dropna()
        for selected_label in self.classes_list.selectedItems():
            # color = list(np.random.choice(range(256), size=3))
            # def_color = tuple([int(c) for c in color]) # TODO: cleaning this part

            selected_y0 = dataset["x0"].values[self.classification_result == selected_label.text()]
            selected_x0 = dataset["y0"].values[self.classification_result == selected_label.text()]
            for i in range(len(selected_x0)):
                cv2.rectangle(self.img_overlay, (selected_x0[i]-4, selected_y0[i]-4), (selected_x0[i] + 4, selected_y0[i] + 4),
                              self.classes[selected_label.text()], 1)
        level_min = np.amin(self.img_overlay[1, :, :])
        level_max = np.amax(self.img_overlay[1, :, :])
        levels_g = [(level_min, level_max), (level_min, level_max), (level_min, level_max)]

        np_levels = np.array(levels_g)
        print(np.shape(self.img_overlay)[-1])
        print(np.shape(np_levels))
        self.img_window.setImage(img=self.img_overlay)
        return None



if __name__ == '__main__':
    # Qt.AA_EnableHighDpiScaling = 1
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    main = Main()
    main.show()
    sys.exit(app.exec())
