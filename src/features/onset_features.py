import numpy as np


class GetFeatures:
    def __init__(self, input_filepaths, output_filepath):
        self.input_data_0 = np.genfromtxt(input_filepaths[0], delimiter=',', skip_header=False)
        self.input_data_1 = np.genfromtxt(input_filepaths[1], delimiter=',', skip_header=False)

        self.log_path = input_filepaths[2]

        self.vid_0_path = input_filepaths[3]
        self.vid_1_path = input_filepaths[4]

        self.output_filepath = output_filepath

        self.mouse_presence = []
        self.ground_truths = []
        self.nps = []
        self.time_at_gl = []
        self.log_gl_15 = []

    def ground_truth(self):
        vid_t = (3600 * int(self.vid_0_path[63:65]) + 60 * int(self.vid_0_path[66:68]) + int(self.vid_0_path[69:71])) * 15

        act = []
        act_line = ["Trial ++++++++ active ++++++++"]

        inact = []
        inact_line = ["Trial ------- inactive -------"]

        with open(self.log_path) as f:
            f = f.readlines()

            for line in f:
                for phrase in act_line:
                    if phrase in line:
                        act.append(line)
                for phrase in inact_line:
                    if phrase in line:
                        inact.append(line)

        log_onsets, log_offsets = [], []

        for i in range(len(act)):
            on_t = (3600 * int(act[i][11:13]) + 60 * int(act[i][14:16]) + int(act[i][17:19])) * 15 - vid_t
            off_t = (3600 * int(inact[i][11:13]) + 60 * int(inact[i][14:16]) + int(inact[i][17:19])) * 15 - vid_t

            on_tf = np.argwhere(self.input_data_0 == on_t)
            if on_t in self.input_data_0:
                on_tf = on_tf[0][0]
            else:
                on_tf = np.nan
            log_onsets.append(on_tf)

            off_tf = np.argwhere(self.input_data_1 == off_t)
            if off_t in self.input_data_1:
                off_tf = off_tf[0][0]
            else:
                off_tf = np.nan
            log_offsets.append(off_tf)

        log = 0
        k=0
        for i in range(len(self.input_data_0)):
            # Keep track if trial is active by log files
            if i < len(self.input_data_0)-7:
                if i+10 in log_onsets:
                    log = 1
                    k = 20
            if k > 0:
                log = 1
                k -= 1
            else:
                log = 0

            self.ground_truths.append(log)

    def mouse_in(self):

        for i in range(len(self.input_data_0)):

            if not np.isnan(self.input_data_0[i, 4]) or not np.isnan(self.input_data_1[i, 4]):
                mouse_in = 1
            else:
                mouse_in = 0

            self.mouse_presence.append(mouse_in)

    def search_onsets(self):

        self.diff = np.diff(self.mouse_presence)
        self.diff = np.append(self.diff, 0)

    def build_features_log(self):

        output_data = open(self.output_filepath, 'w')
        output_data.write('Frame number, ground truth, mouse presence, diff\n')

        for i in range(len(self.input_data_0)):

            output_data.write('{}, {}, {}, {}\n'.format(i, self.ground_truths[i], self.mouse_presence[i], self.diff[i]))

        output_data.close()


if __name__ == '__main__':
    path_0 = r"C:/Users/Gebruiker/Documents/HexClassifier/data/raw/pos_log_file_lin_0.csv"
    path_1 = r"C:/Users/Gebruiker/Documents/HexClassifier/data/raw/pos_log_file_lin_1.csv"
    path_log = r"C:/Users/Gebruiker/Documents/HexClassifier/data/raw/2019-05-07_14-47-59_hextrack_log"
    vid_0_path = r'C:/Users/Gebruiker/Documents/HexClassifier/data/raw/2019-05-07_14-53-54_cam_0.avi'
    vid_1_path = r'C:/Users/Gebruiker/Documents/HexClassifier/data/raw/2019-05-07_14-53-54_cam_1.avi'

    input_paths = [path_0, path_1, path_log, vid_0_path, vid_1_path]
    output_path = r"C:/Users/Gebruiker/Documents/HexClassifier/data/processed/features.csv"

    G = GetFeatures(input_paths, output_path)
    G.ground_truth()
    G.mouse_in()
    G.search_onsets()
    G.build_features_log()
