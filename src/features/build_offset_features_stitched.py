import numpy as np
import pkg_resources
from pathlib import Path
import os


class GetFeatures:
    def __init__(self, input_filepaths, output_filepath):
        self.input_data_0 = np.genfromtxt(input_filepaths[0], delimiter=',', skip_header=False)
        self.input_data_1 = np.genfromtxt(input_filepaths[1], delimiter=',', skip_header=False)

        self.log_path = input_filepaths[2]

        self.vid_0_path = input_filepaths[3]

        self.output_filepath = output_filepath

        self.mouse_presence = []
        self.ground_truths = []
        self.nps = []
        self.time_at_gl = []
        self.log_gl_15 = []

        self.gl = np.nan

        self.offsets = []

    def ground_truth(self):
        vid_t = (3600 * int(self.vid_0_path[len(self.vid_0_path)-8:len(self.vid_0_path)-6]) + 60 * int(self.vid_0_path[len(self.vid_0_path)-5:len(self.vid_0_path)-3]) + int(self.vid_0_path[len(self.vid_0_path)-2:])) * 15
        act = []
        act_line = ["Trial ++++++++ active ++++++++"]

        inact = []
        inact_line = ["Trial ------- inactive -------"]

        gl = []

        with open(self.log_path) as f:
            f = f.readlines()

            i=0
            for line in f:
                for phrase in act_line:
                    if phrase in line:
                        act.append(line)
                for phrase in inact_line:
                    if phrase in line:
                        inact.append(line)
                        for phrase in act_line:
                            if not phrase in f[i - 1]:
                                gl.append(int(f[i - 1][59:]))
                i += 1

        counts = np.bincount(gl)
        gls = np.where(counts >= 5)[0]

        self.log_onsets, self.log_offsets = [], []

        for i in range(len(act)):
            on_t = (3600 * int(act[i][11:13]) + 60 * int(act[i][14:16]) + int(act[i][17:19])) * 15 - vid_t
            off_t = (3600 * int(inact[i][11:13]) + 60 * int(inact[i][14:16]) + int(inact[i][17:19])) * 15 - vid_t

            on_tf = np.argwhere(self.input_data_0 == on_t)
            if on_t in self.input_data_0:
                on_tf = on_tf[0][0]
            else:
                on_tf = np.nan
            self.log_onsets.append(on_tf)

            off_tf = np.argwhere(self.input_data_1 == off_t)
            if off_t in self.input_data_1:
                off_tf = off_tf[0][0]
            else:
                off_tf = np.nan
            self.log_offsets.append(off_tf)

        log = 0
        for i in range(len(self.input_data_0)):
            # Keep track if trial is active by log files
            if i in self.log_onsets:
                log = 0
            elif i+7 in self.log_offsets:
                log = 1
            elif i-7 in self.log_offsets:
                log = 0
            self.offsets.append(log)

        node_0 = self.input_data_0[:, 5].astype('uint8')
        node_1 = self.input_data_1[:, 5].astype('uint8')

        nodes = np.append(node_0, node_1)

        nodes_filtered = nodes[nodes != 0]

        counts = np.bincount(nodes_filtered)

        gl_counts = []
        for val in gls:
            gl_counts.append(counts[val])

        self.gl = str(gls[np.argmax(gl_counts)]) + '.0'

    def mouse_in(self):

        for i in range(len(self.input_data_0)):

            if not np.isnan(self.input_data_0[i, 4]) or not np.isnan(self.input_data_1[i, 4]):
                mouse_in = 1
            else:
                mouse_in = 0

            self.mouse_presence.append(mouse_in)

    def nodes_per_second(self):

        nodes_per_s = 0
        for i in range(len(self.input_data_0)):

            if i >= 15:
                diff0 = np.diff(self.input_data_0[i-7:i+7, 5])
                for k in range(len(diff0)):
                    if np.isnan(diff0[k]):
                        diff0[k] = 0
                diff1 = np.diff(self.input_data_1[i-7:i+7, 5])
                for k in range(len(diff1)):
                    if np.isnan(diff1[k]):
                        diff1[k] = 0
                nodes_per_s = np.count_nonzero(diff0) + np.count_nonzero(diff1)
            self.nps.append(nodes_per_s)

    def find_gl(self):
        with open(self.log_path) as f:
            pass

    def at_gl_s(self):
        gl = self.gl
        gl_f = 0

        for i in range(len(self.input_data_0)):

            # Amount of frames spent at goal location
            if self.input_data_0[i, 5].astype('str') == gl or self.input_data_1[i, 5].astype('str') == gl:
                gl_f += 1
            else:
                gl_f = 0

            if gl_f >= 15:
                gl_15 = 1
            else:
                gl_15 = 0

            self.time_at_gl.append(gl_f)
            self.log_gl_15.append(gl_15)

    def build_features_log(self):

        output_data = open(self.output_filepath, 'w')
        output_data.write('Frame number, ground truth, mouse presence, nodes per second, time at gl, time at gl >=15\n')

        for i in range(len(self.input_data_0)):

            output_data.write('{}, {}, {}, {}, {}, {}\n'.format(i, self.offsets[i], self.mouse_presence[i], self.nps[i], self.time_at_gl[i], self.log_gl_15[i]))

        output_data.close()


class Stitch_Those_Files:

    def __init__(self):
        outdir = "{}/data/interim/features_stitched/features.csv".format(os.getcwd())
        self.features_log = open(outdir, 'w')
        self.features_log.write('Frame number, ground truth, mouse presence, nodes per second, time at gl, time at gl >=15\n')

    def stitch_those_files(self):

        rootdir = "{}/data/interim/features".format(os.getcwd())

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                root = os.path.join(rootdir, file)
                data = np.genfromtxt(root, delimiter=',', skip_header=True)

                for i in range(len(data)):
                    self.features_log.write(
                        '{}, {}, {}, {}, {}, {}\n'.format(data[i,0], data[i,1], data[i,2], data[i,3],
                                                          data[i, 4], data[i,5]))

        self.features_log.close()


if __name__ == '__main__':
    rootdir = "{}/data/raw".format(os.getcwd())

    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            for s, d, f in os.walk(os.path.join(subdir, dir)):
                path_log = os.path.join(rootdir, dir, f[0])
                pos_0_log = os.path.join(rootdir, dir, f[1])
                pos_1_log = os.path.join(rootdir, dir, f[2])
                start_track = os.path.join(rootdir, dir)

                input_paths = [pos_0_log, pos_1_log, path_log, start_track]

                output_path = "{}/data/interim/features/features_{}.csv".format(os.getcwd(), dir)

                G = GetFeatures(input_paths, output_path)
                G.ground_truth()
                G.mouse_in()
                G.nodes_per_second()
                G.at_gl_s()
                G.build_features_log()

    S = Stitch_Those_Files()
    S.stitch_those_files()
