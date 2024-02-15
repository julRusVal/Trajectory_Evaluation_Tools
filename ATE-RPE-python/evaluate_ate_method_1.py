#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy
import argparse
import associate
import os


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if (numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


if __name__ == "__main__":
    # Set parameters in script
    # Input
    data_set_directory = "/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/iros_method_2"
    if not os.path.isdir(data_set_directory):
        raise ValueError("Invalid data set directory")

    first_file_path = data_set_directory + "/analysis_final_3d.csv"
    second_file_path = data_set_directory + "/analysis_dr_3d.csv"

    # output
    aligned_trajectory_path = data_set_directory + "/analysis_aligned_trajectory.csv"
    aligned_associations_path = data_set_directory + "/analysis_aligned_associations.csv"
    plot_trajectory_path = data_set_directory + "/analysis_ate_trajectory.pdf"
    plot_error_path = data_set_directory + "/analysis_ate_error.pdf"
    verbose = True

    # Other settings
    offset = 0.0
    scale = 1.0
    max_difference = 0.0001


    # Load data
    first_list = associate.read_file_list(first_file_path)
    second_list = associate.read_file_list(second_file_path)

    matches = associate.associate(first_list, second_list, float(offset), float(max_difference))
    if len(matches) < 2:
        sys.exit(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix(
        [[float(value) * float(scale) for value in second_list[b][0:3]] for a, b in matches]).transpose()
    rot, trans, trans_error = align(second_xyz, first_xyz)

    second_xyz_aligned = rot * second_xyz + trans

    first_stamps = list(first_list.keys())  # Convert to list
    first_stamps.sort()
    first_xyz_full = numpy.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()

    second_stamps = list(second_list.keys())  # convert to list
    second_stamps.sort()
    second_xyz_full = numpy.matrix(
        [[float(value) * float(scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans

    ate_rmse = numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))

    if verbose:
        print(f"compared_pose_pairs {len(trans_error)} pairs")

        print(
            f"absolute_translational_error.rmse {ate_rmse} m")
        print(f"absolute_translational_error.mean {numpy.mean(trans_error)} m")
        print(f"absolute_translational_error.median {numpy.median(trans_error)} m")
        print(f"absolute_translational_error.std {numpy.std(trans_error)} m")
        print(f"absolute_translational_error.min {numpy.min(trans_error)} m")
        print(f"absolute_translational_error.max {numpy.max(trans_error)} m")
    else:
        print(numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)))

    if aligned_associations_path:
        file = open(aligned_associations_path, "w")
        file.write("\n".join(
            ["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (a, b), (x1, y1, z1), (x2, y2, z2) in
             zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
        file.close()

    if aligned_trajectory_path:
        file = open(aligned_trajectory_path, "w")
        file.write("\n".join(["%f " % stamp + " ".join(["%f" % d for d in line]) for stamp, line in
                              zip(second_stamps, second_xyz_full_aligned.transpose().A)]))
        file.close()

    if plot_trajectory_path:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        from matplotlib.patches import Ellipse

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(ax, second_stamps, second_xyz_full.transpose().A, '-', "red", "Original DR")
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A, '-', "green", "Final trajectory")
        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose().A, '-', "blue", "Aligned DR")

        label = "Difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A,
                                                      second_xyz_aligned.transpose().A):
            ax.plot([x1, x2], [y1, y2], '-', color="orange", label=label)
            label = ""

        ax.legend()

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f"ATE RMSE: {ate_rmse}")
        plt.savefig(plot_trajectory_path, format="pdf")
