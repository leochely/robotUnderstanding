import os
import numpy as np
from libsvm.svmutil import *

train_dir = './dataset/train/'
test_dir = './dataset/test/'

training_file = 'rad_d1'
test_file = 'rad_d1.t'

N = 15
M = 20

star_joints = [0, 3, 11, 19, 15, 7]
custom_joints = [0, 5, 7, 9, 11, 19]


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def distance_finder(one, two):
    dist = one - two
    return np.linalg.norm(dist)


def angle_finder(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


def generate_file(output_file_name, folder_name, joints):
    with open(output_file_name, 'w') as output_file:
        for filename in os.listdir(folder_name):
            output_file.write(filename + ' ')
            print('Starting instance {}'.format(filename))
            with open(folder_name + filename) as file:
                lines = file.readlines()
                frames = list(divide_chunks(lines, 20))

                T = 0
                distances = []
                angles = []

                # Initializes distance and angle lists for the instance
                for i in range(len(star_joints) - 1):
                    distances.append([])
                    angles.append([])

                for frame in frames:
                    T += 1
                    pos = []
                    for joint in joints:
                        # Find positions of joints
                        pos.append(np.array([float(number)
                                             for number in frame[joint].split()][2:]))

                    # Computes distances and angles
                    for i in range(len(star_joints) - 1):
                        distances[i].append(distance_finder(pos[0], pos[i]))
                        angles[i].append(angle_finder(
                            pos[i + 1], pos[0], pos[(i + 2) % (len(joints) - 1) + 1]))

                # Creates normalized histograms for the instance
                try:
                    for i in range(len(star_joints) - 1):
                        d = np.histogram(distances[i], bins=N, density=True)
                        a = np.histogram(angles[i], bins=M, density=True)

                        # Outputs to file
                        for j in range(len(d[0])):
                            output_file.write(
                                '{}-{} {} '.format(d[1][j], d[1][j + 1], d[0][j]))
                        for j in range(len(a[0])):
                            output_file.write(
                                '{}-{} {} '.format(a[1][j], a[1][j + 1], a[0][j]))
                except Exception:
                    # Skips frame when NaN
                    print('Skipped frame {} from file {}'.format(T, filename))
                    pass
                output_file.write('\n')


def convert_to_libsvm_format(input_file, output_file):
    with open(output_file, 'w') as output:
        with open(input_file, 'r') as input:
            lines = input.readlines()
            for line in lines:
                split = line.split()
                output.write('{} '.format(int(split[0][1:3])))
                values = split[2::2]
                i = 0
                for value in values:
                    output.write('{}:{} '.format(i, value))
                    i += 1
                output.write('\n')


def learn_and_predict(train, test):
    prob = svm_read_problem(train)
    prob_test = svm_read_problem(test)
    param = svm_parameter(kernel_type=RBF, C=10)
    m = svm_model(prob, param)

    # generate_file('rad_d1', train_dir, star_joints)
    # generate_file('rad_d1.t', test_dir, star_joints)
    # generate_file('cust_d1', train_dir, custom_joints)
    # generate_file('cust_d1.t', test_dir, custom_joints)


convert_to_libsvm_format('rad_d1', 'rad_d2')
