import os
import numpy as np

train_dir = './dataset/train/'
test_dir = './dataset/test/'

training_file = 'rad_d1'
test_file = 'rad_d1.t'

M = 20
N = 10

star_joints = [0, 3, 7, 11, 15, 19]


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


for filename in os.listdir(train_dir):
    print('Starting instance {}'.format(filename))
    with open(train_dir + filename) as file:
        lines = file.readlines()
        frames = list(divide_chunks(lines, 20))

        T = 0
        distances = []
        angles = []

        # Initializes distance and angle lists for the instance
        for i in range(5):
            distances.append([])
            angles.append([])

        for frame in frames:
            T += 1
            pos = []
            for joint in star_joints:
                # Find positions of joints
                pos.append(np.array([float(number)
                                     for number in frame[joint].split()][2:]))

            # Computes distances and angles
            for i in range(5):
                distances[i].append(distance_finder(pos[0], pos[i]))
                angles[i].append(angle_finder(
                    pos[i + 1], pos[0], pos[(i + 2) % 5 + 1]))

        distance_hist = []
        angle_hist = []

        # Creates normalized histograms for the instance
        try:
            for i in range(5):
                distance_hist.append(np.histogram(
                    distances[i], bins=M, density=True))
                angle_hist.append(np.histogram(
                    angles[i], bins=N, density=True))
        except Exception:
            print('Skipped frame {} from file {}'.format(T, filename))
            pass
