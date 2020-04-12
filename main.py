import os
import numpy as np

train_dir = './dataset/train/'
test_dir = './dataset/test/'

training_file = 'rad_d1'
test_file = 'rad_d1.t'

M = 10
N = 8

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


def vectors_generator(output_file_name, folder_name, joints):
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
                        d = np.histogram(distances[i], bins=M, density=True)
                        a = np.histogram(angles[i], bins=N, density=True)

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


vectors_generator('rad_d1', train_dir, star_joints)
vectors_generator('rad_d1.t', test_dir, star_joints)
