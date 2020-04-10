import os
import numpy as np

train_dir = './dataset/train'
test_dir = './dataset/test'

training_file = 'rad_d1'
test_file = 'rad_d1.t'


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


# for filename in os.listdir(train_dir):
with open('./dataset/train/a08_s01_e01_skeleton_proj.txt') as file:
    lines = file.readlines()

    frames = list(divide_chunks(lines, 20))

    distances = []
    angles = []

    for frame in frames:
        #frame[1, 4, 8, 12, 16, 20]
        hip_pos = np.array([float(number)
                            for number in frame[0].split()][2:])
        head_pos = np.array([float(number)
                             for number in frame[3].split()][2:])
        left_hand_pos = np.array([float(number)
                                  for number in frame[7].split()][2:])
        right_hand_pos = np.array([float(number)
                                   for number in frame[11].split()][2:])
        left_foot_pos = np.array([float(number)
                                  for number in frame[15].split()][2:])
        right_foot_pos = np.array([float(number)
                                   for number in frame[19].split()][2:])

        d = {}
        d['head to hip'] = distance_finder(hip_pos, head_pos)
        d['l hand to hip'] = distance_finder(hip_pos, left_hand_pos)
        d['r hand to hip'] = distance_finder(hip_pos, right_hand_pos)
        d['l foot to hip'] = distance_finder(hip_pos, left_foot_pos)
        d['r foot to hip'] = distance_finder(hip_pos, right_foot_pos)

        a = {}
        a['lhand_hip_head'] = angle_finder(
            left_hand_pos, hip_pos, head_pos)
        a['head_hip_rhand'] = angle_finder(
            head_pos, hip_pos, right_hand_pos)
        a['rhand_hip_rfoot'] = angle_finder(
            right_hand_pos, hip_pos, right_foot_pos)
        a['rfoot_hip_lfoot'] = angle_finder(
            right_foot_pos, hip_pos, left_foot_pos)
        a['lfoot_hip_lhand'] = angle_finder(
            left_foot_pos, hip_pos, left_hand_pos)
