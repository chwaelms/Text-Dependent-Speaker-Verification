#!/usr/bin/env python
# encoding: utf-8

import argparse
import os

import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str,
                        default="datasets/VoxCeleb/voxceleb1/")
    parser.add_argument('--src_trials_path', help='src_trials_path',
                        type=str, default="voxceleb1_test_v2.txt")
    parser.add_argument('--dst_trials_path', help='dst_trials_path',
                        type=str, default="data/trial.lst")
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trials_path, dtype=str)

    # ==========================
    #   제공된 경로 생성 코드
    # ==========================
    # with open(args.dst_trials_path, "w") as f:
    #     for item in trials:
    #         enroll_path = os.path.join(
    #             args.voxceleb1_root, "wav", item[1])
    #         test_path = os.path.join(args.voxceleb1_root, "wav", item[2])
    #         f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

    # ==========================
    #  채널 지정 경로 생성 코드
    # ==========================    
    # "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"
    # "00", "04", "08", "12", "15"
    with open(args.dst_trials_path, "w") as f:
        for item in trials:
            # 중첩 for 루프로 모든 조합 생성
            for enroll_replacement in ["00", "02", "04", "06", "08", "10", "12", "14"]:
                # 현재 enroll 채널에 대한 경로 생성
                enroll_path = os.path.join(
                    args.voxceleb1_root,
                    item[0].replace("{}", enroll_replacement)
                )
                
                for test_replacement in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]:
                    # 현재 test 채널에 대한 경로 생성
                    test_path = os.path.join(
                        args.voxceleb1_root,
                        item[1].replace("{}", test_replacement)
                    )
                    # 현재 enroll/test 조합에 대해 파일에 쓰기
                    target = 1 if item[2]=="target" else 0
                    f.write("{} {} {}\n".format(target, enroll_path, test_path))