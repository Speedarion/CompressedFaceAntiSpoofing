# Rizhao Version
import os
import sys
import time
import zipfile
import io

import cv2
import dlib
from glob import glob
import zip_helper
import copy_im_with_roi
from multiprocessing import Pool
# switch the SPLIT_SVM/SVM/CNN detector
import sys

ext_ratio = 0.0
DETECTOR = "SPLIT_SVM"

DATASET_DIR = {
    'OULU-NPU': '/home/Dataset/OULU-NPU/*/*.avi',  #
    'CASIA-FASD': '/home/Dataset/CASIA-FASD/*/*/*.avi',  #
    'REPLAY-ATTACK': '/home/Dataset/REPLAY-ATTACK/*/*/*.mov',  #
    'REPLAY-ATTACK-SPOOF': '/home/Dataset/REPLAY-ATTACK/*/*/*/*.mov',  #
    'SIW': '/home/Dataset/SIW/*/*/*/*.mov',
    'ROSE-CASIA-YOUTU': '/home/Dataset/ROSE-CASIA-YOUTU/*/*/*.mp4',  #
    'MSU-MFSD': '/home/Dataset/MSU-MFSD/scene01/*/*.mp4',
    'MSU-MFSD2': '/home/Dataset/MSU-MFSD/scene01/*/*.mov',

}
DATASET_DIR_ROOT = '/home/rizhao/data/FAS/'


def process_one_video(input_fn):
    # get the input_fn ext_ratio

    output_fn = os.path.relpath(input_fn, '/home/Dataset')
    output_fn = os.path.join(DATASET_DIR_ROOT, "all_public_datasets_zip/EXT{}/+".format(ext_ratio) + output_fn + ".zip")
    output_folder_dir = os.path.dirname(output_fn)
    # import pdb; pdb.set_trace()
    print('input_fn: ', input_fn)
    print("output_fn: ", output_fn)

    # skip if output_fn exists
    if os.path.exists(output_fn):
        print("output_fn exists, skip")
        return
    elif not os.path.exists(output_folder_dir):
        print('Create ', output_folder_dir)
        os.makedirs(output_folder_dir, exist_ok=True)

    # init dlib
    if DETECTOR == "SPLIT_SVM":
        det = dlib.fhog_object_detector("part0.svm")
    elif DETECTOR == "SVM":
        det = dlib.get_frontal_face_detector()
    elif DETECTOR == "CNN":
        det = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    else:
        assert False

    # init VideoCapture
    cap = cv2.VideoCapture(input_fn)

    # get frame
    face_count = 0
    frame_count = 0
    bbox_string = ''
    with io.BytesIO() as bio:
        with zipfile.ZipFile(bio, "w") as zip:
            # write pngs to zip in memory
            for frame_idx in range(1000000000):
                # for i in range(5):
                #    ret, im = cap.read()
                ret, im = cap.read()
                if not ret:
                    print("video ends")
                    assert im is None
                    break

                # generate smaller im2 to detected in
                im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                rescale = min(1.0, max(640.0 / im.shape[0], 640.0 / im.shape[
                    1]))  # shrink with low limitation of 640px * 640px. never expand
                if rescale != 1.0:
                    im2 = cv2.resize(im2, None, fx=rescale, fy=rescale, interpolation=cv2.INTER_NEAREST)

                # detect
                # t = time.clock()
                if DETECTOR == "SPLIT_SVM" or DETECTOR == "SVM":
                    rect, _, _ = det.run(im2, 0, 0.0)
                elif DETECTOR == "CNN":
                    rect = det(im2, 1)
                else:
                    assert False
                # print("detected faces:", len(rect), "t:", time.clock() - t)

                # only keep the first box
                frame_count += 1
                if len(rect) == 0:
                    continue
                if DETECTOR == "SPLIT_SVM" or DETECTOR == "SVM":
                    rect = rect[0]
                elif DETECTOR == "CNN":
                    rect = rect[0].rect
                else:
                    assert False
                face_count += 1

                # rescale the bounding box
                t = rect.top() / rescale
                b = rect.bottom() / rescale
                l = rect.left() / rescale
                r = rect.right() / rescale

                bbox_string += "%d,%d,%d,%d\n" % (round(t), round(l), round(b), round(r))
                # extend the bounding box
                h = b - t
                w = r - l
                t -= h * ext_ratio
                b += h * ext_ratio
                l -= w * ext_ratio
                r += w * ext_ratio

                # crop
                crop = copy_im_with_roi.copy_im_with_roi(im, round(t), round(b), round(l), round(r))

                # resize
                crop = cv2.resize(crop, (300, 300))

                # save crop
                zip_helper.write_im_to_zip(zip, str(frame_idx) + ".png", crop)

            # write info.txt to zip in memory
            print(face_count, frame_count)
            zip_helper.write_bytes_to_zip(zip, "bbox.txt", bytes(bbox_string, "utf-8"), )
            zip_helper.write_bytes_to_zip(zip, "info.txt", bytes("%d\t%d" % (face_count, frame_count), "utf-8"))

        # finally, flush bio to disk once
        path = os.path.dirname(output_fn)
        if path != "":
            os.makedirs(path, exist_ok=True)
        with open(output_fn, "wb") as f:
            f.write(bio.getvalue())

    cap.release()


def main():
    pool = Pool(16)

    for key, matching_pattern in DATASET_DIR.items():
        print(key, matching_pattern)
        video_fns = glob(matching_pattern)
        pool.map(process_one_video, video_fns)


if __name__ == "__main__":
    main()
    # preprocess()
