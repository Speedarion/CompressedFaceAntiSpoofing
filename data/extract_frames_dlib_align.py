# Rizhao Version
import os
import sys
import time
import zipfile
import io
import numpy as np
import cv2
import dlib
from glob import glob
import zip_helper

from multiprocessing import Pool
# switch the SPLIT_SVM/SVM/CNN detector
from facenet_pytorch import MTCNN # https://github.com/timesler/facenet-pytorch
import sys
from PIL import Image

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

def _clamp(t1, t2, m, M):
    if t1 < m:
        t2 = t2 - t1 + m
        t1 = t1 - t1 + m
    if t1 > M:
        t2 = t2 - t1 + M
        t1 = t1 - t1 + M

    return t1, t2


def copy_im_with_roi(im1, t1, b1, l1, r1):
    t2 = 0
    b2 = b1 - t1
    l2 = 0
    r2 = r1 - l1

    if len(im1.shape) == 2:
        im2 = np.zeros([b2, r2], im1.dtype)
    elif len(im1.shape) == 3:
        assert (im1.shape[2] == 3)
        im2 = np.zeros([b2, r2, 3], im1.dtype)
    else:
        assert False

    t1, t2 = _clamp(t1, t2, 0, im1.shape[0] - 1)
    b1, b2 = _clamp(b1, b2, 0, im1.shape[0] - 1)
    l1, l2 = _clamp(l1, l2, 0, im1.shape[1] - 1)
    r1, r2 = _clamp(r1, r2, 0, im1.shape[1] - 1)

    im2[t2:b2, l2:r2] = im1[t1:b1, l1:r1]

    return im2

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth


    def align(self, image, pos):
        leftEyeCenter = pos[1]
        rightEyeCenter = pos[0]

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


def process_one_video(input_fn):
    # get the input_fn ext_ratio
    mtcnn = MTCNN()
    output_fn = os.path.relpath(input_fn, '/home/Dataset')
    output_fn = os.path.join(DATASET_DIR_ROOT, "MTCNN/align/+".format(ext_ratio) + output_fn + ".zip")
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
    """
    if DETECTOR == "SPLIT_SVM":
        det = dlib.fhog_object_detector("part0.svm")
    elif DETECTOR == "SVM":
        det = dlib.get_frontal_face_detector()
    elif DETECTOR == "CNN":
        det = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    else:
        assert False
    """

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
                im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                rescale = min(1.0, max(640.0 / im.shape[0], 640.0 / im.shape[
                    1]))  # shrink with low limitation of 640px * 640px. never expand
                if rescale != 1.0:
                    im2 = cv2.resize(im2, None, fx=rescale, fy=rescale, interpolation=cv2.INTER_NEAREST)

                # detect
                # t = time.clock()

                import pdb;
                pdb.set_trace()
                pil_img = Image.fromarray(im2)
                boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)
                rect = boxes
                # print("detected faces:", len(rect), "t:", time.clock() - t)

                # only keep the first box

                """
                if len(rect) == 0:
                    continue
                if DETECTOR == "SPLIT_SVM" or DETECTOR == "SVM":
                    rect = rect[0]
                elif DETECTOR == "CNN":
                    rect = rect[0].rect
                else:
                    assert False
                face_count += 1
                """
                frame_count += 1

                # rescale the bounding box
                #t = rect.top() / rescale
                #b = rect.bottom() / rescale
                #l = rect.left() / rescale
                #r = rect.right() / rescale

                # 1, 3 y1, y2. 0,2 x1, x2
                t = rect[0][1] / rescale
                b = rect[0][3] / rescale
                l = rect[0][0] / rescale
                r = rect[0][2] / rescale

                bbox_string += "%d,%d,%d,%d\n" % (round(t), round(l), round(b), round(r))
                # extend the bounding box
                h = b - t
                w = r - l
                t -= h * ext_ratio
                b += h * ext_ratio
                l -= w * ext_ratio
                r += w * ext_ratio

                # crop
                crop = copy_im_with_roi(im, int(round(t)), int(round(b)), int(round(l)), int(round(r)))

                # resize
                crop = cv2.resize(crop, (256, 256))

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

def single_test_case():

    process_one_video('/home/Dataset/CASIA-FASD/train_release/1/1.avi')

def main():

    mtcnn = MTCNN()

    pool = Pool(16)

    for key, matching_pattern in DATASET_DIR.items():
        print(key, matching_pattern)
        video_fns = glob(matching_pattern)
        pool.map(process_one_video, video_fns)


if __name__ == "__main__":
    # main()
    # preprocess()
    single_test_case()
