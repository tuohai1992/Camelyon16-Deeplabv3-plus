from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image, ImageStat
import glob
import os
import numpy as np
import cv2
import pdb
from PIL import Image, ImageStat, ImageDraw, ImageFont
import multiprocessing
import tensorflow as tf

num_threads = 8
tf_coord = tf.train.Coordinator()

PATCH_SIZE = 768

# modify below directory entries as per your local file system
"/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor"

TRAIN_TUMOR_WSI_PATH = '/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor'
TRAIN_NORMAL_WSI_PATH = '/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Normal'
TRAIN_TUMOR_MASK_PATH = '/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask'
PROCESSED_PATCHES_POSITIVE_PATH = '/home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_2_patch/patch_768_came16/image_neg/'
PROCESSED_PATCHES_POSITIVE_MASK_PATH = '/home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_2_patch/patch_768_came16/mask_neg/'
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'


class WSI(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """
    index = 0
    negative_patch_index = 0
    positive_patch_index = 0
    wsi_paths = []
    mask_paths = []
    def_level = 7
    level_extract = 2
    key = 0

    def extract_patches_mask(self, bounding_boxes):
        """
        Extract positive patches targeting annotated tumor region

        Save extracted patches to desk as .png image files

        :param bounding_boxes: list of bounding boxes corresponds to tumor regions
        :return:

        """
        mag_factor = pow(2, self.level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
            # X = np.arange(b_x_start, b_x_end-256, 5)
            # Y = np.arange(b_y_start, b_y_end-256, 5)

            for x, y in zip(X, Y):
                mask = self.mask_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                mask_gt = np.array(mask)
                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)

                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)

                if white_pixel_cnt_gt > ((PATCH_SIZE * PATCH_SIZE) * 0.90):
                    # mask = Image.fromarray(mask)
                    patch = self.wsi_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                    patch.save(
                        PROCESSED_PATCHES_FROM_USE_MASK_POSITIVE_PATH + PATCH_TUMOR_PREFIX + str(PATCH_SIZE) + '_' +
                        str(self.positive_patch_index), 'PNG')
                    self.positive_patch_index += 1
                    patch.close()

                mask.close()

    def extract_patches_normal(self, bounding_boxes):
        """
            Extract negative patches from Normal WSIs

            Save extracted patches to desk as .png image files

            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :return:

        """
        mag_factor = pow(2, self.level_used)
        mag_factor2 = pow(2, self.level_extract)
        bbx_co = []
        bby_co = []

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if (bounding_box[2] * bounding_box[3]) < 2500:
                continue
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor

            # X = np.arange(b_x_start, b_x_end-256, 5)
            # Y = np.arange(b_y_start, b_y_end-256, 5)
            h = int(bounding_box[2]) * mag_factor
            w = int(bounding_box[3]) * mag_factor
            print("Size of bounding box = %s" % h + " by %s" % w)

            tumoridx = 0
            ntumoridx = 0
            patchidx = 0
            for y_left in range(b_y_start, b_y_end, PATCH_SIZE* mag_factor2):

                bby_co.append(range(y_left, y_left + PATCH_SIZE* mag_factor2))

                for x_left in range(b_x_start, b_x_end, PATCH_SIZE* mag_factor2):

                    if x_left in bbx_co and y_left in bby_co:
                        print("Skipping double Bounding Box %s " %self.cur_wsi_path)
                        continue

                    bbx_co.append(range(x_left,x_left+PATCH_SIZE* mag_factor2))


                    patch = self.wsi_image.read_region((x_left, y_left), 2, (PATCH_SIZE, PATCH_SIZE))
                    mask = np.zeros((PATCH_SIZE,PATCH_SIZE),dtype=np.uint8)
                    mask=Image.fromarray(mask)
                    # mask = self.mask_image.read_region((x_left, y_left), 2, (PATCH_SIZE, PATCH_SIZE))

                    _std = ImageStat.Stat(patch).stddev
                    patch_array = np.array(patch)
                    # thresholding stddev for patch extraction
                    patchidx += 1
                    if (sum(_std[:3]) / len(_std[:3])) < 15:
                        continue

                    patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                    # [20, 20, 20]
                    lower_red = np.array([0, 0, 0])
                    # [255, 255, 255]
                    upper_red = np.array([200, 200, 200])
                    mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                    white_pixel_cnt = cv2.countNonZero(mask_patch)

                    if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.04):
                        # mask = Image.fromarray(mask)
                        if self.negative_patch_index % 2 == 0:
                            patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_NORMAL_PREFIX + str(PATCH_SIZE) + '_' +
                                    str(self.negative_patch_index)+ '.png', 'PNG')
                            mask.save(PROCESSED_PATCHES_POSITIVE_MASK_PATH + 'mask_' + PATCH_NORMAL_PREFIX + str(
                                PATCH_SIZE) + '_' + str(self.negative_patch_index)+ '.png',
                                    'PNG')
                        self.negative_patch_index += 1
                        ntumoridx += 1

                    patch.close()

            print("Processed patches in bounding box %s of %s :"%(i,self.cur_wsi_path), "%s" % patchidx, " negative: %s" % ntumoridx)

    def extract_patches_tumor(self, bounding_boxes):
        """
            Extract both, negative patches from Normal area and positive patches from Tumor area

            Save extracted patches to desk as .png image files

            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :return:

        """
        mag_factor = pow(2, self.level_used)
        mag_factor2 = pow(2, self.level_extract)
        bbx_co = []
        bby_co = []
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
        for i, bounding_box in enumerate(bounding_boxes):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if (bounding_box[2] * bounding_box[3]) < 2500:
                continue
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor


            # pdb.set_trace()
            h = int(bounding_box[2]) * mag_factor
            w = int(bounding_box[3]) * mag_factor
            print("Size of bounding box = %s" % h + " by %s" % w)

            tumoridx = 0
            ntumoridx = 0
            patchidx = 0
            for y_left in range(b_y_start, b_y_end, PATCH_SIZE* mag_factor2):

                bby_co.append(range(y_left, y_left + PATCH_SIZE* mag_factor2))

                for x_left in range(b_x_start, b_x_end, PATCH_SIZE* mag_factor2):

                    if x_left in bbx_co and y_left in bby_co:
                        print("Skipping double Bounding Box %s " %self.cur_wsi_path)
                        continue

                    bbx_co.append(range(x_left,x_left+PATCH_SIZE* mag_factor2))


                    patch = self.wsi_image.read_region((x_left, y_left), 2, (PATCH_SIZE, PATCH_SIZE))
                    mask = self.mask_image.read_region((x_left, y_left), 2, (PATCH_SIZE, PATCH_SIZE))
                    # pdb.set_trace()
                    _std = ImageStat.Stat(patch).stddev
                    # thresholding stddev for patch extraction
                    patchidx += 1
                    if (sum(_std[:3]) / len(_std[:3])) < 15:
                        continue

                    mask_gt = np.array(mask)
                    # mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                    mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                    patch_array = np.array(patch)

                    white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
                    if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
                        pass
                    #     patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                    #     lower_red = np.array([0, 0, 0])
                    #     upper_red = np.array([200, 200, 220])
                    #     mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                    #     white_pixel_cnt = cv2.countNonZero(mask_patch)

                    #     if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
                    #         # mask = Image.fromarray(mask)
                    #         if self.negative_patch_index % 4 == 0:
                    #             patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_NORMAL_PREFIX + str(PATCH_SIZE) + '_' +
                    #                     str(self.negative_patch_index)+ '.png', 'PNG')
                    #             mask.save(PROCESSED_PATCHES_POSITIVE_MASK_PATH + 'mask_' + PATCH_NORMAL_PREFIX + str(PATCH_SIZE) + '_' + 
                    #             str(self.negative_patch_index)+ '.png','PNG')
                    #         self.negative_patch_index += 1
                    #         ntumoridx += 1
                    else:  # mask_gt contains tumor area
                        if white_pixel_cnt_gt >= ((PATCH_SIZE * PATCH_SIZE) * 0.005):
                            patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                            lower_red = np.array([0, 0, 0])
                            upper_red = np.array([200, 200, 220])
                            mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                            white_pixel_cnt = cv2.countNonZero(mask_patch)
                            # if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.10):
                                # if self.positive_patch_index % 2 == 0:
                            patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_TUMOR_PREFIX + str(PATCH_SIZE) + '_' +
                                    str(self.positive_patch_index)+ '.png', 'PNG')
                            mask.save(
                                PROCESSED_PATCHES_POSITIVE_MASK_PATH + 'mask_' + PATCH_TUMOR_PREFIX + str(PATCH_SIZE) + '_' +
                                str(self.positive_patch_index)+ '.png', 'PNG')

                            self.positive_patch_index += 1
                            tumoridx += 1

                    patch.close()
                    mask.close()

            print("Processed patches in bounding box %s " % i, "%s" % patchidx, " positive: %s " % tumoridx,
                  " negative: %s" % ntumoridx)


    def read_wsi_mask(self, wsi_path, mask_path):
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)

            self.mask_pil = self.mask_image.read_region((0, 0), self.level_used,
                                                        self.mask_image.level_dimensions[self.level_used])
            self.mask = np.array(self.mask_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def read_wsi_normal(self, wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.level_used = min(self.def_level, self.wsi_image.level_count - 1)

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used,
                                                            self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def read_wsi_tumor(self, wsi_path, mask_path):
        """
            # =====================================================================================
     i       # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)
            # print(self.level_used)
            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used,
                                                            self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def find_roi_n_extract_patches_mask(self):
        mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        contour_mask, bounding_boxes = self.get_image_contours_mask(np.array(mask), np.array(self.mask))

        # contour_mask = cv2.resize(contour_mask, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_mask', np.array(contour_mask))
        self.mask_pil.close()
        self.extract_patches_mask(bounding_boxes)
        self.wsi_image.close()
        self.mask_image.close()

    def find_roi_n_extract_patches_normal(self):
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([20, 50, 20])
        # [255, 255, 255]
        upper_red = np.array([200, 150, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # (50, 50)
        close_kernel = np.ones((25, 25), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        contour_rgb, bounding_boxes = self.get_image_contours_normal(np.array(image_open), self.rgb_image)

        # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_rgb', np.array(contour_rgb))
        self.rgb_image_pil.close()
        self.extract_patches_normal(bounding_boxes)
        self.wsi_image.close()

    def find_roi_n_extract_patches_tumor(self):
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        contour_rgb, bounding_boxes = self.get_image_contours_tumor(np.array(image_open), self.rgb_image)
        # pdb.set_trace()
        # Image.fromarray(np.array(contour_rgb)).show()
        # pdb.set_trace()
        # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_rgb', np.array(contour_rgb))
        self.rgb_image_pil.close()
        self.extract_patches_tumor(bounding_boxes)
        self.wsi_image.close()
        self.mask_image.close()

    @staticmethod
    def get_image_contours_mask(cont_img, mask_img):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours_mask_image_array = np.array(mask_img)
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_mask_image_array, contours, -1, line_color, 1)
        return contours_mask_image_array, bounding_boxes

    @staticmethod
    def get_image_contours_normal(cont_img, rgb_image):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours_rgb_image_array = np.array(rgb_image)
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        return contours_rgb_image_array, bounding_boxes

    @staticmethod
    def get_image_contours_tumor(cont_img, rgb_image):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours_rgb_image_array = np.array(rgb_image)

        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, bounding_boxes

    def wait(self):
        self.key = cv2.waitKey(0) & 0xFF
        print('key: %d' % self.key)

        if self.key == 27:  # escape
            return False
        elif self.key == 81:  # <- (prev)
            self.index -= 1
            if self.index < 0:
                self.index = len(self.wsi_paths) - 1
        elif self.key == 83:  # -> (next)
            self.index += 1
            if self.index >= len(self.wsi_paths):
                self.index = 0

        return True


def run_on_mask_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(TRAIN_TUMOR_MASK_PATH, '*.tif'))
    wsi.mask_paths.sort()

    wsi.index = 0
    # for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths):
    #     if wsi.read_wsi_mask(wsi_path, mask_path):
    #         wsi.find_roi_n_extract_patches_mask()

    assert len(wsi.wsi_paths) == len(wsi.mask_paths), "Not all images have masks"

    for g in range(0, len(wsi.wsi_paths), num_threads):
        p = []
        for wsi_path, mask_path in zip(wsi.wsi_paths[g:g + num_threads], wsi.mask_paths[g:g + num_threads]):
            if wsi.read_wsi_mask(wsi_path, mask_path):
                print("Processing (run_on_mask_data)", wsi_path)
                pp = multiprocessing.Process(target=wsi.find_roi_n_extract_patches_mask)
                pp.start()
                p.append(pp)
        [pp.join() for pp in p]
    # while True:
    #     wsi_path = ops.wsi_paths[ops.index]
    #     mask_path = ops.mask_paths[ops.index]
    #     if ops.read_wsi_mask(wsi_path, mask_path):
    #         ops.find_roi_n_extract_patches_mask()
    #         if not ops.wait():
    #             break
    #     else:
    #         if ops.key == 81:
    #             ops.index -= 1
    #             if ops.index < 0:
    #                 ops.index = len(ops.wsi_paths) - 1
    #         elif ops.key == 83:
    #             ops.index += 1
    #             if ops.index >= len(ops.wsi_paths):
    #                 ops.index = 0


def run_on_tumor_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(TRAIN_TUMOR_MASK_PATH, '*.tif'))
    wsi.mask_paths.sort()
    # pdb.set_trace()
    wsi.index = 0

    # Parallel(n_jobs=8)(delayed(wsi.find_roi_n_extract_patches_tumor)() for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths) if wsi.read_wsi_tumor(wsi_path, mask_path))
    assert len(wsi.wsi_paths) == len(wsi.mask_paths), "Not all images have masks"

    # Non - parallel
    for wsi_path, mask_path in zip(wsi.wsi_paths[0:109], wsi.mask_paths[0:109]):
        # pdb.set_trace()
        wsi.read_wsi_tumor(wsi_path, mask_path)
        if wsi.read_wsi_tumor(wsi_path, mask_path):
            print("Processing (run_on_mask_data)", wsi_path)
            wsi.find_roi_n_extract_patches_tumor()

    # for g in range(0, len(wsi.wsi_paths), num_threads):
    #     p = []
    #     for wsi_path, mask_path in zip(wsi.wsi_paths[g:g + num_threads], wsi.mask_paths[g:g + num_threads]):
    #         if wsi.read_wsi_tumor(wsi_path, mask_path):
    #             print("Processing (run_on_tumor_data)", wsi_path)
    #             pp = multiprocessing.Process(target=wsi.find_roi_n_extract_patches_tumor)
    #             p.append(pp)
    #             pp.start()
    #     [pp.join() for pp in p]

    # while True:
    #     wsi_path = ops.wsi_paths[ops.index]
    #     mask_path = ops.mask_paths[ops.index]
    #     print(wsi_path)
    #     print(mask_path)
    #     if ops.read_wsi_tumor(wsi_path, mask_path):
    #         ops.find_roi_n_extract_patches_tumor()
    #         if not ops.wait():
    #             break
    #     else:
    #         if ops.key == 81:
    #             ops.index -= 1
    #             if ops.index < 0:
    #                 ops.index = len(ops.wsi_paths) - 1
    #         elif ops.key == 83:
    #             ops.index += 1
    #             if ops.index >= len(ops.wsi_paths):
    #                 ops.index = 0


def run_on_normal_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_NORMAL_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()

    # ops.wsi_paths = ops.wsi_paths[:1]
    wsi.index = 0

    for wsi_path in wsi.wsi_paths[0:143]:
        # pdb.set_trace()
        if wsi.read_wsi_normal(wsi_path):
            print("Processing (run_on_normal_data)", wsi_path)
            wsi.find_roi_n_extract_patches_normal()

    # for wsi_path in wsi.wsi_paths:
    #     if wsi.read_wsi_normal(wsi_path):
    #         wsi.find_roi_n_extract_patches_normal()

    # while True:
    #     wsi_path = ops.wsi_paths[ops.index]
    #     print(wsi_path)
    #     if ops.read_normal_wsi(wsi_path):
    #         ops.find_roi_normal()
    #         if not ops.wait():
    #             break
    #     else:
    #         if ops.key == 81:
    #             ops.index -= 1
    #             if ops.index < 0:
    #                 ops.index = len(ops.wsi_paths) - 1
    #         elif ops.key == 83:
    #             ops.index += 1
    #             if ops.index >= len(ops.wsi_paths):
    #                 ops.index = 0


if __name__ == '__main__':
    wsi = WSI()
    # run_on_tumor_data()
    run_on_normal_data()
    # run_on_mask_data()

