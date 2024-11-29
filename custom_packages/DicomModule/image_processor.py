import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, image_list, image_flag=False):
        self.image_flag = image_flag  # Is it an image or just the path of image
        self.image_list = image_list
        self.export_extracted_contours = []

    def display_threshold_image(self, loop=False):
        win_name = 'Threshold Image'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Value High', win_name, 0, 255, self.empty)
        cv2.createTrackbar('Value Low', win_name, 0, 255, self.empty)
        cv2.setTrackbarPos('Value High', win_name, 255)
        cv2.setTrackbarPos('Value Low', win_name, 151)
        idx = 0
        while idx < len(self.image_list):
            if self.image_flag:
                img = self.image_list[idx]
            else:
                img = cv2.imread(self.image_list[idx])

            binary_img = self.updateThreshold(win_name, img)
            extracted_img = cv2.bitwise_and(img, img, mask=binary_img)
            extracted_img = self.apply_contour(img, extracted_img, -1, 10000)

            cv2.imshow(win_name, extracted_img)
            idx += 1

            if loop and (idx == len(self.image_list)):
                idx = 0

            keyPressed = cv2.waitKey(1)
            if keyPressed == 27 or keyPressed == ord('q'):
                break

            print(f"Current Printed Frame = {idx}")

        cv2.destroyAllWindows()
        return self.export_extracted_contours

    def empty(self, x):
        pass

    @staticmethod
    def updateThreshold(win_name, img):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        vValueH = cv2.getTrackbarPos('Value High', win_name)
        vValueL = cv2.getTrackbarPos('Value Low', win_name)

        upperThreshold = np.array([179, 255, vValueH])
        lowerThreshold = np.array([0, 0, vValueL])

        binary_img = cv2.inRange(hsv_image, lowerThreshold, upperThreshold)
        # binary_img = cv2.Canny(binary_img, 100, 200)

        return binary_img

    def apply_contour(self, org_img, img, total_conts, min_area_cont=10000):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_img = cv2.medianBlur(gray_img, 17)
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel, iterations=3)
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel, iterations=1)

        conts, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        max_area_pix = []
        max_id = []
        for idx, cont in enumerate(conts):
            temp_area_pix = cv2.contourArea(cont)

            if (min_area_cont <= temp_area_pix) and (hierarchy[0][idx][3] != -1):
                print(temp_area_pix)
                max_area_pix.append(temp_area_pix)
                max_id.append(idx)

        valid_contours = [conts[i] for i in max_id]

        if len(max_id):
            self.export_extracted_contours.append(valid_contours)
            cv2.drawContours(org_img, tuple(valid_contours), -1, (0, 0, 255), 2)

        return org_img

    def process_the_contour_list(self, contour_list):
        pcd_points = np.array([0, 0, 0])
        layer_res = 0.007
        pixel_res = 0.007
        for idx, layer in enumerate(contour_list):
            for cont in layer:
                temp_point = np.reshape(cont, (-1, 2))
                temp_point = temp_point * pixel_res
                layer_stack = np.ones((temp_point.shape[0], 1)) * (layer_res * idx)
                temp_point = np.hstack((temp_point, layer_stack))
                pcd_points = np.append(pcd_points, temp_point.flatten())

        return pcd_points
