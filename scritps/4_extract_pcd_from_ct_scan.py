import os
import cv2
import glob
import numpy as np



class ExtractContour:
    def __init__(self, image_path, mask_path=None):
        self.image_path = image_path
        self.image_file_list = self.load_image_file_list()
        self.colors = self.get_color_list(10)
        self.mask = None
        if mask_path:
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def get_color_list(num):
        np.random.seed(10)
        color = [tuple(np.random.random(3) * 256) for i in range(10)]
        return color
    @staticmethod
    def extract_3d_points_from_2d_contours(conts, valid_cont_ids, uv_resolution, layer_height):
        all_pcd_points = np.zeros((1, 1, 2))
        for idx in valid_cont_ids:
            all_pcd_points = np.vstack((all_pcd_points, conts[idx] * uv_resolution))

        total_points = all_pcd_points.shape[0]
        layer_height_array = np.ones((total_points, 1, 1)) * layer_height
        all_pcd_points = np.dstack((all_pcd_points, layer_height_array))
        reshaped_pcd = all_pcd_points.reshape((1, -1, 3))

        return reshaped_pcd[:, 1:, :]

    def load_image_file_list(self):
        image_file_list = [file for file in sorted(glob.glob(self.image_path),
                                                     key=lambda s: int(os.path.splitext(os.path.basename(s))[0][-4:]))]
        return image_file_list

    def display_loaded_images(self):
        main_window = 'Display Image'
        blank_image = np.zeros((800, 800), dtype=np.uint8)

        cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
        cv2.imshow(main_window, blank_image)
        k = cv2.waitKey(10)
        total_images = len(self.image_file_list)

        for idx, image_name in enumerate(self.image_file_list):
            if k != ord('q'):
                curr_frame = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                cv2.imshow(main_window, curr_frame)
                print(f"Displaying {idx}/{total_images} : {image_name}")
                k = cv2.waitKey(1)
            elif k == ord('q'):
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        print("All images Displayed")

    def extract_contours_from_ct_images(self, threshold_range=(179, 255), minimum_contour_area=3000, scan_resolution=(7, 7)):
        contour_win = 'Contour Extracted'
        cv2.namedWindow(contour_win, cv2.WINDOW_NORMAL)
        cv2.imshow(contour_win, np.zeros((800, 800), dtype=np.uint8))
        keyPressed = cv2.waitKey(1)

        lower_boundary = np.array([0, 0, threshold_range[0]])
        upper_boundary = np.array([185, 255, threshold_range[1]])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        layer_height, pcd = 0, []
        for idx, image_name in enumerate(self.image_file_list):
            curr_frame = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)

            hsv_img = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
            binary  = cv2.inRange(hsv_img, lower_boundary, upper_boundary)
            extracted_img = cv2.bitwise_and(curr_frame, curr_frame, mask=binary)
            extracted_img = cv2.bitwise_and(extracted_img, extracted_img, mask=self.mask)

            morphed_img = cv2.medianBlur(extracted_img[:, :, 0], 25)  # 25
            morphed_img = cv2.morphologyEx(morphed_img, cv2.MORPH_CLOSE, kernel, iterations=3)
            morphed_img = cv2.morphologyEx(morphed_img, cv2.MORPH_DILATE, kernel, iterations=1)

            conts, hierarchy = cv2.findContours(morphed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            valid_cont_ids = []
            colors = iter(self.colors)
            for cont_id, cont in enumerate(conts):
                cont_area = cv2.contourArea(cont)
                if minimum_contour_area <= cont_area and (hierarchy[0][cont_id][3] != -1):
                    valid_cont_ids.append(cont_id)
                    cv2.drawContours(curr_frame, [cont], 0, next(colors), 5)

            layer_pcd = self.extract_3d_points_from_2d_contours(conts, valid_cont_ids, scan_resolution[0], layer_height)
            pcd = pcd + layer_pcd.tolist()[0]
            layer_height += scan_resolution[1]

            cv2.imshow(contour_win, curr_frame)
            keyPressed = cv2.waitKey(1)
            if keyPressed == 27 or keyPressed == ord('q'):
                break

            print(f"Current Printed Frame = {idx} of Shape {curr_frame.shape}")

        cv2.destroyAllWindows()
        return pcd


if __name__ == '__main__':

    image_path = "./*.png"
    mask_path  = "Mask/*.png"
    result_path = ".PCD/"
    contour_handler = ExtractContour(image_path, mask_path)

    # Optional
    contour_handler.image_file_list = contour_handler.image_file_list[50:]

    print(f"Total number of images: {len(contour_handler.image_file_list)}")

    pcd = contour_handler.extract_contours_from_ct_images(threshold_range= (155, 255), minimum_contour_area=5000, scan_resolution=(7,7))
    np.savetxt(result_path, np.array(pcd), delimiter=',')