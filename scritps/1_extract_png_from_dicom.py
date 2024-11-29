import pydicom as dcm
import glob
import numpy as np
import cv2
import os

class DicomHandler:
    def __init__(self, file_path, result_path=None):
        self.file_path = file_path
        self.result_path = result_path
        self.win_name = 'Threshold Display'

        self.image_name_list = []
        self.png_image_name_list = []
        self.count = 1

        self.image_name_list = self.load_images_name()

    def load_images_name(self):
        image_name_list = [file for file in sorted(glob.glob(self.file_path), key=self.sort_func)]
        return image_name_list

    def load_png_images_name(self):
        png_image_path = self.result_path + "*png"
        image_name_list = [file for file in sorted(glob.glob(png_image_path), key=self.sort_func)]
        return image_name_list

    def save_dicom_as_png(self, prefix_name="WPI"):

        for image_name in self.image_name_list:
            image = dcm.dcmread(image_name).pixel_array
            image = (int(255) * ((image.astype(int)) - int(image.min())) / (int(image.max()) - int(image.min()))).astype(np.uint8)
            image = cv2.merge((image, image, image))
            self.count = os.path.splitext(os.path.basename(image_name))[0][-5:]
            print(f"Image Read: {self.count}")
            cv2.imwrite(self.result_path + f"{prefix_name}_{self.count}.png", image)
            self.png_image_name_list.append(self.result_path + f"{prefix_name}_{self.count}.png")

    def sort_func(self, s):
        return int(os.path.splitext(os.path.basename(s))[0][-5:])

    def display_image(self, loop=False):
        if not self.png_image_name_list:
            self.png_image_name_list = self.load_png_images_name()

        idx = 0
        print(f"Total number of scan layers = {len(self.png_image_name_list)}")
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        while idx < len(self.png_image_name_list):
            key_pressed = cv2.waitKey(1)
            if key_pressed == 27:
                break

            original_img = cv2.imread(self.png_image_name_list[idx])
            cv2.putText(original_img, f'Frame: {idx}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10 )
            cv2.imshow(self.win_name, original_img)
            print(f"Displayed Image = {os.path.splitext(os.path.basename(self.png_image_name_list[idx]))[0][0:]}")
            idx += 1

            if loop and idx == len(self.image_name_list):
                idx = 0

        cv2.destroyAllWindows()


if __name__ == '__main__':
    dicom_file_path = "./*DCM"
    png_file_path = "./"

    dicom_handler_object = DicomHandler(dicom_file_path, png_file_path)
    # Extract and save scans in PNG format from DCM
    dicom_handler_object.save_dicom_as_png(prefix_name='CT')  # Comment out if you do not want to save/overwrite
    # Display the extracted PNG Images
    dicom_handler_object.display_image()