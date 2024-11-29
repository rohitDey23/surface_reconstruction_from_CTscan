import cv2
import numpy as np
import glob
import os


scan_set_path1 = ""
scan_set_path2 = ""
# scan_set_path3 = ""
# scan_set_path4 = ""

aligned_images_file_path = ""
recorded_offsets_file_path = ""

class AligningImages:
    def __init__(self, image_path_list, offset_file_path=None):
        self.offset_success = False
        self.aligned_win = 'Aligned Images'

        self.all_image_lists = [self.load_image_name_list(image_path) for image_path in image_path_list]
        self.canvas_shape = self.determine_canvas_shape()
        self.canvas = np.zeros(self.canvas_shape, dtype=np.uint8)

        self.offsets = []
        self.offset_file_path = offset_file_path

    @staticmethod
    def load_image_name_list(file_path):
        image_path = file_path + "*png"
        image_name_list = [file for file in sorted(glob.glob(image_path),
                                                   key=lambda s: int(os.path.splitext(os.path.basename(s))[0][-5:]))]

        print(f"Total number of images: {len(image_name_list)}")
        return image_name_list

    def determine_canvas_shape(self):
        # Load single img from each img set and store it in image_list
        image_list = [cv2.imread(image_set[0]) for image_set in self.all_image_lists if image_set]

        # Determine the maximum width and height among all images to create the canvas
        max_height = max(img.shape[0] for img in image_list)
        max_width  = max(img.shape[1] for img in image_list)

        # Create a blank canvas large enough to fit all images with the same center
        canvas_height, canvas_width = max_height, max_width
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        return  canvas.shape

    def get_offset_by_manual_overlapping(self, save_offsets=False):
        # Load single img from each img set and store it in image_list
        image_list_first = [cv2.imread(image_set[0]) for image_set  in self.all_image_lists[1:] if image_set]
        image_list_last  = [cv2.imread(image_set[-1]) for image_set  in self.all_image_lists[:-1] if image_set]

        cv2.namedWindow(self.aligned_win, cv2.WINDOW_NORMAL)

        # Loop through the image sets that needs to be aligned and output the offsets for each
        for ref_image, curr_set in zip(image_list_last, image_list_first):

            x_shift, y_shift,angle, total_angle = 0, 0, 0, 0     # For each set the reset the x & y shifts
            ref_shape = ref_image.shape # Ref image changes for consecutive sets

            overlay = True

            # Based on the logic the offset shall be cumulative if considered wrt to first set.
            while True:
                keyPressed = cv2.waitKey(1)

                if keyPressed == ord('w'):
                    y_shift -= 1
                elif keyPressed == ord('s'):
                    y_shift += 1
                elif keyPressed == ord('d'):
                    x_shift += 1
                elif keyPressed == ord('a'):
                    x_shift -= 1
                elif keyPressed == ord('z'):
                    angle += 0.5
                elif keyPressed == ord('c'):
                    angle -= 0.5
                elif keyPressed == ord('o'):
                    overlay = not overlay
                elif keyPressed == ord('q'):
                    self.offsets.append([y_shift, x_shift, total_angle])
                    break

                canvas = np.zeros(self.canvas_shape, dtype=np.uint8)

                canvas_y_range = (max(0, y_shift), min(self.canvas_shape[0], curr_set.shape[0] + y_shift))
                canvas_x_range = (max(0, x_shift), min(self.canvas_shape[1], curr_set.shape[1] + x_shift))

                image_y_range = (abs(min(0, y_shift)), min(self.canvas_shape[0]-y_shift, curr_set.shape[0]))
                image_x_range = (abs(min(0, x_shift)), min(self.canvas_shape[1]-x_shift, curr_set.shape[1]))

                if angle != 0:
                    curr_set = self.rotate_image(curr_set, angle)

                canvas[0:ref_shape[0], 0:ref_shape[1], :] = ref_image // 2
                if overlay:
                    canvas[canvas_y_range[0]:canvas_y_range[1],
                           canvas_x_range[0]:canvas_x_range[1], :] += (curr_set[image_y_range[0]:image_y_range[1],
                                                                              image_x_range[0]:image_x_range[1], :] * 1) // 2
                else:
                    canvas[canvas_y_range[0]:canvas_y_range[1], canvas_x_range[0]:canvas_x_range[1], :] = ref_image

                canvas = np.clip(canvas, a_min=0, a_max=255)

                cv2.imshow(self.aligned_win, canvas)
                print(f"Translated by (rows = {y_shift}, cols = {x_shift}, angle={total_angle})")
                total_angle += angle
                angle = 0


        cv2.destroyAllWindows()

        # Save the array to a CSV file
        if save_offsets:
            if self.offset_file_path:
                np.savetxt(self.offset_file_path, np.array(self.offsets), delimiter=",")
            else:
                np.savetxt("Offsets.csv", np.array(self.offsets), delimiter=",")

        return self.offsets

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def stack_and_save_images_prev(self, file_path, offsets, save=False):
        # Display the overlaid image
        cv2.namedWindow(self.aligned_win, cv2.WINDOW_NORMAL)

        # Variables to track
        count = 0 #count_str = str(count).zfill(5)
        image_set_count = 0
        image_data_len = 0


        for image_set, offset in zip(self.all_image_lists, offsets):
            image_set_count += 1
            image_data_len = image_data_len + len(image_set)
            for image in image_set:
                image = cv2.imread(image)
                canvas = np.zeros(self.canvas_shape, dtype=np.uint8)

                angle = offset[2]

                canvas_y_range = (max(0, int(offset[0])), min(self.canvas_shape[0], image.shape[0] + int(offset[0])))
                canvas_x_range = (max(0, int(offset[1])), min(self.canvas_shape[1], image.shape[1] + int(offset[1])))

                image_y_range = (abs(min(0, int(offset[0]))), min(self.canvas_shape[0]-int(offset[0]), image.shape[0]))
                image_x_range = (abs(min(0, int(offset[1]))), min(self.canvas_shape[1]-int(offset[1]), image.shape[1]))


                canvas[canvas_y_range[0]:canvas_y_range[1],
                canvas_x_range[0]:canvas_x_range[1], :] = (image[image_y_range[0]:image_y_range[1],
                                                           image_x_range[0]:image_x_range[1], :])

                if angle != 0:
                    canvas = self.rotate_image(canvas, angle)
                    print("Rotated")

                cv2.imshow(self.aligned_win, canvas)
                count += 1
                print(f"{image_set_count} : {count}/{image_data_len}")

                if save:
                    cv2.imwrite(file_path + "CT_" +  str(count).zfill(5) + ".png", canvas)

                keyPressed = cv2.waitKey(1)
                if keyPressed == ord('n'):
                    break
                if keyPressed == ord('p'):
                    cv2.waitKey(0)

            ## Can be used to pause between set changes
            # keyPressed = cv2.waitKey(1)
            # if keyPressed == ord('q'):
            #     break

        cv2.destroyAllWindows()



if __name__ == '__main__':
    # Creating the list of folders whose images needs to be stacked
    images2Stack_filepath = [scan_set_path1, scan_set_path2]

    # Creating an object of class ImageStackHandler
    image_stack_handler = AligningImages(images2Stack_filepath)

    # Use either methods to gather the offsets
    # offsets =  image_stack_handler.get_offset_by_manual_overlapping()
    offsets = np.genfromtxt(recorded_offsets_file_path, delimiter=',')

    offsets = np.vstack((np.zeros((1,3)), offsets))
    offsets_cumm = np.cumsum(offsets, axis=0).astype(np.float32)
    image_stack_handler.stack_and_save_images_prev(aligned_images_file_path, offsets_cumm, save=True)