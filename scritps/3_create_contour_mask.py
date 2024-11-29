import numpy as np
import cv2

class ExtractMask:
    def __init__(self, image_path, binary_path=None):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError("The image file was not found or could not be loaded.")

        self.binary_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        self.masked_image = cv2.bitwise_and(self.img, self.img, mask=self.binary_mask)

        self.win_name = 'Reference Image'
        self.bin_win = 'Masked Image'
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.bin_win, cv2.WINDOW_NORMAL)

        self.binary_path = binary_path
        self.drawing = False
        self.finish = False

        self.points = []
        self.temp_point = (0,0)

    def set_poly_points(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points.append((x,y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_point =(x,y)

    def dummy_callback(self, event, x ,y, flags, params):
        pass

    def run(self):
        cv2.setMouseCallback(self.win_name, self.set_poly_points)
        keyPressed = None

        while keyPressed != ord('q'):

            if self.finish:
                cv2.setMouseCallback(self.win_name, self.dummy_callback)

            mask = self.draw_polygon()
            temp_image = cv2.addWeighted(self.img, 0.5, mask, 0.5, 0)

            self.binary_mask = np.zeros_like(mask[:, :, 0])
            for channel in range(3):
                    self.binary_mask |= (mask[:,:,channel] > 0)

            self.binary_mask = self.binary_mask *255

            cv2.imshow(self.win_name, temp_image)
            cv2.imshow(self.bin_win, self.binary_mask)

            keyPressed = cv2.waitKey(10)

            if keyPressed == ord('c'):
                self.finish = True
                self.drawing = False
            elif keyPressed == ord('s'):
                self.save_binary_mask()



        cv2.destroyAllWindows()

    def draw_polygon(self):
        ref_img = np.zeros_like(self.img)
        if self.drawing == True and self.finish == False:
            pts1 = self.points.copy()
            pts2 = self.points[1:]
            pts2.append(self.temp_point)

            for (x1, y1), (x2, y2) in zip(pts1, pts2):
                cv2.line(ref_img, (x1,y1), (x2,y2), (0,255,0), 5)

        elif self.drawing == False and self.finish == True:
            pts = np.array(self.points, np.int32)
            pts = pts.reshape((-1,1,2))

            cv2.fillPoly(ref_img, [pts], color=(255,0 ,0), lineType=cv2.LINE_AA)

        return ref_img

    def save_binary_mask(self):
        if self.binary_path:
            cv2.imwrite(self.binary_path, self.binary_mask)
        else:
            cv2.imwrite('Mask_Unnamed.png', self.binary_mask)


if __name__ =='__main__':
    image_path_to_create_mask  = "CT_00025.png"
    binary_mask_path = "mask_3mm.png"

    mask_generator = ExtractMask(image_path_to_create_mask, binary_path=binary_mask_path)
    mask_generator.run()

    ## Key Controls
    # c : closes the mask profile
    # s : saves the current mask once c is pressed

    print("Mask successfully saved!!")

