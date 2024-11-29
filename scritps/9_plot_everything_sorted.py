import numpy as np
import matplotlib.pyplot as plt
import itertools
import glob
import  os


class PlotSorted:
    def __init__(self, contour_file_path, scale=None):
        self.colors = itertools.cycle(plt.cm.tab10.colors)  # 10 distinctive colors
        self.contours_name_list = self.load_contour_file_list(contour_file_path)
        self.contours = [np.genfromtxt(file, delimiter=',') for file in self.contours_name_list]
        self.scale = scale

    @staticmethod
    def load_contour_file_list(contour_file_path):
        contour_file_list = [file for file in sorted(glob.glob(contour_file_path),
                                                     key=lambda s: int(os.path.splitext(os.path.basename(s))[0][-4:]))]
        return contour_file_list

    @staticmethod
    def best_fit_circle(contour):
        '''# Have to find summation of x, y, x^2, y^2, x^3, y^3, xy, xy^2, x^2y
        # A = n(x^2) - (x)^2
        # B = n(xy) - (x)(y)
        # C = n(y^2) - (y)^2
        # D = 0.5 {n(xy^2) - (x)(y^2) + n(x^3) - (x)(x^2)}
        # E = 0.5 {n(x^2y) - (x^2)(y) + n(y^3) - (y)(y^2)}'''

        X, Y, X2, Y2, XY, XY2, X2Y, X3, Y3, n = [0] * 10
        radius = 0

        for point in contour:
            x, y = point.ravel()

            X = X + x
            Y = Y + y

            X2 = X2 + x ** 2
            Y2 = Y2 + y ** 2

            X3 = X3 + x ** 3
            Y3 = Y3 + y ** 3

            XY = XY + (x * y)
            XY2 = XY2 + (x * (y ** 2))
            X2Y = X2Y + (y * (x ** 2))

            n = n + 1

        A = (n * X2) - X * X
        B = (n * XY) - X * Y
        C = (n * Y2) - Y ** 2
        D = 0.5 * ((n * XY2) - (X * Y2) + (n * X3) - (X * X2))
        E = 0.5 * ((n * X2Y) - (X2 * Y) + (n * Y3) - (Y * Y2))

        xc = ((D * C) - (B * E)) / ((A * C) - (B * B))
        yc = ((A * E) - (B * D)) / ((A * C) - (B * B))

        for point in contour:
            x, y, = point.ravel()

            norm_distance = (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) / n
            radius = radius + norm_distance

        return xc, yc, radius

    @staticmethod
    def moving_average(data, window_size=5):
        # Smooth data with a simple moving average
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    def plot_all_contours_and_cylindricity(self, plot_path=None):
        fig1, ax1 = plt.subplots(figsize=(15, 15))
        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(15, 15))
        ax1.set_aspect('equal')
        ax2.set_ylim(-800, 200)

        max_x, max_y = -np.inf, -np.inf
        min_x, min_y =  np.inf,  np.inf

        upper_bound, lower_bound, max_rad = -np.inf, np.inf, 0
        error_list = []

        for idx, contour in enumerate(self.contours):
            contour = contour * self.scale[idx]
            center_x, center_y, radius = self.best_fit_circle(contour)
            theta = np.arctan2(contour[:, 1], contour[:, 0])
            dist = np.linalg.norm(contour, axis=1) - radius
            error = self.moving_average(dist, window_size=50)
            error_list.extend(error)

            theta = np.append(theta, theta[0])
            error= np.append(error, error[0])

            color = next(self.colors)
            legend_val = str(22.5 * (idx+1)) + "°"
            # legend_val = str(250*idx) + "μm"

            ax1.plot(contour[:, 0], contour[:, 1], label='Contour @ '+legend_val, color=color, linewidth=2)
            ax2.plot(theta, error, label='Contour @ '+legend_val, color=color, linewidth=2)

            # Plot the best-fit circle
            circle = plt.Circle((0, 0), radius, color=color, fill=False, linestyle='--', label='Best-Fit Circle @ '+legend_val, linewidth=2,zorder=2)
            ax1.add_artist(circle)


            max_x = max(max(contour[:, 0]), max_x)
            max_y = max(max(contour[:, 1]), max_y)
            min_x = min(min(contour[:, 0]), min_x)
            min_y = min(min(contour[:, 1]), min_y)

            # Calculate the bounds for the error (radial differences)
            upper_bound = max(max(error), upper_bound)
            lower_bound = min(min(error), lower_bound)
            max_rad = max(radius, max_rad)

        # Plot the Upper and Lower Bounds
        ax2.plot(theta, [0] * len(theta), linestyle='--', color='black', linewidth=2.5, label="Zero-Error Line", zorder=2)
        ax2.plot(theta, [upper_bound] * len(theta), linestyle=':', color='blue', linewidth=2, label="Upper & Lower Bounds", zorder=2)
        ax2.plot(theta, [lower_bound] * len(theta), linestyle=':', color='blue', linewidth=2,  zorder=2)

        rmse =  np.linalg.norm(np.asarray(error_list))/np.sqrt(len(error_list))
        print(f"Upper Bound = {upper_bound},  Lower bound: {lower_bound},  Avg_Error: {rmse}")

        #
        padding = max_rad * 0.2
        ax1.set_xlim(min_x - padding, max_x + padding)
        ax1.set_ylim(min_y - padding, max_y + padding)

        # Labels, legend, and plot adjustments
        ax1.set_title('Overlayed contours along the channel with best-fit circles', fontweight='bold', fontsize=15)
        ax1.set_yticks([ -1500,  1500])
        ax1.set_xticks([ -1500,  1500])
        ax1.set_xlabel('X (μm)',fontweight='bold')
        ax1.set_ylabel('Y (μm)', fontweight='bold')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True)


        # Set specific ticks for the radial axis at the zero, upper, and lower bounds
        ax2.set_title('Circularity along the Channel',fontweight='bold', fontsize=15)
        ax2.set_yticks([-200, -100, 0, 100, 200])
        ax2.set_yticklabels(['-200 μm','-100 μm', '0 μm', '100 μm','200 μm' ])
        ax2.set_rlabel_position(270)
        ax2.yaxis.grid(True, color='gray', linestyle=':', linewidth=0.8)
        ax2.legend(loc='upper right', bbox_to_anchor=(0, 1))
        ax2.grid(True, which='minor', axis='x')

        # Show the plot
        fig1.tight_layout(pad=2.0)
        fig2.tight_layout(pad=2.0)

        if plot_path:
            fig1.savefig(plot_path + "contours_overlayed.png", format='png', dpi=600, bbox_inches='tight')
            fig2.savefig(plot_path + "channel_circularity.png", format='png', dpi=600, bbox_inches='tight')

        plt.show()

if __name__ == '__main__':
    file_str = ""
    contour_file_path = f"./*.csv"
    plot_path = ""

    scale_3ug = [0.78, 0.81, 0.81, 0.81, 0.81, 0.81, 0.78]
    scale_3g  = [1.00] * 7

    scale_5ug = [1.00, 1.00, 1.00, 1.00, 0.96, 0.96, 1.00]
    scale_5g  = [1.05, 1.05, 1.08, 1.08, 1.09, 1.09, 1.10]

    scale_7ug = [1.02, 1.00, 1.00]
    scale_7g = [1.00, 1.00, 1.02]

    plots = PlotSorted(contour_file_path, scale = scale_7g)
    plots.plot_all_contours_and_cylindricity(plot_path)