import numpy as np
import matplotlib.pyplot as plt


class ContourPlotter:
    def __init__(self, ground_contour_path, unground_contour_path, scale=None):
        if scale is None:
            scale = [1, 1]

        self.scale = scale
        self.ground_contour   = np.genfromtxt(ground_contour_path, delimiter=',') * scale[0]
        self.unground_contour = np.genfromtxt(unground_contour_path, delimiter=',') * scale[1]

    def plot(self, plot_section_id="22.5°",  plot_path=None):
        fig1, ax1 = plt.subplots(figsize=(15, 15))
        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(15, 15))
        ax1.set_aspect('equal')
        ax2.set_ylim(-800, 200)

        # plot_title = str(22.5 * (idx+1)) + "°"
        # plot_title = str(250) + "μm"

        max_x, max_y = -np.inf, -np.inf
        min_x, min_y = np.inf, np.inf

        ## --------------------------------------------- ##
        center_xug, center_yug, radius_ug = self.best_fit_circle(self.unground_contour)
        theta_ug = np.arctan2(self.unground_contour[:, 1], self.unground_contour[:, 0])
        dist_ug = np.linalg.norm(self.unground_contour, axis=1) - radius_ug
        error_ug = self.moving_average(dist_ug, window_size=50)
        rmse_ug = np.linalg.norm(error_ug) / np.sqrt(len(error_ug))

        theta_ug = np.append(theta_ug, theta_ug[0])
        error_ug = np.append(error_ug, error_ug[0])

        ax1.plot(self.unground_contour[:, 0], self.unground_contour[:, 1], label='Before Grinding', color='r', linewidth=2)
        ax2.plot(theta_ug, error_ug, label='Before Grinding', color='r', linewidth=2)
        # Plot the best-fit circle
        circle = plt.Circle((0, 0), radius_ug, color='r', fill=False, linestyle='--', label='Best-Fit Circle @ Before Grinding', linewidth=2, zorder=2)
        ax1.add_artist(circle)
        ## -------------------------------------------------------------- ##

        center_xg, center_yg, radius_g = self.best_fit_circle(self.ground_contour)
        theta_g = np.arctan2(self.ground_contour[:, 1], self.ground_contour[:, 0])
        dist_g = np.linalg.norm(self.ground_contour, axis=1) - radius_g
        error_g = self.moving_average(dist_g, window_size=50)
        rmse_g = np.linalg.norm(error_g) / np.sqrt(len(error_g))

        theta_g = np.append(theta_g, theta_g[0])
        error_g = np.append(error_g, error_g[0])

        ax1.plot(self.ground_contour[:, 0], self.ground_contour[:, 1], label='After Grinding', color='b', linewidth=2)
        ax2.plot(theta_g, error_g, label='After Grinding', color='b', linewidth=2)
        # Plot the best-fit circle
        circle = plt.Circle((0, 0), radius_g, color='b', fill=False, linestyle='--', label='Best-Fit Circle @ After Grinding', linewidth=2, zorder=2)
        ax1.add_artist(circle)
        ## ----------------------------------------------------------------------- ##

        print(f"Sample \t Before Grinding(μm) \t After Grinding(μm)")
        print(f"RMSE:  \t {rmse_ug:.2f},  \t\t\t\t {rmse_g:.2f}")
        print(f"Radius:\t {radius_ug:.2f},  \t\t\t\t {radius_g:.2f}")
        print(f"Scale :\t {self.scale[1]:.2f},  \t\t\t\t {self.scale[0]:.2f}")

        max_x = max(max(self.ground_contour[:, 0]), max(self.unground_contour[:, 0]))
        max_y = max(max(self.ground_contour[:, 1]), max(self.unground_contour[:, 1]))
        min_x = min(min(self.ground_contour[:, 0]), min(self.unground_contour[:, 0]))
        min_y = min(min(self.ground_contour[:, 1]), min(self.unground_contour[:, 1]))

        # Mark the center of the circle
        ax1.plot(0, 0, 'o', color=[0, 0, 0], markersize=5)
        # Plot the Upper and Lower Bounds
        ax2.plot(theta_g, [0] * len(theta_g), linestyle='--', color='black', linewidth=2.5, label="Zero-Error Line",zorder=2)



        #
        padding = ((max_x**2 + max_y**2)**0.5) * 0.2
        ax1.set_xlim(min_x - padding, max_x + padding)
        ax1.set_ylim(min_y - padding, max_y + padding)

        # Labels, legend, and plot adjustments
        ax1.set_title(f'Channel Sections Before and After Grinding @ {plot_section_id}', fontweight='bold', fontsize=15)
        ax1.set_yticks([-1500,  1500])
        ax1.set_xticks([-1500,  1500])
        ax1.set_xlabel('X (μm)', fontweight='bold')
        ax1.set_ylabel('Y (μm)', fontweight='bold')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True)

        # Set specific ticks for the radial axis at the zero, upper, and lower bounds
        ax2.set_title(f'Circularity of the Channel Sections Before and After Grinding @ {plot_section_id}', fontweight='bold', fontsize=15)
        ax2.set_yticks([-200, -100, 0, 100, 200])
        ax2.set_yticklabels(['-200 μm', '-100 μm', '0 μm', '100 μm', '200 μm'])
        ax2.set_rlabel_position(270)
        ax2.yaxis.grid(True, color='gray', linestyle=':', linewidth=0.8)
        ax2.legend(loc='upper right', bbox_to_anchor=(0, 1))
        ax2.grid(True, which='minor', axis='x')

        # Show the plot
        fig1.tight_layout(pad=2.0)
        fig2.tight_layout(pad=2.0)

        if plot_path:
            fig1.savefig(plot_path+"contours_overlayed.png", format='png', dpi=600, bbox_inches='tight')
            fig2.savefig(plot_path+"channel_circularity.png", format='png', dpi=600, bbox_inches='tight')

        plt.show()


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


if __name__ == '__main__':
    set_name = ["", ""]
    slice_id = ""
    ground_contour_path   = f"{set_name[1]}_1{slice_id}.csv"
    unground_contour_path = f"{set_name[1]}_2{slice_id}.csv"
    plot_path = f"{set_name[0]}/Compare2/{slice_id}/"

    contour_plotter = ContourPlotter(ground_contour_path, unground_contour_path, [1.02, 1.0])
    contour_plotter.plot(plot_section_id="67.5°", plot_path=plot_path)
