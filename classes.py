from active_contour import *
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from scipy.signal import convolve2d


class Image:
    def __init__(self, image):
        self._original_img = image
        self._gray_scale_image = self.calculate_gray_scale_image(self.original_img)
        self._img_histogram = self.calculate_image_histogram()
        self.equalized_img, self.equalized_hist = self.equalize_image()
        self.normalized_img, self.normalized_hist = self.equalize_image(normalize=True)
        self._bgr_img_histograms = self.calculate_bgr_histogram_and_distribution()

    @property
    def original_img(self):
        return self._original_img

    @property
    def gray_scale_image(self):
        return self._gray_scale_image

    @property
    def bgr_img_histograms(self):
        return self._bgr_img_histograms

    @property
    def img_histogram(self):
        return self._img_histogram

    def calculate_gray_scale_image(self, img):
        # Extract the B, G, and R channels
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Convert to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
        gray_image = 0.299 * r + 0.587 * g + 0.114 * b

        # Convert to uint8 data type
        return gray_image.astype(np.uint8)

    def calculate_bgr_histogram_and_distribution(self):
        # Split the image into its three color channels: B, G, and R
        b, g, r = self.original_img[:, :, 0], self.original_img[:, :, 1], self.original_img[:, :, 2]

        # Calculate histograms for each color channel
        hist_b, _ = np.histogram(b.flatten(), bins=256, range=(0.0, 256.0))
        hist_g, _ = np.histogram(g.flatten(), bins=256, range=(0.0, 256.0))
        hist_r, _ = np.histogram(r.flatten(), bins=256, range=(0.0, 256.0))

        # Calculate grayscale histogram
        gray_image = np.dot(self.original_img[..., :3], [0.2989, 0.5870, 0.1140])
        hist_gray, _ = np.histogram(gray_image.flatten(), bins=256, range=(0.0, 256.0))

        return hist_b, hist_g, hist_r, hist_gray

    def calculate_image_histogram(self):
        hist, _ = np.histogram(self.original_img.flatten(), 256, (0, 256))
        return hist

    def calculate_image_histogram_cv(self):
        hist = cv2.calcHist([self.original_img], [0], None, [256], [0, 256])
        return hist.flatten()

    def equalize_image(self, normalize=False):
        hist, _ = np.histogram(self.original_img.flatten(), 256, (0, 256))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        sk = np.round(cdf_normalized * 255)
        equalized_image = sk[self.original_img]
        equalized_hist, _ = np.histogram(equalized_image.flatten(), 256, (0, 256))
        if normalize:
            normalized_image = equalized_image / 255.0
            normalized_hist, _ = np.histogram(normalized_image.flatten(), 256, (0, 1), density=True)
            return normalized_image, normalized_hist
        return equalized_image, equalized_hist

    def equalize_image_cv(self, normalize=False):
        equalized_image = cv2.equalizeHist(self.original_img)
        if normalize:
            equalized_image = equalized_image / 255.0
        return equalized_image


class WorkerSignals(QObject):
    finished = pyqtSignal()
    update = pyqtSignal(np.ndarray)
    calc_area_perimeter = pyqtSignal(float, float)


class WorkerThread(QThread):
    def __init__(self, input_image, contour_points, epochs=100, edges_img=None):
        super(WorkerThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image
        self.contour_points = np.array(contour_points).astype(int)
        self.epochs = epochs
        self.edges_img = edges_img
        self.contour = None

    def run(self):
        img = np.array(self.input_image)

        if self.edges_img is None:
            edges = np.array(get_img_edges(img))
        else:
            edges = np.array(self.edges_img)

        edges = add_border(edges)

        self.contour = Contour(self.contour_points)

        # # Add a ton more points
        self.contour.insert_points()
        self.contour.insert_points()

        # Create series of images fitting contour

        for i in range(self.epochs):
            updated_img_con = np.copy(edges)
            img_copy = np.copy(img)

            self.contour.calc_energies(updated_img_con)
            self.contour.update_points()

            self.contour.draw_contour(img_copy)
            self.signals.update.emit(img_copy)

            points = list(zip(self.contour.contour_r, self.contour.contour_c))
            area, perimeter = self.calculate_area(points), self.calculate_perimeter(points)
            self.signals.calc_area_perimeter.emit(area, perimeter)

        self.signals.finished.emit()

    def calculate_perimeter(self, points):
        perimeter = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            perimeter += np.sqrt(dx ** 2 + dy ** 2)
        return perimeter

    def calculate_area(self, points):
        area = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            area += p1[0] * p2[1] - p1[1] * p2[0]
        return abs(area) / 2


class HoughTransform:
    def __init__(self, input_image):
        self.input_image = input_image
        self.img_gaussian = None
        self.gradient_magnitude = None
        self.gradient_theta = None
        self.non_max_suppression_img = None
        self.hysteresis_image = None
        self.edge_image_canny = None
        self.lines = []
        self.accumlator = None
        self.num_rho = None
        self.num_theta = None
        self.bin_threshold = None
        self.output_image = None
        self.out_lines = []

    def gaussian(self, img, sigma=1.0):
        """
        Apply Gaussian filter to an image.

        The Gaussian filter is a type of convolution filter that is used to 'blur' the image or reduce detail and noise.

        Parameters:
            img (numpy.ndarray): Input image.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            result (numpy.ndarray): The image after applying the Gaussian filter.
        """
        kernel_size = (int(6 * sigma) + 1, int(6 * sigma) + 1)  # Determine kernel size based on sigma

        # Apply Gaussian blur
        result = cv2.GaussianBlur(img, kernel_size, sigma)

        return result

    def sobel(self, img):
        """
        Apply Sobel operator to an image.

        The Sobel operator is used in image processing and computer vision, particularly within edge detection
        algorithms.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            result (numpy.ndarray): The image after applying the Sobel operator.
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = convolve2d(img, kernel_x, mode='same', boundary='symm')
        grad_y = convolve2d(img, kernel_y, mode='same', boundary='symm')

        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_theta = np.arctan2(grad_y, grad_x)

        self.gradient_magnitude = gradient_magnitude
        self.gradient_theta = gradient_theta

        return gradient_magnitude, gradient_theta

    def non_max_suppression(self, image, theta):
        M, N = image.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = image[i, j + 1]
                        r = image[i, j - 1]
                    # Angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = image[i + 1, j - 1]
                        r = image[i - 1, j + 1]
                    # Angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = image[i + 1, j]
                        r = image[i - 1, j]
                    # Angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = image[i - 1, j - 1]
                        r = image[i + 1, j + 1]

                    if (image[i, j] >= q) and (image[i, j] >= r):
                        Z[i, j] = image[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def threshold(self, image, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
        high_threshold = image.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        M, N = image.shape
        result = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(image >= high_threshold)
        zeros_i, zeros_j = np.where(image < low_threshold)

        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

        result[strong_i, strong_j] = strong
        result[weak_i, weak_j] = weak

        return result, weak, strong

    def hysteresis(self, image, weak, strong=255):
        M, N = image.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if image[i, j] == weak:
                    try:
                        if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (
                                image[i + 1, j + 1] == strong)
                                or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                                or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                        image[i - 1, j + 1] == strong)):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass
        self.hysteresis_image = image
        return image

    def canny_edge_detection(self, sigma=0.5, low_threshold_ratio=0.05, high_threshold_ratio=0.07):
        image = self.input_image
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian filter
        self.img_gaussian = self.gaussian(gray_image, sigma)

        # Sobel filtering
        gradient_magnitude, gradient_theta = self.sobel(self.img_gaussian)

        # Non-maximum suppression
        suppressed_image = self.non_max_suppression(gradient_magnitude, gradient_theta)

        # Thresholding
        thresholded_image, weak_pixel, strong_pixel = self.threshold(suppressed_image, low_threshold_ratio,
                                                                     high_threshold_ratio)

        # Hysteresis
        final_image = self.hysteresis(thresholded_image, weak_pixel, strong_pixel)

        self.edge_image_canny = final_image

        return final_image

    def calculate_ranges(self, edge_image, num_rhos, num_thetas):
        img_height, img_width = edge_image.shape[:2]
        diag_len = np.sqrt(np.square(img_height) + np.square(img_width))
        dtheta = 180 / num_thetas
        drho = (2 * diag_len) / num_rhos
        thetas = np.arange(0, 180, step=dtheta)
        rhos = np.arange(-diag_len, diag_len, step=drho)
        return img_height, img_width, diag_len, dtheta, drho, thetas, rhos

    def calculate_cos_sin_thetas(self, thetas):
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))
        return cos_thetas, sin_thetas

    def process_pixel(self, edge_pt, cos_thetas, sin_thetas, thetas, rhos):
        hough_rhos, hough_thetas = [], []
        for theta_idx in range(len(thetas)):
            rho = (edge_pt[1] * cos_thetas[theta_idx]) + (edge_pt[0] * sin_thetas[theta_idx])
            theta = thetas[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            hough_rhos.append(rho)
            hough_thetas.append(theta)
        return hough_rhos, hough_thetas

    def find_hough_lines(self, image, edge_image, num_rhos, num_thetas, bin_threshold):
        img_height, img_width, diag_len, dtheta, drho, thetas, rhos = self.calculate_ranges(edge_image, num_rhos,
                                                                                            num_thetas)
        cos_thetas, sin_thetas = self.calculate_cos_sin_thetas(thetas)
        self.accumulator = np.zeros((len(rhos), len(thetas)))
        img_height_half = img_height / 2
        img_width_half = img_width / 2

        for y in range(img_height):
            for x in range(img_width):
                if edge_image[y][x] != 0:
                    edge_pt = [y - img_height_half, x - img_width_half]
                    hough_rhos, hough_thetas = self.process_pixel(edge_pt, cos_thetas, sin_thetas, thetas, rhos)
                    for rho, theta in zip(hough_rhos, hough_thetas):
                        self.accumulator[np.argmin(np.abs(rhos - rho))][np.argmin(np.abs(thetas - theta))] += 1

        output_img = image.copy()
        out_lines = []

        for y in range(self.accumulator.shape[0]):
            for x in range(self.accumulator.shape[1]):
                if self.accumulator[y][x] > bin_threshold:
                    rho = rhos[y]
                    theta = thetas[x]
                    a = np.cos(np.deg2rad(theta))
                    b = np.sin(np.deg2rad(theta))
                    x0 = (a * rho) + img_width_half
                    y0 = (b * rho) + img_height_half
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    output_img = cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    out_lines.append((rho, theta, x1, y1, x2, y2))
        self.output_image = output_img
        self.out_lines = out_lines
        return output_img, out_lines

    def assess_edge_quality(self, edge_image):
        # Example: Assess edge quality based on edge density
        edge_density = np.sum(edge_image) / (edge_image.shape[0] * edge_image.shape[1])
        return edge_density

    def tune_parameters(self, edge_image, num_rho=180, num_theta=180, bin_threshold=156):
        # Initialize parameters
        default_num_rho = num_rho
        default_num_theta = num_theta
        default_bin_threshold = bin_threshold

        # Assess edge quality
        edge_quality = self.assess_edge_quality(edge_image)
        print(edge_quality)
        # Define threshold to determine edge quality
        threshold1 = 30  # Example threshold value
        # Adjust parameters based on edge quality
        if edge_quality < threshold1:
            num_rho = default_num_rho
            num_theta = default_num_theta
            bin_threshold = default_bin_threshold
        else:
            # Adjust parameters for better edge quality
            # Example: Increase num_rho and num_theta
            num_rho = default_num_rho * 2
            num_theta = default_num_theta * 2
            bin_threshold = default_bin_threshold

        return num_rho, num_theta, bin_threshold

    # Circle ---------------------------------------

    def draw_circles(self, image, centers):
        for center in centers:
            cv2.circle(image, (center[0], center[1]), center[2], (0, 0, 255), 2)

    def hough_circle(self, img_edges, min_radius, max_radius, threshold, min_dist):
        h, w = img_edges.shape
        accumulator = np.zeros((h, w, max_radius - min_radius + 1))

        # Generate arrays of x and y coordinates for edge pixels
        y_coords, x_coords = np.where(img_edges > 0)

        # Generate arrays of radius and angle values
        radius_values = np.arange(min_radius, max_radius + 1)
        angle_values = np.deg2rad(np.arange(0, 360))

        # Calculate corresponding (a, b) coordinates for each edge pixel and radius
        for radius in radius_values:
            for angle in angle_values:
                a_coords = np.round(x_coords - radius * np.cos(angle)).astype(int)
                b_coords = np.round(y_coords - radius * np.sin(angle)).astype(int)

                # Filter out of bounds coordinates
                valid_coords_mask = (a_coords >= 0) & (
                        a_coords < w) & (b_coords >= 0) & (b_coords < h)
                a_coords = a_coords[valid_coords_mask]
                b_coords = b_coords[valid_coords_mask]

                # Increment accumulator at valid coordinates
                accumulator[b_coords, a_coords, radius - min_radius] += 1

        circles = []
        for radius in range(max_radius - min_radius + 1):
            acc_slice = accumulator[:, :, radius]
            peaks = np.argwhere((acc_slice >= threshold))
            for peak in peaks:
                x, y, r = peak[1], peak[0], radius + min_radius
                # Check if the new circle center is at least min_distance away from existing centers
                # if all(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) >= min_dist for cx, cy, _ in circles):
                #     circles.append((x, y, r))
                for cx, cy, _ in circles:
                    if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < min_dist:
                        break
                else:
                    circles.append((x, y, r))

        return circles

    def main_hough_circle(self, img, img_edges, min_radius, max_radius, threshold,
                          min_dist_factor):  # img: original image, img_edges: after Canny
        circles = self.hough_circle(img_edges, min_radius, max_radius, threshold, img_edges.shape[0] / min_dist_factor)
        print("no. of circles", len(circles))
        self.draw_circles(img, circles)
        return img
