import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from gui import Ui_MainWindow

class ImageProcessor:
    def __init__(self, ui: Ui_MainWindow):
        self.ui = ui
        self.original_image = None  # Original loaded image
        self.current_image = None   # Image after edits

        # Store brightness and contrast values
        self.brightness_value = 0
        self.contrast_value = 1.0

        self.connect_signals()

    def connect_signals(self):
        self.ui.uploadButton.clicked.connect(self.load_image)
        self.ui.saveButton.clicked.connect(self.save_image)
        self.ui.resetButton.clicked.connect(self.reset_image)

        self.ui.sketchButton.clicked.connect(self.apply_sketch)
        self.ui.SepiaButton.clicked.connect(self.apply_sepia)
        self.ui.cannyedgedetectButton.clicked.connect(self.apply_canny)
        self.ui.medianblurButton.clicked.connect(self.apply_median_blur)
        self.ui.gaussianblurButton.clicked.connect(self.apply_gaussian_blur)
        self.ui.sobeledgedetectButton.clicked.connect(self.apply_sobel)
        self.ui.grayscaleButton.clicked.connect(self.apply_grayscale)
        self.ui.negativeButton.clicked.connect(self.apply_negative)

        self.ui.brightnessSlider.valueChanged.connect(self.update_brightness_contrast)
        self.ui.contrastSlider.valueChanged.connect(self.update_brightness_contrast)
        self.ui.RotationcomboBox.currentTextChanged.connect(self.rotate_image)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                QMessageBox.warning(None, "Open Image", "Failed to load image!")
                return
            self.current_image = self.original_image.copy()
            self.show_image(self.current_image)
            # Reset sliders and combo box
            self.ui.brightnessSlider.setValue(0)
            self.ui.contrastSlider.setValue(10)  # Default 1.0 * 10
            self.ui.RotationcomboBox.setCurrentIndex(0)

    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(None, "Save Image", "No image to save!")
            return
        path, _ = QFileDialog.getSaveFileName(None, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path:
            cv2.imwrite(path, self.current_image)

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.show_image(self.current_image)
            self.ui.brightnessSlider.setValue(0)
            self.ui.contrastSlider.setValue(10)
            self.ui.RotationcomboBox.setCurrentIndex(0)

    def show_image(self, cv_img):
        """Convert cv image to QPixmap and show in QLabel"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.ui.imageViewlabel.setPixmap(pixmap)

    def update_brightness_contrast(self):
        if self.original_image is None:
            return
        # Brightness slider range: -100 to 100 (example), contrast slider range: 1 to 30 (mapped from 0-30)
        brightness = self.ui.brightnessSlider.value()  # Assuming -100 to 100
        contrast = self.ui.contrastSlider.value() / 10  # Slider 0-30, scaled to 0-3
        if contrast < 0.1:
            contrast = 0.1  # Avoid zero or negative contrast

        # Apply brightness and contrast
        img = cv2.convertScaleAbs(self.original_image, alpha=contrast, beta=brightness)
        self.current_image = img
        self.show_image(self.current_image)

    def rotate_image(self, angle_str):
        if self.current_image is None:
            return
        angle = int(angle_str)
        if angle == 0:
            rotated = self.current_image
        else:
            (h, w) = self.current_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(self.current_image, M, (w, h))
        self.current_image = rotated
        self.show_image(self.current_image)

    # --- Image effect functions ---

    def apply_grayscale(self):
        if self.current_image is None:
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back for display consistency
        self.show_image(self.current_image)

    def apply_sketch(self):
        if self.current_image is None:
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = cv2.bitwise_not(blur)
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        self.current_image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        self.show_image(self.current_image)

    def apply_sepia(self):
        if self.current_image is None:
            return
        img = self.current_image.copy().astype(np.float64)
        img = cv2.transform(img, np.matrix([[0.393, 0.769, 0.189],
                                            [0.349, 0.686, 0.168],
                                            [0.272, 0.534, 0.131]]))
        img = np.clip(img, 0, 255)
        self.current_image = img.astype(np.uint8)
        self.show_image(self.current_image)

    def apply_canny(self):
        if self.current_image is None:
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        self.current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.show_image(self.current_image)

    def apply_median_blur(self):
        if self.current_image is None:
            return
        blur = cv2.medianBlur(self.current_image, 5)
        self.current_image = blur
        self.show_image(self.current_image)

    def apply_gaussian_blur(self):
        if self.current_image is None:
            return
        blur = cv2.GaussianBlur(self.current_image, (7, 7), 0)
        self.current_image = blur
        self.show_image(self.current_image)

    def apply_sobel(self):
        if self.current_image is None:
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        self.show_image(self.current_image)

    def apply_negative(self):
        if self.current_image is None:
            return
        negative = cv2.bitwise_not(self.current_image)
        self.current_image = negative
        self.show_image(self.current_image)
