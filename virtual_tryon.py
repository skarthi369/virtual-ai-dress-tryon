import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import os

class VirtualTryOn:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("AR Virtual Try-On")
        self.window.geometry("1400x800")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize clothing items
        self.current_shirt = None
        self.current_pants = None
        
        # Initialize scaling factors
        self.shirt_scale = 1.0
        self.pants_scale = 1.0
        self.shirt_width_scale = 1.0
        self.pants_width_scale = 1.0
        self.shirt_length_scale = 1.0
        self.pants_length_scale = 1.0
        
        # Load clothing images
        self.shirts = self.load_clothing_images('shirts')
        self.pants = self.load_clothing_images('pants')
        
        # Recommended combinations
        self.combinations = [
            ("white-tshirt.png", "blue-jeans.png"),
            ("black-tshirt.png", "blue-jeans.png"),
            ("blue-tshirt.png", "black-pants.png"),
            ("red-tshirt.png", "gray-pants.png")
        ]
        
        # Create UI elements
        self.setup_ui()

    def load_clothing_images(self, folder):
        images = {}
        folder_path = os.path.join('images', folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    img_path = os.path.join(folder_path, filename)
                    # Read image with alpha channel
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # If image doesn't have alpha channel, add it
                        if img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            # Make white background transparent
                            white = np.all(img[:, :, :3] > 240, axis=2)
                            img[white, 3] = 0
                        
                        # Normalize alpha channel
                        if img.shape[2] == 4:
                            img[:, :, 3] = cv2.normalize(img[:, :, 3], None, 0, 255, cv2.NORM_MINMAX)
                        
                        images[filename] = {
                            'image': img,
                            'name': os.path.splitext(filename)[0]
                        }
        return images

    def setup_ui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video frame
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(side="left", padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack()
        
        # Control panel
        self.control_panel = ctk.CTkFrame(self.main_frame)
        self.control_panel.pack(side="right", padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            self.control_panel,
            text="Virtual Try-On",
            font=("Helvetica", 24, "bold")
        ).pack(pady=20)
        
        # Shirts selection
        ctk.CTkLabel(
            self.control_panel,
            text="Select Shirt:",
            font=("Helvetica", 16)
        ).pack(pady=5)
        
        self.shirt_var = tk.StringVar()
        self.shirt_combo = ctk.CTkComboBox(
            self.control_panel,
            values=list(self.shirts.keys()),
            variable=self.shirt_var,
            command=self.update_shirt
        )
        self.shirt_combo.pack(pady=5)
        
        # Shirt size adjustment
        size_frame = ctk.CTkFrame(self.control_panel)
        size_frame.pack(pady=10, padx=5, fill="x")
        
        ctk.CTkLabel(size_frame, text="Shirt Size:", font=("Helvetica", 14)).pack()
        
        # Shirt width scale
        ctk.CTkLabel(size_frame, text="Width:").pack()
        self.shirt_width_slider = ctk.CTkSlider(
            size_frame,
            from_=0.5,
            to=2.0,
            number_of_steps=100,
            command=lambda v: setattr(self, 'shirt_width_scale', float(v))
        )
        self.shirt_width_slider.set(1.0)
        self.shirt_width_slider.pack(pady=5)
        
        # Shirt length scale
        ctk.CTkLabel(size_frame, text="Length:").pack()
        self.shirt_length_slider = ctk.CTkSlider(
            size_frame,
            from_=0.5,
            to=2.0,
            number_of_steps=100,
            command=lambda v: setattr(self, 'shirt_length_scale', float(v))
        )
        self.shirt_length_slider.set(1.0)
        self.shirt_length_slider.pack(pady=5)
        
        # Pants selection
        ctk.CTkLabel(
            self.control_panel,
            text="Select Pants:",
            font=("Helvetica", 16)
        ).pack(pady=5)
        
        self.pants_var = tk.StringVar()
        self.pants_combo = ctk.CTkComboBox(
            self.control_panel,
            values=list(self.pants.keys()),
            variable=self.pants_var,
            command=self.update_pants
        )
        self.pants_combo.pack(pady=5)
        
        # Pants size adjustment
        pants_size_frame = ctk.CTkFrame(self.control_panel)
        pants_size_frame.pack(pady=10, padx=5, fill="x")
        
        ctk.CTkLabel(pants_size_frame, text="Pants Size:", font=("Helvetica", 14)).pack()
        
        # Pants width scale
        ctk.CTkLabel(pants_size_frame, text="Width:").pack()
        self.pants_width_slider = ctk.CTkSlider(
            pants_size_frame,
            from_=0.5,
            to=2.0,
            number_of_steps=100,
            command=lambda v: setattr(self, 'pants_width_scale', float(v))
        )
        self.pants_width_slider.set(1.0)
        self.pants_width_slider.pack(pady=5)
        
        # Pants length scale
        ctk.CTkLabel(pants_size_frame, text="Length:").pack()
        self.pants_length_slider = ctk.CTkSlider(
            pants_size_frame,
            from_=0.5,
            to=2.0,
            number_of_steps=100,
            command=lambda v: setattr(self, 'pants_length_scale', float(v))
        )
        self.pants_length_slider.set(1.0)
        self.pants_length_slider.pack(pady=5)
        
        # Recommended combinations
        ctk.CTkLabel(
            self.control_panel,
            text="Recommended Combinations",
            font=("Helvetica", 16, "bold")
        ).pack(pady=20)
        
        self.combo_frame = ctk.CTkFrame(self.control_panel)
        self.combo_frame.pack(pady=5)
        
        for shirt_file, pants_file in self.combinations:
            shirt_name = os.path.splitext(shirt_file)[0].replace('-', ' ').title()
            pants_name = os.path.splitext(pants_file)[0].replace('-', ' ').title()
            
            combo_btn = ctk.CTkButton(
                self.combo_frame,
                text=f"{shirt_name} + {pants_name}",
                command=lambda s=shirt_name, p=pants_name: self.apply_combination(s, p),
                font=("Helvetica", 12)
            )
            combo_btn.pack(pady=2)
    
    def update_shirt(self, selection):
        self.current_shirt = self.shirts[selection]['image']
    
    def update_pants(self, selection):
        self.current_pants = self.pants[selection]['image']
    
    def apply_combination(self, shirt, pants):
        self.shirt_var.set(shirt)
        self.pants_var.set(pants)
        self.current_shirt = self.shirts[shirt]['image']
        self.current_pants = self.pants[pants]['image']
    
    def overlay_image(self, frame, overlay, x, y):
        try:
            if overlay is None:
                return frame
            
            # Get dimensions
            h, w = overlay.shape[:2]
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate placement coordinates
            y1 = max(0, y)
            y2 = min(frame_h, y + h)
            x1 = max(0, x)
            x2 = min(frame_w, x + w)
            
            # Crop overlay if it goes outside frame bounds
            overlay_y1 = 0 if y >= 0 else -y
            overlay_y2 = h - max(0, (y + h) - frame_h)
            overlay_x1 = 0 if x >= 0 else -x
            overlay_x2 = w - max(0, (x + w) - frame_w)
            
            # Get alpha channel
            if overlay.shape[2] == 4:
                alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)
                
                # Extract BGR channels
                overlay_colors = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
                
                # Get frame region
                frame_region = frame[y1:y2, x1:x2]
                
                # Blend images
                blended = frame_region * (1 - alpha) + overlay_colors * alpha
                
                # Place blended region back in frame
                frame[y1:y2, x1:x2] = blended
            
            return frame
        except Exception as e:
            print(f"Error in overlay_image: {str(e)}")
            return frame
    
    def overlay_clothing(self, frame, landmarks):
        if landmarks.pose_landmarks:
            h, w, _ = frame.shape
            
            # Get key body landmarks
            left_shoulder = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            
            # Draw shirt
            if self.current_shirt is not None:
                # Calculate shoulder width and add padding
                shoulder_width = abs(right_shoulder.x - left_shoulder.x) * w
                padding = shoulder_width * 0.4  # 40% padding for better fit
                
                # Calculate torso height
                torso_height = abs(left_shoulder.y - left_hip.y) * h
                
                shirt_points = np.array([
                    [int(left_shoulder.x * w - padding), int(left_shoulder.y * h)],
                    [int(right_shoulder.x * w + padding), int(right_shoulder.y * h)],
                    [int(right_hip.x * w + padding * 0.8), int(right_hip.y * h + torso_height * 0.1)],
                    [int(left_hip.x * w - padding * 0.8), int(left_hip.y * h + torso_height * 0.1)]
                ], np.int32)
                
                # Calculate center of shirt
                center_x = (shirt_points[0][0] + shirt_points[1][0]) // 2
                center_y = (shirt_points[0][1] + shirt_points[1][1]) // 2
                
                # Calculate shirt width and height
                shirt_width = abs(shirt_points[1][0] - shirt_points[0][0])
                shirt_height = abs(shirt_points[3][1] - shirt_points[0][1])
                
                # Resize shirt
                resized_shirt = cv2.resize(self.current_shirt, (int(shirt_width * self.shirt_width_scale), int(shirt_height * self.shirt_length_scale)))
                
                # Overlay shirt
                frame = self.overlay_image(frame, resized_shirt, center_x - resized_shirt.shape[1] // 2, center_y - resized_shirt.shape[0] // 2)
            
            # Draw pants
            if self.current_pants is not None:
                # Calculate hip width and add padding
                hip_width = abs(right_hip.x - left_hip.x) * w
                padding = hip_width * 0.5  # 50% padding for better fit
                
                # Calculate leg length and adjust bottom position
                leg_length = abs(left_hip.y - left_knee.y) * h
                bottom_y = min(int(h), int(max(left_knee.y, right_knee.y) * h + leg_length * 0.8))
                
                pants_points = np.array([
                    [int(left_hip.x * w - padding), int(left_hip.y * h)],
                    [int(right_hip.x * w + padding), int(right_hip.y * h)],
                    [int(right_knee.x * w + padding * 0.7), int(right_knee.y * h)],
                    [int(right_knee.x * w + padding * 0.6), bottom_y],
                    [int(left_knee.x * w - padding * 0.6), bottom_y],
                    [int(left_knee.x * w - padding * 0.7), int(left_knee.y * h)]
                ], np.int32)
                
                # Calculate center of pants
                center_x = (pants_points[0][0] + pants_points[1][0]) // 2
                center_y = (pants_points[0][1] + pants_points[1][1]) // 2
                
                # Calculate pants width and height
                pants_width = abs(pants_points[1][0] - pants_points[0][0])
                pants_height = abs(pants_points[5][1] - pants_points[0][1])
                
                # Resize pants
                resized_pants = cv2.resize(self.current_pants, (int(pants_width * self.pants_width_scale), int(pants_height * self.pants_length_scale)))
                
                # Overlay pants
                frame = self.overlay_image(frame, resized_pants, center_x - resized_pants.shape[1] // 2, center_y - resized_pants.shape[0] // 2)
        
        return frame
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                frame = self.overlay_clothing(frame, results)
                
                # Draw pose landmarks for debugging
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
            
            # Convert frame to PhotoImage
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        
        self.window.after(10, self.update_frame)
    
    def run(self):
        self.update_frame()
        self.window.mainloop()
    
    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    app = VirtualTryOn()
    app.run()
