import cv2
import mediapipe as mp
import tkinter as tk
import time
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import threading
import random
from datetime import datetime
import json
import os

class EyeGazeTracker:
    def __init__(self):
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)
        
        # Screen setup
        self.screen_w = 1920  # Will be updated when UI starts
        self.screen_h = 1080
        
        # Tracking variables
        self.is_tracking = False
        self.is_calibrating = False
        self.cap = None
        self.current_frame = None
        
        # User management
        self.models = {}  # face_id -> (model_x, model_y)
        self.calibration_data = {}  # face_id -> (features, targets)
        self.face_registry = {}  # face_id -> (last known nose position, name, face_image)
        self.smoothed_points = {}
        self.user_colors = {}  # face_id -> color
        self.alpha = 0.25  # smoothing factor
        
        # Calibration setup
        self.calibration_points = []
        self.current_point_idx = 0
        self.frames_per_point = 20
        
        # UI components
        self.ui = None
        self.tracking_window = None
        self.canvas = None
        self.dot_items = []
        self.name_labels = []  # Store name labels for dots
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Face detection parameters
        self.face_detection_threshold = 0.05
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Available colors for user dots
        self.available_colors = ['#00ff00', '#ff0000', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', 
                               '#ffa500', '#800080', '#008000', '#ffc0cb']
        
        # Data persistence
        self.data_file = "user_data.json"
        self.load_user_data()
        
        # Face recognition enhancement
        self.face_embeddings = {}  # Store face embeddings for better recognition
        
    def set_ui(self, ui):
        """Set the UI reference"""
        self.ui = ui
        
    def initialize_camera(self):
        """Initialize camera capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
    def setup_tracking_window(self):
        """Setup fullscreen tracking window"""
        if self.tracking_window is None:
            self.tracking_window = tk.Toplevel()
            self.tracking_window.attributes('-fullscreen', True)
            self.tracking_window.attributes('-topmost', True)
            self.tracking_window.attributes('-transparentcolor', 'black')
            
            self.screen_w = self.tracking_window.winfo_screenwidth()
            self.screen_h = self.tracking_window.winfo_screenheight()
            
            self.canvas = tk.Canvas(
                self.tracking_window, 
                width=self.screen_w, 
                height=self.screen_h, 
                bg='black', 
                highlightthickness=0
            )
            self.canvas.pack()
            
            # Setup calibration points
            self.setup_calibration_points()
            
    def setup_calibration_points(self):
        """Setup calibration points in 3x3 grid"""
        self.calibration_points = [
            (self.screen_w // 5, self.screen_h // 5),
            (self.screen_w // 2, self.screen_h // 5),
            (self.screen_w - self.screen_w // 5, self.screen_h // 5),
            (self.screen_w // 5, self.screen_h // 2),
            (self.screen_w // 2, self.screen_h // 2),
            (self.screen_w - self.screen_w // 5, self.screen_h // 2),
            (self.screen_w // 5, self.screen_h - self.screen_h // 5),
            (self.screen_w // 2, self.screen_h - self.screen_h // 5),
            (self.screen_w - self.screen_w // 5, self.screen_h - self.screen_h // 5)
        ]
        random.shuffle(self.calibration_points)
        
    def get_center(self, landmarks):
        """Get center point of landmarks"""
        x_vals = [p.x for p in landmarks]
        y_vals = [p.y for p in landmarks]
        return (sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals))
    
    def extract_face_region(self, frame, landmarks):
        """Extract face region from frame using landmarks"""
        try:
            h, w, _ = frame.shape
            
            # Get face bounding box
            x_coords = [int(landmark.x * w) for landmark in landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in landmarks.landmark]
            
            x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
            y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
            
            face_region = frame[y_min:y_max, x_min:x_max]
            return face_region
        except:
            return None
    
    def calculate_face_embedding(self, landmarks):
        """Calculate a simple face embedding based on landmark positions"""
        try:
            # Key facial landmarks for identification
            key_points = [1, 33, 263, 61, 291, 199]  # nose, eye corners, mouth corners
            embedding = []
            
            for point_idx in key_points:
                landmark = landmarks.landmark[point_idx]
                embedding.extend([landmark.x, landmark.y, landmark.z if hasattr(landmark, 'z') else 0])
                
            return np.array(embedding)
        except:
            return None
    
    def find_matching_face(self, current_embedding, threshold=0.1):
        """Find matching face based on embedding similarity"""
        if current_embedding is None:
            return None
            
        best_match = None
        best_distance = float('inf')
        
        for face_id, stored_embedding in self.face_embeddings.items():
            # Only match properly registered faces
            if not self.is_face_registered(face_id):
                continue
                
            distance = np.linalg.norm(current_embedding - stored_embedding)
            print(f"Face {face_id} distance: {distance:.4f}")
            
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = face_id
                
        if best_match is not None:
            print(f"Found matching face: {best_match} with distance {best_distance:.4f}")
        else:
            print("No matching face found")
            
        return best_match
    
    def identify_face(self, nose_x, nose_y, frame, landmarks):
        """Identify face with improved recognition"""
        # Calculate face embedding
        current_embedding = self.calculate_face_embedding(landmarks)
        
        # First try embedding-based matching for existing users
        matched_face = self.find_matching_face(current_embedding)
        if matched_face is not None:
            # Update position and embedding
            old_data = self.face_registry[matched_face]
            self.face_registry[matched_face] = (nose_x, nose_y, old_data[2], old_data[3])
            self.face_embeddings[matched_face] = current_embedding
            return matched_face
        
        # Fallback to position-based matching for same session
        for face_id, (old_x, old_y, name, _) in self.face_registry.items():
            if abs(nose_x - old_x) < self.face_detection_threshold and abs(nose_y - old_y) < self.face_detection_threshold:
                # Update position and store embedding
                self.face_registry[face_id] = (nose_x, nose_y, name, self.face_registry[face_id][3])
                if current_embedding is not None:
                    self.face_embeddings[face_id] = current_embedding
                return face_id
        
        # Check if we should create a new face ID
        # Only create new ID if we don't have too many unregistered faces
        unregistered_count = len([fid for fid in self.face_registry if not self.is_face_registered(fid)])
        
        if unregistered_count >= 3:  # Limit unregistered faces to prevent spam
            print(f"Too many unregistered faces ({unregistered_count}), skipping new detection")
            return None
        
        # New face detected
        new_id = len(self.face_registry)
        face_image = self.extract_face_region(frame, landmarks)
        
        # Assign color to new user
        if new_id < len(self.available_colors):
            self.user_colors[new_id] = self.available_colors[new_id]
        else:
            # Generate random color if we run out of predefined colors
            self.user_colors[new_id] = f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"
        
        # Store embedding for new face
        if current_embedding is not None:
            self.face_embeddings[new_id] = current_embedding
        
        # Register new face temporarily
        self.face_registry[new_id] = (nose_x, nose_y, f"User_{new_id}", face_image)
        
        print(f"New face detected with ID {new_id}")
        
        # Handle new face detection (show dialog)
        if self.ui and face_image is not None:
            # Use after method to ensure thread safety
            self.ui.root.after(0, lambda: self.handle_new_face_detection(face_image, new_id))
            
        return new_id
    
    def handle_new_face_detection(self, face_image, face_id):
        """Handle new face detection with improved logic"""
        # Check if face is already registered (this prevents repeated dialogs)
        if face_id in self.face_registry and self.face_registry[face_id][2] != f"User_{face_id}":
            print(f"Face {face_id} ({self.face_registry[face_id][2]}) is already registered, skipping dialog")
            return  # Already has a proper name
            
        # Check if user is already calibrated (skip registration, go to calibration)
        if face_id in self.models:
            print(f"Face {face_id} is already calibrated, proceeding with tracking")
            return
            
        # Check if registration dialog is already pending
        if hasattr(self.ui, 'pending_faces') and face_id in self.ui.pending_faces:
            print(f"Registration dialog already pending for face {face_id}")
            return
            
        # Show registration dialog for new face
        print(f"Showing registration dialog for new face {face_id}")
        self.ui.show_face_capture_dialog(face_image, face_id)
    
    def register_face(self, face_id, name):
        """Register a face with a name"""
        if face_id in self.face_registry:
            old_data = self.face_registry[face_id]
            self.face_registry[face_id] = (old_data[0], old_data[1], name, old_data[3])
            self.save_user_data()
            print(f"Registered face {face_id} as {name}")
            
            # Update UI - this should be called from UI thread
            # if self.ui:
            #     self.ui.add_user_to_list(face_id, name, "Registered", self.user_colors.get(face_id, "#lime"))
    
    def get_registered_users(self):
        """Get list of properly registered users"""
        registered = []
        for face_id, (_, _, name, _) in self.face_registry.items():
            if not name.startswith("User_"):
                registered.append((face_id, name))
        return registered
    
    def debug_show_all_faces(self):
        """Debug method to show information about all detected faces"""
        print("\n=== FACE REGISTRY DEBUG ===")
        for face_id, (x, y, name, _) in self.face_registry.items():
            registered = self.is_face_registered(face_id)
            calibrated = face_id in self.models
            has_embedding = face_id in self.face_embeddings
            print(f"Face {face_id}: {name} | Pos: ({x:.3f}, {y:.3f}) | Reg: {registered} | Cal: {calibrated} | Emb: {has_embedding}")
        print("========================\n")
            
    def is_face_registered(self, face_id):
        """Check if a face is properly registered (has a name)"""
        if face_id in self.face_registry:
            name = self.face_registry[face_id][2]
            return not name.startswith("User_")
        return False
    
    def start_tracking(self):
        """Start the tracking system"""
        self.initialize_camera()
        self.setup_tracking_window()
        self.is_tracking = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self.tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        print("Eye gaze tracking started")
        
    def stop_tracking(self):
        """Stop the tracking system"""
        self.is_tracking = False
        
        if self.tracking_window:
            self.tracking_window.destroy()
            self.tracking_window = None
            self.canvas = None
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        print("Eye gaze tracking stopped")
        
    def start_calibration(self):
        """Start calibration process for registered users only"""
        if not self.is_tracking:
            self.start_tracking()
            
        # Only calibrate faces that are properly registered
        faces_to_calibrate = [fid for fid in self.face_registry if self.is_face_registered(fid)]
        
        if not faces_to_calibrate:
            print("No registered faces to calibrate. Please register faces first.")
            return
            
        self.is_calibrating = True
        self.current_point_idx = 0
        self.calibration_data.clear()
        
        print(f"Starting calibration for {len(faces_to_calibrate)} registered users")
        
        # Start calibration in separate thread
        calibration_thread = threading.Thread(target=self.calibration_loop)
        calibration_thread.daemon = True
        calibration_thread.start()
        
    def calibration_loop(self):
        """Main calibration loop"""
        while self.current_point_idx < len(self.calibration_points) and self.is_calibrating:
            self.collect_calibration_point()
            self.current_point_idx += 1
            time.sleep(0.5)
            
        if self.is_calibrating:
            self.train_models()
            self.is_calibrating = False
            print("Calibration completed")
            
            # Update UI status for all calibrated users
            if self.ui:
                for face_id in self.models:
                    if self.is_face_registered(face_id):
                        name = self.face_registry[face_id][2]
                        color = self.user_colors.get(face_id, "#lime")
                        self.ui.root.after(0, lambda fid=face_id, n=name, c=color: 
                                         self.ui.update_user_status(fid, "Calibrated", c))
                    
    def collect_calibration_point(self):
        """Collect data for a single calibration point"""
        if self.current_point_idx >= len(self.calibration_points):
            return
            
        dot_x, dot_y = self.calibration_points[self.current_point_idx]
        
        # Show calibration dot
        if self.canvas:
            dot = self.canvas.create_oval(
                dot_x - 30, dot_y - 30, dot_x + 30, dot_y + 30, 
                fill='yellow', outline='red', width=3
            )
            # Add instruction text
            instruction = self.canvas.create_text(
                dot_x, dot_y - 60, text=f"Look at the dot\n({self.current_point_idx + 1}/{len(self.calibration_points)})",
                fill='white', font=('Arial', 16), justify='center'
            )
            self.canvas.update()
            
        time.sleep(2)  # Give user time to look at the dot
        
        # Collect samples only from registered faces
        collected = {}
        for _ in range(self.frames_per_point):
            if not self.is_calibrating:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    feature = self.extract_features(landmarks)
                    if feature is not None:
                        nose = landmarks.landmark[1]
                        face_id = self.identify_face(nose.x, nose.y, frame, landmarks)
                        
                        # Only collect data from registered faces
                        if self.is_face_registered(face_id):
                            collected.setdefault(face_id, []).append(feature)
                        
            time.sleep(0.1)
            
        # Store collected data
        for face_id, features in collected.items():
            if features:
                avg_feature = np.mean(features, axis=0)
                self.calibration_data.setdefault(face_id, [[], []])
                self.calibration_data[face_id][0].append(avg_feature)
                self.calibration_data[face_id][1].append((dot_x, dot_y))
                
        # Remove calibration elements
        if self.canvas and 'dot' in locals():
            self.canvas.delete(dot)
            if 'instruction' in locals():
                self.canvas.delete(instruction)
            
    def extract_features(self, landmarks):
        """Extract features from face landmarks"""
        try:
            nose = landmarks.landmark[1]
            
            # Iris landmarks
            left_iris = [landmarks.landmark[i] for i in range(474, 478)]
            right_iris = [landmarks.landmark[i] for i in range(469, 473)]
            
            # Other facial landmarks
            chin = landmarks.landmark[152]
            left_outer = landmarks.landmark[33]
            right_outer = landmarks.landmark[263]
            
            # Calculate iris centers
            cx_left, cy_left = self.get_center(left_iris)
            cx_right, cy_right = self.get_center(right_iris)
            cx = (cx_left + cx_right) / 2
            cy = (cy_left + cy_right) / 2
            
            # Relative gaze direction
            rel_x = cx - nose.x
            rel_y = cy - nose.y
            
            # Face bounding box
            bbox = landmarks.landmark
            min_x = min(p.x for p in bbox)
            min_y = min(p.y for p in bbox)
            max_x = max(p.x for p in bbox)
            max_y = max(p.y for p in bbox)
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            
            # Head pose estimation
            yaw = nose.x - (left_outer.x + right_outer.x) / 2
            pitch = nose.y - chin.y
            roll = left_outer.y - right_outer.y
            
            return [rel_x, rel_y, min_x, min_y, bbox_w, bbox_h, yaw, pitch, roll]
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
            
    def train_models(self):
        """Train gaze prediction models for each user"""
        print("Training models...")
        
        for face_id in self.calibration_data:
            # Only train for registered faces
            if not self.is_face_registered(face_id):
                continue
                
            features, targets = self.calibration_data[face_id]
            
            if len(features) < 5:
                print(f"⚠️ Not enough samples for face {face_id}, skipping.")
                continue
                
            X = np.array(features)
            Y = np.array(targets)
            
            try:
                model_x = make_pipeline(
                    PolynomialFeatures(3), 
                    RANSACRegressor(min_samples=5)
                ).fit(X, Y[:, 0])
                
                model_y = make_pipeline(
                    PolynomialFeatures(3), 
                    RANSACRegressor(min_samples=5)
                ).fit(X, Y[:, 1])
                
                self.models[face_id] = (model_x, model_y)
                face_name = self.face_registry[face_id][2]
                print(f"✅ Model trained for {face_name} (ID: {face_id}) with {len(features)} samples.")
                
            except Exception as e:
                print(f"❌ Model training failed for face {face_id}: {e}")
                
        self.save_user_data()
        
    def tracking_loop(self):
        """Main tracking loop"""
        while self.is_tracking:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                self.current_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                # Clear previous dots and labels
                if self.canvas:
                    for item in self.dot_items:
                        self.canvas.delete(item)
                    self.dot_items.clear()
                    
                    for label in self.name_labels:
                        self.canvas.delete(label)
                    self.name_labels.clear()
                
                if results.multi_face_landmarks and not self.is_calibrating:
                    for landmarks in results.multi_face_landmarks:
                        self.process_face(landmarks, frame)
                        
                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - self.start_time)
                    self.start_time = current_time
                    
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"Tracking loop error: {e}")
                
    def process_face(self, landmarks, frame):
        """Process a single face for gaze tracking"""
        try:
            nose = landmarks.landmark[1]
            face_id = self.identify_face(nose.x, nose.y, frame, landmarks)
            
            print(f"Processing face {face_id}, registered: {self.is_face_registered(face_id)}, calibrated: {face_id in self.models}")
            
            # Show dots for all registered faces, even if not calibrated yet
            if self.is_face_registered(face_id):
                if face_id in self.models:
                    # Calibrated user - show predicted gaze point
                    feature = self.extract_features(landmarks)
                    if feature is not None:
                        model_x, model_y = self.models[face_id]
                        
                        # Predict gaze point
                        px = int(np.clip(model_x.predict([feature])[0], 0, self.screen_w))
                        py = int(np.clip(model_y.predict([feature])[0], 0, self.screen_h))
                        
                        # Apply smoothing
                        if face_id in self.smoothed_points:
                            prev_x, prev_y = self.smoothed_points[face_id]
                            px = int(self.alpha * px + (1 - self.alpha) * prev_x)
                            py = int(self.alpha * py + (1 - self.alpha) * prev_y)
                            
                        self.smoothed_points[face_id] = (px, py)
                        
                        # Draw gaze dot with name
                        if self.canvas:
                            color = self.user_colors.get(face_id, '#00ff00')
                            name = self.face_registry[face_id][2]
                            
                            # Create dot
                            dot = self.canvas.create_oval(
                                px - 15, py - 15, px + 15, py + 15, 
                                fill=color, outline='white', width=2
                            )
                            self.dot_items.append(dot)
                            
                            # Create name label next to dot
                            name_label = self.canvas.create_text(
                                px + 25, py - 5, text=name, 
                                fill='white', font=('Arial', 12, 'bold'),
                                anchor='w'
                            )
                            self.name_labels.append(name_label)
                            
                            print(f"Drew gaze dot for {name} at ({px}, {py})")
                else:
                    # Registered but not calibrated - show center dot as placeholder
                    if self.canvas:
                        color = self.user_colors.get(face_id, '#00ff00')
                        name = self.face_registry[face_id][2]
                        
                        # Show dot at screen center as placeholder
                        center_x, center_y = self.screen_w // 2, self.screen_h // 2
                        
                        # Create dot
                        dot = self.canvas.create_oval(
                            center_x - 15, center_y - 15, center_x + 15, center_y + 15, 
                            fill=color, outline='red', width=3
                        )
                        self.dot_items.append(dot)
                        
                        # Create name label and calibration message
                        name_label = self.canvas.create_text(
                            center_x + 25, center_y - 15, text=f"{name}", 
                            fill='white', font=('Arial', 12, 'bold'),
                            anchor='w'
                        )
                        self.name_labels.append(name_label)
                        
                        calibration_msg = self.canvas.create_text(
                            center_x + 25, center_y + 5, text="(Needs Calibration)", 
                            fill='yellow', font=('Arial', 10),
                            anchor='w'
                        )
                        self.name_labels.append(calibration_msg)
                        
                        print(f"Drew placeholder dot for {name} - needs calibration")
                        
        except Exception as e:
            print(f"Face processing error: {e}")
            import traceback
            traceback.print_exc()
            
    def get_current_frame(self):
        """Get current camera frame for preview"""
        return self.current_frame
        
    def get_status(self):
        """Get current system status"""
        active_users = len([fid for fid in self.face_registry if fid in self.smoothed_points and self.is_face_registered(fid)])
        calibrated_users = len([fid for fid in self.models if self.is_face_registered(fid)])
        
        return {
            'active_users': active_users,
            'calibrated_users': calibrated_users,
            'fps': self.fps,
            'total_users': len([fid for fid in self.face_registry if self.is_face_registered(fid)])
        }
        
    def reset_system(self):
        """Reset all system data"""
        self.models.clear()
        self.calibration_data.clear()
        self.face_registry.clear()
        self.smoothed_points.clear()
        self.user_colors.clear()
        self.face_embeddings.clear()
        
        # Clear data file
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
            
        print("System reset completed")
        
    def save_user_data(self):
        """Save user data to file"""
        try:
            data = {
                'face_registry': {
                    str(k): {
                        'position': v[:2],
                        'name': v[2],
                        'color': self.user_colors.get(k, '#00ff00')
                    } for k, v in self.face_registry.items()
                },
                'models_available': list(self.models.keys()),
                'face_embeddings': {
                    str(k): v.tolist() for k, v in self.face_embeddings.items()
                }
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving user data: {e}")
            
    def load_user_data(self):
        """Load user data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore face registry (without face images)
                for face_id_str, user_data in data.get('face_registry', {}).items():
                    face_id = int(face_id_str)
                    self.face_registry[face_id] = (
                        user_data['position'][0],
                        user_data['position'][1],
                        user_data['name'],
                        None  # Face image not saved
                    )
                    self.user_colors[face_id] = user_data.get('color', '#00ff00')
                
                # Restore face embeddings
                for face_id_str, embedding_list in data.get('face_embeddings', {}).items():
                    face_id = int(face_id_str)
                    self.face_embeddings[face_id] = np.array(embedding_list)
                    
                print(f"Loaded {len(self.face_registry)} users from previous session")
                
        except Exception as e:
            print(f"Error loading user data: {e}")


# Integration point - main application
def main():
    """Main application entry point"""
    from gaze_tracker_ui import EyeGazeTrackerUI
    
    # Create tracker instance
    tracker = EyeGazeTracker()
    
    # Create UI instance
    ui = EyeGazeTrackerUI(tracker)
    tracker.set_ui(ui)
    
    # Load existing users into UI
    for face_id, (_, _, name, _) in tracker.face_registry.items():
        color = tracker.user_colors.get(face_id, '#00ff00')
        status = "Calibrated" if face_id in tracker.models else "Registered"
        ui.add_user_to_list(face_id, name, status, color)
    
    # Start the UI
    ui.run()


if __name__ == "__main__":
    main()