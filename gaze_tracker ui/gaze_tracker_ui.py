import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime

class EyeGazeTrackerUI:
    def __init__(self, gaze_tracker):
        self.gaze_tracker = gaze_tracker
        self.setup_ui()
        self.face_capture_window = None
        self.preview_running = False
        self.preview_thread = None
        self.pending_faces = set()  # Track faces that are being processed
        
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Eye Gaze Tracker - Multi-User System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(True, True)
        
        # Create main frame with modern styling
        self.main_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.create_header()
        self.create_control_panel()
        self.create_user_panel()
        self.create_status_panel()
        self.create_preview_panel()
        
    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg='#1a1a2e')
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🎯 Eye Gaze Tracker",
            font=('Arial', 28, 'bold'),
            fg='#00d4ff',
            bg='#1a1a2e'
        )
        title_label.pack(side='left')
        
        subtitle_label = tk.Label(
            header_frame,
            text="Multi-User Eye Tracking System with Smart Recognition",
            font=('Arial', 12),
            fg='#888',
            bg='#1a1a2e'
        )
        subtitle_label.pack(side='left', padx=(20, 0))
        
        # System status indicator
        self.status_indicator = tk.Label(
            header_frame,
            text="●",
            font=('Arial', 20),
            fg='#ff4757',
            bg='#1a1a2e'
        )
        self.status_indicator.pack(side='right')
        
        status_text = tk.Label(
            header_frame,
            text="System Status",
            font=('Arial', 10),
            fg='#888',
            bg='#1a1a2e'
        )
        status_text.pack(side='right', padx=(0, 10))
        
    def create_control_panel(self):
        control_frame = tk.LabelFrame(
            self.main_frame,
            text="System Controls",
            font=('Arial', 14, 'bold'),
            fg='#00d4ff',
            bg='#16213e',
            bd=2,
            relief='raised'
        )
        control_frame.pack(fill='x', pady=(0, 15))
        
        button_frame = tk.Frame(control_frame, bg='#16213e')
        button_frame.pack(fill='x', padx=20, pady=15)
        
        # Modern button styling
        button_style = {
            'font': ('Arial', 12, 'bold'),
            'height': 2,
            'width': 15,
            'relief': 'flat',
            'cursor': 'hand2'
        }
        
        self.start_btn = tk.Button(
            button_frame,
            text="🚀 Start Tracking",
            bg='#2ed573',
            fg='white',
            activebackground='#26d465',
            command=self.start_tracking,
            **button_style
        )
        self.start_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Stop Tracking",
            bg='#ff4757',
            fg='white',
            activebackground='#ff3838',
            command=self.stop_tracking,
            state='disabled',
            **button_style
        )
        self.stop_btn.pack(side='left', padx=10)
        
        self.calibrate_btn = tk.Button(
            button_frame,
            text="🎯 Calibrate Users",
            bg='#ffa502',
            fg='white',
            activebackground='#ff9500',
            command=self.start_calibration,
            **button_style
        )
        self.calibrate_btn.pack(side='left', padx=10)
        
        self.reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset System",
            bg='#747d8c',
            fg='white',
            activebackground='#57606f',
            command=self.reset_system,
            **button_style
        )
        self.reset_btn.pack(side='left', padx=10)
        
        # Debug button - remove in production
        self.debug_btn = tk.Button(
            button_frame,
            text="🔍 Debug Info",
            bg='#9c88ff',
            fg='white',
            activebackground='#8c7ae6',
            command=self.show_debug_info,
            font=('Arial', 10, 'bold'),
            height=2,
            width=12
        )
        self.debug_btn.pack(side='left', padx=10)
        
        # Add instruction label
        instruction_frame = tk.Frame(control_frame, bg='#16213e')
        instruction_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        instruction_label = tk.Label(
            instruction_frame,
            text="💡 Instructions: Start tracking → New faces will prompt for registration → Calibrate registered users → Track with names!",
            font=('Arial', 10),
            fg='#ffa502',
            bg='#16213e',
            wraplength=800,
            justify='left'
        )
        instruction_label.pack(anchor='w')
        
    def create_user_panel(self):
        user_frame = tk.LabelFrame(
            self.main_frame,
            text="Registered Users",
            font=('Arial', 14, 'bold'),
            fg='#00d4ff',
            bg='#16213e',
            bd=2,
            relief='raised'
        )
        user_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Create scrollable user list
        canvas_frame = tk.Frame(user_frame, bg='#16213e')
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        self.user_canvas = tk.Canvas(canvas_frame, bg='#16213e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.user_canvas.yview)
        self.scrollable_frame = tk.Frame(self.user_canvas, bg='#16213e')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.user_canvas.configure(scrollregion=self.user_canvas.bbox("all"))
        )
        
        self.user_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.user_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.user_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # User list header
        header_frame = tk.Frame(self.scrollable_frame, bg='#16213e')
        header_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(header_frame, text="User ID", font=('Arial', 12, 'bold'), 
                fg='#00d4ff', bg='#16213e', width=10).pack(side='left')
        tk.Label(header_frame, text="Name", font=('Arial', 12, 'bold'), 
                fg='#00d4ff', bg='#16213e', width=20).pack(side='left')
        tk.Label(header_frame, text="Status", font=('Arial', 12, 'bold'), 
                fg='#00d4ff', bg='#16213e', width=15).pack(side='left')
        tk.Label(header_frame, text="Dot Color", font=('Arial', 12, 'bold'), 
                fg='#00d4ff', bg='#16213e', width=15).pack(side='left')
        tk.Label(header_frame, text="Actions", font=('Arial', 12, 'bold'), 
                fg='#00d4ff', bg='#16213e', width=15).pack(side='left')
        
        self.user_entries = {}
        
    def create_status_panel(self):
        status_frame = tk.LabelFrame(
            self.main_frame,
            text="System Status",
            font=('Arial', 14, 'bold'),
            fg='#00d4ff',
            bg='#16213e',
            bd=2,
            relief='raised'
        )
        status_frame.pack(fill='x', pady=(0, 15))
        
        status_content = tk.Frame(status_frame, bg='#16213e')
        status_content.pack(fill='x', padx=20, pady=15)
        
        # Status information
        left_status = tk.Frame(status_content, bg='#16213e')
        left_status.pack(side='left', fill='x', expand=True)
        
        self.active_users_label = tk.Label(
            left_status,
            text="Active Users: 0",
            font=('Arial', 12),
            fg='#2ed573',
            bg='#16213e'
        )
        self.active_users_label.pack(anchor='w')
        
        self.calibrated_users_label = tk.Label(
            left_status,
            text="Calibrated Users: 0",
            font=('Arial', 12),
            fg='#ffa502',
            bg='#16213e'
        )
        self.calibrated_users_label.pack(anchor='w')
        
        right_status = tk.Frame(status_content, bg='#16213e')
        right_status.pack(side='right')
        
        self.fps_label = tk.Label(
            right_status,
            text="FPS: 0",
            font=('Arial', 12),
            fg='#00d4ff',
            bg='#16213e'
        )
        self.fps_label.pack(anchor='e')
        
        self.timestamp_label = tk.Label(
            right_status,
            text=f"Started: {datetime.now().strftime('%H:%M:%S')}",
            font=('Arial', 10),
            fg='#888',
            bg='#16213e'
        )
        self.timestamp_label.pack(anchor='e')
        
    def create_preview_panel(self):
        preview_frame = tk.LabelFrame(
            self.main_frame,
            text="Camera Preview",
            font=('Arial', 14, 'bold'),
            fg='#00d4ff',
            bg='#16213e',
            bd=2,
            relief='raised'
        )
        preview_frame.pack(fill='x')
        
        self.preview_label = tk.Label(
            preview_frame,
            text="Camera preview will appear here when tracking starts",
            font=('Arial', 12),
            fg='#888',
            bg='#16213e',
            height=8
        )
        self.preview_label.pack(padx=20, pady=15)
        
    def start_tracking(self):
        self.gaze_tracker.start_tracking()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_indicator.config(fg='#2ed573')
        self.start_camera_preview()
        self.update_status()
        
    def stop_tracking(self):
        self.gaze_tracker.stop_tracking()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_indicator.config(fg='#ff4757')
        self.stop_camera_preview()
        
    def start_calibration(self):
        # Check if there are registered users
        registered_users = [fid for fid in self.gaze_tracker.face_registry 
                          if self.gaze_tracker.is_face_registered(fid)]
        
        if not registered_users:
            messagebox.showwarning(
                "No Registered Users", 
                "Please register at least one user before calibration.\n\nStart tracking to detect and register new faces."
            )
            return
            
        if messagebox.askyesno(
            "Calibration", 
            f"Start calibration process for {len(registered_users)} registered user(s)?\n\nThis will improve gaze tracking accuracy."
        ):
            self.gaze_tracker.start_calibration()
            self.show_calibration_window()
            
    def reset_system(self):
        if messagebox.askyesno(
            "Reset System", 
            "This will clear all user data including:\n• Registered faces\n• Calibration data\n• User preferences\n\nContinue?"
        ):
            self.gaze_tracker.reset_system()
            self.clear_user_list()
            self.update_status()
            messagebox.showinfo("Reset Complete", "System has been reset successfully!")
    
    def show_debug_info(self):
        """Show debug information about the current system state"""
        self.gaze_tracker.debug_show_all_faces()
        
        # Create debug window
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Debug Information")
        debug_window.geometry("600x400")
        debug_window.configure(bg='#1a1a2e')
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(debug_window, bg='#1a1a2e')
        text_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, bg='#16213e', fg='white', font=('Courier', 10))
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Debug information
        debug_info = "=== GAZE TRACKER DEBUG INFO ===\n\n"
        debug_info += f"Tracking: {self.gaze_tracker.is_tracking}\n"
        debug_info += f"Calibrating: {self.gaze_tracker.is_calibrating}\n"
        debug_info += f"Total registered faces: {len(self.gaze_tracker.face_registry)}\n"
        debug_info += f"Calibrated models: {len(self.gaze_tracker.models)}\n"
        debug_info += f"Face embeddings: {len(self.gaze_tracker.face_embeddings)}\n\n"
        
        debug_info += "FACE REGISTRY:\n"
        for face_id, (x, y, name, _) in self.gaze_tracker.face_registry.items():
            registered = self.gaze_tracker.is_face_registered(face_id)
            calibrated = face_id in self.gaze_tracker.models
            has_embedding = face_id in self.gaze_tracker.face_embeddings
            color = self.gaze_tracker.user_colors.get(face_id, "N/A")
            debug_info += f"  Face {face_id}: {name}\n"
            debug_info += f"    Position: ({x:.3f}, {y:.3f})\n"
            debug_info += f"    Registered: {registered}\n"
            debug_info += f"    Calibrated: {calibrated}\n"
            debug_info += f"    Has Embedding: {has_embedding}\n"
            debug_info += f"    Color: {color}\n\n"
        
        debug_info += "UI USER ENTRIES:\n"
        for user_id, entry in self.user_entries.items():
            debug_info += f"  UI Entry {user_id}: {entry['name']}\n"
        
        text_widget.insert('1.0', debug_info)
        text_widget.config(state='disabled')
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
            
    def show_face_capture_dialog(self, face_image, face_id):
        """Show dialog to name a new face with improved logic"""
        # Prevent multiple dialogs for the same face
        if face_id in self.pending_faces:
            print(f"Dialog already pending for face {face_id}")
            return
            
        # Check if face is already registered
        if self.gaze_tracker.is_face_registered(face_id):
            print(f"Face {face_id} is already registered")
            return
            
        # Check if face is already calibrated (skip registration)
        if face_id in self.gaze_tracker.models:
            print(f"Face {face_id} is already calibrated, skipping registration")
            return
            
        print(f"Showing registration dialog for face {face_id}")
        self.pending_faces.add(face_id)
        
        # Close any existing dialog
        if self.face_capture_window:
            self.face_capture_window.destroy()
            
        self.face_capture_window = tk.Toplevel(self.root)
        self.face_capture_window.title("New Face Detected!")
        self.face_capture_window.geometry("450x600")
        self.face_capture_window.configure(bg='#1a1a2e')
        self.face_capture_window.transient(self.root)
        self.face_capture_window.grab_set()
        
        # Center the window
        self.face_capture_window.geometry("+{}+{}".format(
            int(self.root.winfo_x() + self.root.winfo_width()/2 - 225),
            int(self.root.winfo_y() + self.root.winfo_height()/2 - 300)
        ))
        
        # Header
        header_frame = tk.Frame(self.face_capture_window, bg='#1a1a2e')
        header_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            header_frame,
            text="👤 New Face Detected!",
            font=('Arial', 18, 'bold'),
            fg='#00d4ff',
            bg='#1a1a2e'
        ).pack()
        
        tk.Label(
            header_frame,
            text="Would you like to register this person?",
            font=('Arial', 12),
            fg='#888',
            bg='#1a1a2e'
        ).pack(pady=(5, 0))
        
        # Display captured face
        if face_image is not None:
            face_img = cv2.resize(face_image, (200, 200))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_photo = ImageTk.PhotoImage(Image.fromarray(face_img))
            
            img_frame = tk.Frame(self.face_capture_window, bg='#16213e', relief='raised', bd=2)
            img_frame.pack(pady=20)
            
            img_label = tk.Label(img_frame, image=face_photo, bg='#16213e')
            img_label.image = face_photo  # Keep a reference
            img_label.pack(padx=10, pady=10)
        
        # Name input section
        input_frame = tk.Frame(self.face_capture_window, bg='#1a1a2e')
        input_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            input_frame,
            text="Enter a name for this person:",
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#1a1a2e'
        ).pack(anchor='w')
        
        name_entry = tk.Entry(
            input_frame,
            font=('Arial', 14),
            width=25,
            justify='center',
            bg='#16213e',
            fg='white',
            insertbackground='white'
        )
        name_entry.pack(pady=(10, 0), fill='x')
        name_entry.focus()
        
        # Information section
        info_frame = tk.Frame(self.face_capture_window, bg='#16213e', relief='raised', bd=1)
        info_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            info_frame,
            text="ℹ️ After registration:",
            font=('Arial', 11, 'bold'),
            fg='#00d4ff',
            bg='#16213e'
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        info_text = "• Use 'Calibrate Users' to improve tracking accuracy\n• The person's name will appear next to their gaze dot\n• System will remember this person in future sessions"
        tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 10),
            fg='white',
            bg='#16213e',
            justify='left'
        ).pack(anchor='w', padx=20, pady=(0, 10))
        
        # Button section
        button_frame = tk.Frame(self.face_capture_window, bg='#1a1a2e')
        button_frame.pack(fill='x', padx=20, pady=20)
        
        def save_face():
            name = name_entry.get().strip()
            if name:
                if len(name) > 20:
                    messagebox.showwarning("Name Too Long", "Please enter a name with 20 characters or less.")
                    return
                    
                print(f"Registering face {face_id} as {name}")
                self.gaze_tracker.register_face(face_id, name)
                self.add_user_to_list(face_id, name, "Registered", self.gaze_tracker.user_colors.get(face_id, "#lime"))
                
                self.face_capture_window.destroy()
                self.face_capture_window = None
                self.pending_faces.discard(face_id)
                
                # Show success message
                messagebox.showinfo(
                    "Registration Successful!", 
                    f"✅ {name} has been registered successfully!\n\nUse 'Calibrate Users' to improve tracking accuracy for {name}."
                )
            else:
                messagebox.showwarning("Invalid Name", "Please enter a valid name.")
                
        def skip_face():
            print(f"Skipping registration for face {face_id}")
            # Remove the unregistered face from registry
            if face_id in self.gaze_tracker.face_registry:
                del self.gaze_tracker.face_registry[face_id]
            if face_id in self.gaze_tracker.user_colors:
                del self.gaze_tracker.user_colors[face_id]
            if face_id in self.gaze_tracker.face_embeddings:
                del self.gaze_tracker.face_embeddings[face_id]
                
            self.face_capture_window.destroy()
            self.face_capture_window = None
            self.pending_faces.discard(face_id)
            
        tk.Button(
            button_frame,
            text="✅ Register",
            bg='#2ed573',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=12,
            height=2,
            command=save_face
        ).pack(side='left', padx=(0, 10))
        
        tk.Button(
            button_frame,
            text="⏭️ Skip",
            bg='#ff4757',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=12,
            height=2,
            command=skip_face
        ).pack(side='left')
        
        # Bind Enter key to save
        name_entry.bind('<Return>', lambda e: save_face())
        
        # Handle window closing
        def on_closing():
            print(f"Dialog closed for face {face_id}")
            self.pending_faces.discard(face_id)
            # Remove the unregistered face from registry
            if face_id in self.gaze_tracker.face_registry:
                name = self.gaze_tracker.face_registry[face_id][2]
                if name.startswith("User_"):  # Only remove if not properly registered
                    del self.gaze_tracker.face_registry[face_id]
            if face_id in self.gaze_tracker.user_colors:
                del self.gaze_tracker.user_colors[face_id]
            if face_id in self.gaze_tracker.face_embeddings:
                del self.gaze_tracker.face_embeddings[face_id]
                
            self.face_capture_window.destroy()
            self.face_capture_window = None

        self.face_capture_window.protocol("WM_DELETE_WINDOW", on_closing)

    def add_user_to_list(self, user_id, name, status="Detected", color="#lime"):
        """Add user to the user list display with enhanced features"""
        # Check if user already exists and update instead
        if user_id in self.user_entries:
            self.update_user_in_list(user_id, name, status, color)
            return
            
        user_frame = tk.Frame(self.scrollable_frame, bg='#0f172a', relief='raised', bd=1)
        user_frame.pack(fill='x', pady=2, padx=5)
        
        # User ID
        tk.Label(user_frame, text=f"#{user_id}", font=('Arial', 10, 'bold'), 
                fg='#00d4ff', bg='#0f172a', width=10).pack(side='left', padx=5, pady=5)
        
        # Name
        name_label = tk.Label(user_frame, text=name, font=('Arial', 10, 'bold'), 
                             fg='white', bg='#0f172a', width=20)
        name_label.pack(side='left', padx=5, pady=5)
        
        # Status
        status_color = '#2ed573' if status == "Calibrated" else '#ffa502' if status == "Registered" else '#888'
        status_label = tk.Label(user_frame, text=status, font=('Arial', 10, 'bold'), 
                               fg=status_color, bg='#0f172a', width=15)
        status_label.pack(side='left', padx=5, pady=5)
        
        # Color indicator
        color_indicator = tk.Label(user_frame, text="●", font=('Arial', 16), 
                                  fg=color, bg='#0f172a', width=15)
        color_indicator.pack(side='left', padx=5, pady=5)
        
        # Action buttons
        action_frame = tk.Frame(user_frame, bg='#0f172a')
        action_frame.pack(side='left', padx=5, pady=2)
        
        if status != "Calibrated":
            calibrate_btn = tk.Button(
                action_frame,
                text="Cal",
                bg='#ffa502',
                fg='white',
                font=('Arial', 8, 'bold'),
                width=4,
                command=lambda uid=user_id: self.calibrate_single_user(uid)
            )
            calibrate_btn.pack(side='left', padx=1)
        
        delete_btn = tk.Button(
            action_frame,
            text="×",
            bg='#ff4757',
            fg='white',
            font=('Arial', 8, 'bold'),
            width=3,
            command=lambda uid=user_id: self.delete_user(uid)
        )
        delete_btn.pack(side='left', padx=1)
        
        self.user_entries[user_id] = {
            'frame': user_frame,
            'name_label': name_label,
            'status_label': status_label,
            'color_indicator': color_indicator,
            'action_frame': action_frame,
            'name': name
        }
        
    def update_user_in_list(self, user_id, name, status, color):
        """Update existing user in the list"""
        if user_id in self.user_entries:
            entry = self.user_entries[user_id]
            entry['name_label'].config(text=name)
            
            status_color = '#2ed573' if status == "Calibrated" else '#ffa502' if status == "Registered" else '#888'
            entry['status_label'].config(text=status, fg=status_color)
            entry['color_indicator'].config(fg=color)
            entry['name'] = name
            
            # Update action buttons
            for widget in entry['action_frame'].winfo_children():
                widget.destroy()
                
            if status != "Calibrated":
                calibrate_btn = tk.Button(
                    entry['action_frame'],
                    text="Cal",
                    bg='#ffa502',
                    fg='white',
                    font=('Arial', 8, 'bold'),
                    width=4,
                    command=lambda uid=user_id: self.calibrate_single_user(uid)
                )
                calibrate_btn.pack(side='left', padx=1)
            
            delete_btn = tk.Button(
                entry['action_frame'],
                text="×",
                bg='#ff4757',
                fg='white',
                font=('Arial', 8, 'bold'),
                width=3,
                command=lambda uid=user_id: self.delete_user(uid)
            )
            delete_btn.pack(side='left', padx=1)
        
    def calibrate_single_user(self, user_id):
        """Calibrate a single user"""
        if user_id not in self.gaze_tracker.face_registry:
            messagebox.showerror("Error", "User not found!")
            return
            
        if not self.gaze_tracker.is_face_registered(user_id):
            messagebox.showwarning("Not Registered", "Please register this user first!")
            return
        
        name = self.gaze_tracker.face_registry[user_id][2]
        if messagebox.askyesno("Calibrate User", f"Start calibration for {name}?"):
            # Implement single user calibration (you may need to modify the tracker for this)
            self.gaze_tracker.start_calibration()
            self.show_calibration_window()
    
    def delete_user(self, user_id):
        """Delete a user from the system"""
        if user_id not in self.gaze_tracker.face_registry:
            return
            
        name = self.gaze_tracker.face_registry[user_id][2]
        if messagebox.askyesno("Delete User", f"Delete {name} from the system?\n\nThis will remove all data for this user."):
            # Remove from tracker
            if user_id in self.gaze_tracker.face_registry:
                del self.gaze_tracker.face_registry[user_id]
            if user_id in self.gaze_tracker.models:
                del self.gaze_tracker.models[user_id]
            if user_id in self.gaze_tracker.user_colors:
                del self.gaze_tracker.user_colors[user_id]
            if user_id in self.gaze_tracker.face_embeddings:
                del self.gaze_tracker.face_embeddings[user_id]
            if user_id in self.gaze_tracker.smoothed_points:
                del self.gaze_tracker.smoothed_points[user_id]
                
            # Remove from UI
            if user_id in self.user_entries:
                self.user_entries[user_id]['frame'].destroy()
                del self.user_entries[user_id]
                
            # Save changes
            self.gaze_tracker.save_user_data()
            self.update_status()
                
    def update_user_status(self, user_id, status, color=None):
        """Update user status in the list"""
        if user_id in self.user_entries:
            status_color = '#2ed573' if status == "Calibrated" else '#ffa502' if status == "Registered" else '#888'
            self.user_entries[user_id]['status_label'].config(text=status, fg=status_color)
            if color:
                self.user_entries[user_id]['color_indicator'].config(fg=color)
                
            # Update action buttons based on status
            action_frame = self.user_entries[user_id]['action_frame']
            for widget in action_frame.winfo_children():
                widget.destroy()
                
            if status != "Calibrated":
                calibrate_btn = tk.Button(
                    action_frame,
                    text="Cal",
                    bg='#ffa502',
                    fg='white',
                    font=('Arial', 8, 'bold'),
                    width=4,
                    command=lambda uid=user_id: self.calibrate_single_user(uid)
                )
                calibrate_btn.pack(side='left', padx=1)
            
            delete_btn = tk.Button(
                action_frame,
                text="×",
                bg='#ff4757',
                fg='white',
                font=('Arial', 8, 'bold'),
                width=3,
                command=lambda uid=user_id: self.delete_user(uid)
            )
            delete_btn.pack(side='left', padx=1)
                
    def clear_user_list(self):
        """Clear all users from the list"""
        for user_id in list(self.user_entries.keys()):
            self.user_entries[user_id]['frame'].destroy()
        self.user_entries.clear()
        
    def update_status(self):
        """Update system status display"""
        if hasattr(self.gaze_tracker, 'get_status'):
            status = self.gaze_tracker.get_status()
            self.active_users_label.config(text=f"Active Users: {status.get('active_users', 0)}")
            self.calibrated_users_label.config(text=f"Calibrated Users: {status.get('calibrated_users', 0)}")
            self.fps_label.config(text=f"FPS: {status.get('fps', 0):.1f}")
            
        # Schedule next update
        if self.gaze_tracker.is_tracking:
            self.root.after(1000, self.update_status)
            
    def start_camera_preview(self):
        """Start camera preview in a separate thread"""
        self.preview_running = True
        if self.preview_thread is None or not self.preview_thread.is_alive():
            self.preview_thread = threading.Thread(target=self.update_preview)
            self.preview_thread.daemon = True
            self.preview_thread.start()
            
    def stop_camera_preview(self):
        """Stop camera preview"""
        self.preview_running = False
        self.preview_label.config(image='', text="Camera preview stopped")
        
    def update_preview(self):
        """Update camera preview"""
        while self.preview_running and self.gaze_tracker.is_tracking:
            try:
                frame = self.gaze_tracker.get_current_frame()
                if frame is not None:
                    # Resize frame for preview
                    preview_frame = cv2.resize(frame, (320, 240))
                    preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(Image.fromarray(preview_frame))
                    
                    # Update label in main thread
                    self.root.after(0, lambda: self.preview_label.config(image=photo, text=''))
                    self.root.after(0, lambda: setattr(self.preview_label, 'image', photo))
                    
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Preview update error: {e}")
                break
                
    def show_calibration_window(self):
        """Show calibration progress window"""
        cal_window = tk.Toplevel(self.root)
        cal_window.title("Calibration in Progress")
        cal_window.geometry("500x300")
        cal_window.configure(bg='#1a1a2e')
        cal_window.transient(self.root)
        cal_window.grab_set()
        
        # Center the window
        cal_window.geometry("+{}+{}".format(
            int(self.root.winfo_x() + self.root.winfo_width()/2 - 250),
            int(self.root.winfo_y() + self.root.winfo_height()/2 - 150)
        ))
        
        tk.Label(
            cal_window,
            text="🎯 Calibration in Progress",
            font=('Arial', 18, 'bold'),
            fg='#00d4ff',
            bg='#1a1a2e'
        ).pack(pady=20)
        
        instruction_frame = tk.Frame(cal_window, bg='#16213e', relief='raised', bd=2)
        instruction_frame.pack(fill='x', padx=20, pady=10)
        
        instructions = [
            "👀 Look at each yellow dot as it appears",
            "⏱️ Hold your gaze steady for 2 seconds per dot",
            "🎯 9 calibration points will be shown",
            "✨ This improves tracking accuracy significantly"
        ]
        
        for instruction in instructions:
            tk.Label(
                instruction_frame,
                text=instruction,
                font=('Arial', 11),
                fg='white',
                bg='#16213e',
                justify='left'
            ).pack(anchor='w', padx=15, pady=5)
        
        progress = ttk.Progressbar(cal_window, mode='indeterminate')
        progress.pack(pady=20, padx=40, fill='x')
        progress.start()
        
        status_label = tk.Label(
            cal_window,
            text="Preparing calibration...",
            font=('Arial', 12),
            fg='#ffa502',
            bg='#1a1a2e'
        )
        status_label.pack(pady=10)
        
        def close_cal_window():
            progress.stop()
            cal_window.destroy()
            
        def update_calibration_status():
            if hasattr(self.gaze_tracker, 'current_point_idx') and hasattr(self.gaze_tracker, 'calibration_points'):
                if self.gaze_tracker.is_calibrating:
                    current = self.gaze_tracker.current_point_idx + 1
                    total = len(self.gaze_tracker.calibration_points)
                    status_label.config(text=f"Calibrating... Point {current} of {total}")
                    cal_window.after(500, update_calibration_status)
                else:
                    status_label.config(text="Calibration completed!")
                    cal_window.after(2000, close_cal_window)
        
        # Start status updates
        cal_window.after(1000, update_calibration_status)
        
        # Auto close after calibration
        cal_window.protocol("WM_DELETE_WINDOW", close_cal_window)
        
    def run(self):
        """Start the UI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        if self.gaze_tracker.is_tracking:
            self.stop_tracking()
        self.preview_running = False
        if self.face_capture_window:
            self.face_capture_window.destroy()
        self.root.destroy()

# Example usage
if __name__ == "__main__":
    # This would be called from your main application
    pass