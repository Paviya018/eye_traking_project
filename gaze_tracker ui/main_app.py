#!/usr/bin/env python3
"""
Multi-User Eye Gaze Tracker - Enhanced Version
Main Application Entry Point

Key Improvements:
- Smart face recognition that remembers users between sessions
- Enhanced UI with user management and individual calibration
- Names displayed alongside gaze dots for easy identification
- Automatic skip of registration for already registered faces
- Better user feedback and status tracking

This application provides a comprehensive eye gaze tracking system with:
- Multi-user support with automatic face recognition
- Individual calibration for each user with improved accuracy
- Real-time gaze tracking with unique colored dots AND names
- Modern UI with user management and individual controls
- Data persistence across sessions with face embeddings
- Smart registration flow that avoids duplicate registrations

Required Dependencies:
- opencv-python
- mediapipe
- scikit-learn
- numpy
- pillow
- tkinter (usually comes with Python)

Usage:
    python main_app.py

New Features in this Version:
1. Names appear next to gaze dots for easy user identification
2. Smart face recognition prevents duplicate registration dialogs
3. Already registered faces skip registration and go directly to tracking
4. Enhanced UI with individual user controls and better status display
5. Improved calibration process with better visual feedback
6. Face embeddings for more robust user recognition
7. Individual user calibration and deletion options
"""

import sys
import os
import traceback
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'PIL': 'pillow'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def print_features():
    """Print the new features of this enhanced version"""
    print("\n🌟 Enhanced Features:")
    print("=" * 50)
    print("✨ Smart Face Recognition")
    print("   • Automatically remembers users between sessions")
    print("   • Skips registration for already registered faces")
    print("   • Uses face embeddings for robust identification")
    
    print("\n👤 Enhanced User Experience")
    print("   • Names displayed next to gaze dots")
    print("   • Individual user calibration options")
    print("   • User deletion and management controls")
    print("   • Better status tracking and feedback")
    
    print("\n🎯 Improved Tracking")
    print("   • More accurate calibration process")
    print("   • Smoother gaze point tracking")
    print("   • Better handling of multiple users")
    print("   • Enhanced visual feedback during calibration")
    
    print("\n💾 Data Persistence")
    print("   • User data saved across sessions")
    print("   • Face embeddings stored for recognition")
    print("   • Calibration data preserved")

def main():
    """Main application entry point"""
    print("🎯 Enhanced Multi-User Eye Gaze Tracker")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    # Print new features
    print_features()
    
    try:
        # Import after dependency check
        from gaze_tracker_logic import EyeGazeTracker
        from gaze_tracker_ui import EyeGazeTrackerUI
        
        print("\n🚀 Starting Enhanced Eye Gaze Tracker...")
        
        # Create tracker instance
        tracker = EyeGazeTracker()
        
        # Create UI instance
        ui = EyeGazeTrackerUI(tracker)
        tracker.set_ui(ui)
        
        # Load existing users into UI
        loaded_users = 0
        for face_id, (_, _, name, _) in tracker.face_registry.items():
            # Only show properly registered users (not temporary User_X names)
            if not name.startswith("User_"):
                color = tracker.user_colors.get(face_id, '#00ff00')
                status = "Calibrated" if face_id in tracker.models else "Registered"
                ui.add_user_to_list(face_id, name, status, color)
                loaded_users += 1
        
        if loaded_users > 0:
            print(f"✅ Loaded {loaded_users} registered users from previous session")
        else:
            print("ℹ️ No previous users found - start tracking to register new faces")
        
        print("\n📋 Quick Start Guide:")
        print("=" * 50)
        print("1. 🚀 Click 'Start Tracking' to begin face detection")
        print("2. 👤 New faces will prompt for registration (enter names)")
        print("3. 🎯 Use 'Calibrate Users' to improve tracking accuracy")
        print("4. 👀 Watch gaze dots with names appear on screen")
        print("5. 💾 System remembers users for next session")
        
        print("\n🔧 Pro Tips:")
        print("• Ensure good lighting for better face detection")
        print("• Look directly at calibration dots for best accuracy")
        print("• Use individual 'Cal' buttons for specific user calibration")
        print("• Delete users with '×' button if no longer needed")
        
        print(f"\n✅ System initialized successfully with {loaded_users} users")
        print("Starting UI...")
        
        # Start the UI
        ui.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nMake sure all files are in the same directory:")
        print("   - main_app.py (this file)")
        print("   - gaze_tracker_logic.py") 
        print("   - gaze_tracker_ui.py")
        print("\nAnd that you're using the updated versions of these files.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nFull error traceback:")
        traceback.print_exc()
        
    finally:
        print("\n👋 Enhanced Eye Gaze Tracker closed")
        print("Thank you for using the improved system!")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()