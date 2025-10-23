"""
Advanced Virtual Mouse Control using Hand Gestures
Гарын хөдөлгөөнөөр хулганыг удирдах систем - Advanced Edition
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Configure PyAutoGUI
pyautogui.FAILSAFE = False  # Disable failsafe
pyautogui.PAUSE = 0  # No pause between actions

# Camera settings
cam_width, cam_height = 640, 480

# Smoothing variables
prev_x, prev_y = 0, 0
smoothing = 7  # Higher = smoother but slower

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# Screen size
screen_width, screen_height = pyautogui.size()
cam_width, cam_height = 640, 480

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_angle(a, b, c):
    """Calculate angle between three points (from GitHub repo)"""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

class VirtualMouse:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)
        
        # Position tracking
        self.prev_x = 0
        self.prev_y = 0
        self.click_cooldown = 0
        self.last_gesture = "none"
        self.last_finger_status = [0, 0, 0, 0, 0]  # For display
        
        # Advanced features
        self.gesture_history = deque(maxlen=10)  # Last 10 gestures
        self.fps_history = deque(maxlen=30)  # For FPS calculation
        self.confidence_scores = deque(maxlen=30)
        self.gesture_start_time = {}
        self.gesture_durations = {}
        
        # Statistics
        self.total_clicks = 0
        self.total_moves = 0
        self.total_double_clicks = 0  # NEW!
        self.total_drags = 0  # NEW!
        self.session_start = time.time()
        
        # Drag state tracking (SIMPLE!)
        self.is_dragging = False
        self.drag_start_pos = None
        
        # For display
        self.last_finger_status = [0, 0, 0, 0, 0]
        
        # Settings
        self.show_advanced_info = True
        self.show_trails = True
        self.trail_points = deque(maxlen=20)
        
        # Color schemes (SIMPLIFIED!)
        self.colors = {
            'move': (0, 255, 0),           # Green - 1 finger
            'left_click': (0, 255, 255),   # Cyan - 2 fingers
            'drag_start': (255, 0, 255),   # Magenta - 3 fingers (start)
            'drag_hold': (255, 0, 255),    # Magenta - 3 fingers (holding)
            'double_click': (255, 255, 0), # Yellow - Thumb up
            'stop': (128, 128, 128),       # Gray - Open hand
            'none': (255, 255, 255)        # White
        }
        
    def get_finger_status(self, landmarks):
        """Check which fingers are extended"""
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_status = []
        
        for tip in finger_tips:
            # Check if fingertip is above the knuckle
            if landmarks[tip].y < landmarks[tip - 2].y:
                finger_status.append(1)  # Extended
            else:
                finger_status.append(0)  # Folded
                
        # Check thumb separately
        if landmarks[4].x < landmarks[3].x:  # Thumb
            finger_status.insert(0, 1)
        else:
            finger_status.insert(0, 0)
            
        return finger_status
    
    def detect_gesture(self, landmarks):
        """Detect specific hand gestures - SIMPLIFIED & PRACTICAL!"""
        finger_status = self.get_finger_status(landmarks)
        
        # Save for display on screen
        self.last_finger_status = finger_status
        
        # 1 finger (index only) = Move cursor
        if finger_status == [0, 1, 0, 0, 0]:
            return "move"
        
        # 2 fingers (index + middle) = Left Click
        elif finger_status == [0, 1, 1, 0, 0]:
            return "left_click"
        
        # 3 fingers (index + middle + ring) = Drag mode
        elif finger_status == [0, 1, 1, 1, 0]:
            if self.is_dragging:
                return "drag_hold"
            else:
                return "drag_start"
        
        # Thumb up only = Double Click (keep this for file opening)
        elif finger_status == [1, 0, 0, 0, 0]:
            return "double_click"
        
        # All fingers extended = Stop/No action
        elif finger_status == [1, 1, 1, 1, 1]:
            return "stop"
        
        return "none"

    def draw_finger_status_overlay(self, frame):
        """Draw a compact finger-status overlay (always visible)."""
        if not hasattr(self, 'last_finger_status'):
            return

        # Small background box
        h, w = frame.shape[:2]
        box_w, box_h = 220, 120
        x0, y0 = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Draw each finger status line
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for i, (name, status) in enumerate(zip(finger_names, self.last_finger_status)):
            status_text = "UP" if status == 1 else "DOWN"
            status_color = (0, 220, 0) if status == 1 else (180, 180, 180)
            y = y0 + 22 + i * 20
            cv2.putText(frame, f"{name}: ", (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, status_text, (x0 + 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        # Show array explicitly
        arr_text = str(self.last_finger_status)
        cv2.putText(frame, f"Array: {arr_text}", (x0 + 8, y0 + box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    def is_pinching(self, landmarks, finger1_tip, finger2_tip):
        """Check if two fingers are pinching (touching)"""
        tip1 = landmarks[finger1_tip]
        tip2 = landmarks[finger2_tip]
        
        distance = np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2)
        
        return distance < 0.05  # Threshold for pinching
    
    def move_cursor(self, index_finger):
        """Move cursor based on index finger position"""
        # Convert normalized coordinates to screen coordinates
        x = int(index_finger.x * screen_width)
        y = int(index_finger.y * screen_height)
        
        # Apply smoothing
        curr_x = self.prev_x + (x - self.prev_x) / smoothing
        curr_y = self.prev_y + (y - self.prev_y) / smoothing
        
        # Move mouse
        pyautogui.moveTo(curr_x, curr_y, duration=0)
        
        # Update previous position
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        return int(curr_x), int(curr_y)
    
    def run(self):
        """Main loop with advanced features"""
        print("🖱️ Advanced Virtual Mouse Control Started!")
        print("📹 Camera feed opening...")
        print("\n🖐️ Gestures (IMPROVED - Илүү амархан!):")
        print("  ☝️  Index finger up = Move cursor")
        print("  ✊ Fist (all closed) = Left Click (NEW - EASY!)")
        print("  ✌️  Peace sign (2 fingers) = Right Click (NEW!)")
        print("  🖐️  Open hand (5 fingers) = Stop")
        print("  💡 NO MORE PINCHING - Way easier!")
        print("\n⌨️  Keyboard Shortcuts:")
        print("  'q' = Quit")
        print("  'i' = Toggle info display")
        print("  't' = Toggle trails")
        print("  's' = Save screenshot")
        print("  'r' = Reset statistics\n")
        
        print("🎮 NEW SIMPLE GESTURES:")
        print("  ☝️  1 finger (index) = Move cursor")
        print("  ✌️  2 fingers (index+middle) = Left Click")
        print("  🎯 3 fingers (index+middle+ring) = DRAG & DROP")
        print("  👍 Thumb up = Double Click (open files)")
        print("  🖐️  Open hand (5 fingers) = Stop")
        print("  💡 SUPER SIMPLE & PRACTICAL!\n")
        
        frame_count = 0
        
        while True:
            frame_start = time.time()
            success, frame = self.cap.read()
            if not success:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            gesture = "none"
            cursor_pos = None
            hand_confidence = 0
            
            # Draw hand landmarks and detect gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get confidence score
                    if results.multi_handedness:
                        hand_confidence = results.multi_handedness[0].classification[0].score
                        self.confidence_scores.append(hand_confidence)
                    
                    # Draw enhanced landmarks
                    self.draw_enhanced_landmarks(frame, hand_landmarks)
                    
                    # Get landmarks
                    landmarks = hand_landmarks.landmark
                    
                    # Detect gesture
                    gesture = self.detect_gesture(landmarks)
                    
                    # Update gesture history
                    self.gesture_history.append(gesture)
                    
                    # Track gesture duration
                    if gesture != self.last_gesture:
                        if self.last_gesture in self.gesture_start_time:
                            duration = time.time() - self.gesture_start_time[self.last_gesture]
                            if self.last_gesture not in self.gesture_durations:
                                self.gesture_durations[self.last_gesture] = []
                            self.gesture_durations[self.last_gesture].append(duration)
                        self.gesture_start_time[gesture] = time.time()
                    
                    # Handle gestures
                    if gesture in ["move"]:
                        index_finger = landmarks[8]  # Index finger tip
                        cursor_pos = self.move_cursor(index_finger)
                        self.total_moves += 1
                        
                        # Add to trail
                        if self.show_trails:
                            finger_x = int(index_finger.x * cam_width)
                            finger_y = int(index_finger.y * cam_height)
                            self.trail_points.append((finger_x, finger_y))
                        
                    elif gesture == "left_click" and self.click_cooldown == 0:
                        # 2 fingers = Left Click (SIMPLE!)
                        pyautogui.click()
                        self.click_cooldown = 15
                        self.total_clicks += 1
                        print(f"✌️ 2-Finger Click! (Total: {self.total_clicks})")
                        # Flash effect - cyan
                        cv2.circle(frame, (cam_width//2, cam_height//2), 60, (255, 255, 0), -1)
                    
                    elif gesture == "drag_start":
                        # 3 fingers = Start dragging immediately!
                        if not self.is_dragging:
                            pyautogui.mouseDown()
                            self.is_dragging = True
                            self.total_drags += 1
                            self.drag_start_pos = pyautogui.position()
                            print(f"🎯 3-Finger DRAG START! Keep 3 fingers, move to drag!")
                            # Flash effect - magenta
                            cv2.circle(frame, (cam_width//2, cam_height//2), 70, (255, 0, 255), -1)
                        
                    elif gesture == "drag_hold":
                        # Continue dragging with 3 fingers
                        if self.is_dragging:
                            cv2.putText(frame, "DRAGGING... (3 FINGERS)", (cam_width//2 - 150, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                    
                    elif gesture == "move":
                        # 1 finger = Move cursor (and release drag if active)
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            drag_end = pyautogui.position()
                            if self.drag_start_pos:
                                distance = int(np.sqrt((drag_end[0]-self.drag_start_pos[0])**2 + 
                                                      (drag_end[1]-self.drag_start_pos[1])**2))
                                print(f"🎯 DRAG END! Dropped at 1-finger. Distance: {distance}px")
                            self.drag_start_pos = None
                    
                    elif gesture == "double_click" and self.click_cooldown == 0:
                        pyautogui.doubleClick()
                        self.click_cooldown = 25
                        self.total_double_clicks += 1
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                        print(f"👍 Thumb Double Click! (Total: {self.total_double_clicks})")
                        # Flash effect - yellow burst!
                        cv2.circle(frame, (cam_width//2, cam_height//2), 80, (0, 255, 255), -1)
                    
                    elif gesture == "stop":
                        # Stop everything - release if dragging
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            print("🖐️ Stop - Drag cancelled!")
                    
                    self.last_gesture = gesture
            
            else:
                # No hand detected - release drag if active
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                    print("👋 Hand lost - Drag released")
            
            # Decrease click cooldown
            if self.click_cooldown > 0:
                self.click_cooldown -= 1
            
            # Draw trails
            if self.show_trails and len(self.trail_points) > 1:
                for i in range(1, len(self.trail_points)):
                    alpha = i / len(self.trail_points)
                    thickness = int(2 + alpha * 3)
                    cv2.line(frame, self.trail_points[i-1], self.trail_points[i], 
                            self.colors.get(gesture, (255, 255, 255)), thickness)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Draw advanced info overlay
            if self.show_advanced_info:
                self.draw_info_overlay(frame, gesture, cursor_pos, hand_confidence, avg_fps, frame_count)
            else:
                # Minimal info
                cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors.get(gesture, (255, 255, 255)), 2)
            
            # Draw gesture indicator circle
            self.draw_gesture_indicator(frame, gesture)

            # Draw compact finger-status overlay (always visible)
            self.draw_finger_status_overlay(frame)

            # Show frame
            cv2.imshow('Advanced Virtual Mouse - Гарын хулгана', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                self.show_advanced_info = not self.show_advanced_info
                print(f"ℹ️ Info display: {'ON' if self.show_advanced_info else 'OFF'}")
            elif key == ord('t'):
                self.show_trails = not self.show_trails
                print(f"🌟 Trails: {'ON' if self.show_trails else 'OFF'}")
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                self.reset_statistics()
                print("🔄 Statistics reset!")
        
        # Print session summary
        self.print_session_summary()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n✅ Virtual Mouse stopped")
    
    def draw_enhanced_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks with enhanced visuals"""
        # Draw connections
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            
            start_point = (int(start.x * cam_width), int(start.y * cam_height))
            end_point = (int(end.x * cam_width), int(end.y * cam_height))
            
            # Gradient line
            cv2.line(frame, start_point, end_point, (0, 255, 0), 3)
        
        # Draw landmarks with different colors
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * cam_width)
            y = int(landmark.y * cam_height)
            
            # Fingertips are larger
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            else:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    def draw_info_overlay(self, frame, gesture, cursor_pos, confidence, fps, frame_count):
        """Draw comprehensive information overlay"""
        overlay = frame.copy()
        
        # Semi-transparent background - BIGGER for finger status
        cv2.rectangle(overlay, (5, 5), (450, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 25
        line_height = 22
        
        # Gesture info
        color = self.colors.get(gesture, (255, 255, 255))
        cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        # FINGER STATUS - NEW! Show what fingers are detected
        if hasattr(self, 'last_finger_status'):
            finger_icons = ["👍", "☝️", "✌️", "💍", "🤙"]
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            
            cv2.putText(frame, "Finger Status:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_offset += line_height
            
            for i, (name, status) in enumerate(zip(finger_names, self.last_finger_status)):
                status_text = "UP" if status == 1 else "DOWN"
                status_color = (0, 255, 0) if status == 1 else (100, 100, 100)
                cv2.putText(frame, f"  {name}: {status_text}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                y_offset += 18
            
            # Show array format
            cv2.putText(frame, f"Array: {self.last_finger_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
        
        # Cursor position
        if cursor_pos:
            cv2.putText(frame, f"Cursor: ({cursor_pos[0]}, {cursor_pos[1]})", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height
        
        # Confidence
        conf_text = f"Confidence: {confidence*100:.1f}%"
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
        cv2.putText(frame, conf_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        # Statistics
        session_time = int(time.time() - self.session_start)
        cv2.putText(frame, f"Session: {session_time}s", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Clicks: {self.total_clicks}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Double: {self.total_double_clicks}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Drags: {self.total_drags}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Moves: {self.total_moves}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # Gesture history
        if len(self.gesture_history) > 0:
            recent = list(self.gesture_history)[-5:]
            history_text = " > ".join([g[:4] for g in recent])
            cv2.putText(frame, f"History: {history_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Controls reminder
        cv2.putText(frame, "Press 'i' to toggle info", (10, cam_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    def draw_gesture_indicator(self, frame, gesture):
        """Draw a visual indicator for current gesture"""
        radius = 40
        center_x = cam_width - radius - 20
        center_y = radius + 20
        
        color = self.colors.get(gesture, (255, 255, 255))
        
        # Pulsing effect
        pulse = int(10 * math.sin(time.time() * 5))
        current_radius = radius + pulse
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), current_radius, color, 3)
        cv2.circle(frame, (center_x, center_y), current_radius - 10, color, -1)
        
        # Draw gesture icon/text
        icon_map = {
            'move': '☝️',
            'left_click': '✊',
            'double_click': '�',
            'right_click': '✌️',
            'drag': '🎯',
            'stop': '🖐️',
            'none': '?'
        }
        
        icon = icon_map.get(gesture, '?')
        cv2.putText(frame, gesture[:4].upper(), (center_x - 20, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_clicks = 0
        self.total_moves = 0
        self.total_double_clicks = 0
        self.total_drags = 0
        self.session_start = time.time()
        self.gesture_durations.clear()
    
    def print_session_summary(self):
        """Print session statistics"""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        
        duration = int(time.time() - self.session_start)
        print(f"⏱️  Duration: {duration} seconds")
        print(f"🖱️  Total Clicks: {self.total_clicks}")
        print(f"👍 Double Clicks: {self.total_double_clicks}")
        print(f"🎯 Total Drags: {self.total_drags}")
        print(f"↔️  Total Moves: {self.total_moves}")
        
        if self.confidence_scores:
            avg_conf = sum(self.confidence_scores) / len(self.confidence_scores)
            print(f"📈 Avg Confidence: {avg_conf*100:.1f}%")
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"🎬 Avg FPS: {avg_fps:.1f}")
        
        print("\n🖐️ Gesture Usage:")
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.gesture_history)) * 100
            print(f"  {gesture:12s}: {count:4d} ({percentage:5.1f}%)")
        
        print("="*50)

if __name__ == "__main__":
    try:
        mouse = VirtualMouse()
        mouse.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
