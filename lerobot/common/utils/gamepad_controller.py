import hid
import time
import json
import os
import struct
from typing import Dict, List, Optional, Tuple


class ProController:
    VENDOR_ID = 0x057e   # Nintendo's vendor ID
    PRODUCT_ID = 0x2009  # Pro Controller's product ID

    def __init__(self, 
                 button_mapping_file="lerobot/common/utils/gamepad_calibration/controller_mapping.json", 
                 stick_calibration_file="lerobot/common/utils/gamepad_calibration/stick_calibration.json"):
        """Initialize Pro Controller with optional mapping files"""

        os.makedirs(os.path.dirname(button_mapping_file), exist_ok=True)
        os.makedirs(os.path.dirname(stick_calibration_file), exist_ok=True)
        self.dev = None
        self.button_mapping = None
        self.stick_calibration = None
        self.button_mapping_file = button_mapping_file
        self.stick_calibration_file = stick_calibration_file
        self._last_buttons = set()
        self._rumble_active = False

    def connect(self, force_calibration=False) -> bool:
        """Connect to the controller and load/perform calibration"""
        try:
            self.dev = hid.device()
            self.dev.open(self.VENDOR_ID, self.PRODUCT_ID)
            print(f"\nConnected to: {self.dev.get_product_string()}")
            
            # Initialize controller
            self._init_controller()
            self.dev.set_nonblocking(True)
            
            # Load or perform calibrations
            if not force_calibration:
                self.button_mapping = self._load_mapping()
                self.stick_calibration = self._load_stick_calibration()
            
            if not self.button_mapping:
                print("Performing button calibration...")
                self.button_mapping = self._calibrate_buttons()
                if not self.button_mapping:
                    return False
            
            if not self.stick_calibration:
                print("Performing stick calibration...")
                self.stick_calibration = self._calibrate_sticks()
                if not self.stick_calibration:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Safely disconnect the controller"""
        if self.dev:
            self.set_rumble(0)
            time.sleep(0.1)
            self.dev.close()
            self.dev = None

    def read_state(self) -> Dict:
        """Read and return the full controller state"""
        if not self.dev:
            return {}
        
        data = self.dev.read(64, 1)
        if not data:
            return self._last_state if hasattr(self, '_last_state') else {}
        
        # Get button states
        pressed_indices = self._get_pressed_buttons(data)
        pressed_indices = [idx for idx in pressed_indices if idx != 15]
        buttons = [
            self.button_mapping.get(idx, f"Unknown({idx})")
            for idx in pressed_indices
        ]
        
        # Get stick positions
        sticks = self._get_stick_values(data)
        if not sticks:
            sticks = {'left': {'x': 0, 'y': 0}, 'right': {'x': 0, 'y': 0}}
        
        # Create current state
        current_state = {
            'buttons': buttons,
            'sticks': sticks,
            'changed': False
        }
        
        # Check if state changed from last time
        if not hasattr(self, '_last_state'):
            current_state['changed'] = True
        else:
            last_buttons = set(self._last_state.get('buttons', []))
            current_buttons = set(buttons)
            last_sticks = self._last_state.get('sticks', {})
            
            # Check if buttons changed
            if last_buttons != current_buttons:
                current_state['changed'] = True
            
            # Check if sticks moved significantly (reduced threshold)
            threshold = 0.07
            for stick in ['left', 'right']:
                for axis in ['x', 'y']:
                    last_val = last_sticks.get(stick, {}).get(axis, 0)
                    current_val = sticks[stick][axis]
                    if abs(last_val - current_val) > threshold:
                        current_state['changed'] = True
        
        # Store current state
        self._last_state = current_state
        return current_state

    def set_rumble(self, intensity: int):
        """Set rumble intensity (0-255)"""
        if self.dev:
            rumble_data = [0x10, 0x00] + [intensity] * 8
            self.dev.write(rumble_data)
            self._rumble_active = intensity > 0

    def _init_controller(self):
        """Initialize the controller with rumble and input reports"""
        self.dev.write([0x01, 0x01])
        time.sleep(0.1)
        self.dev.write([0x01] + [0x00] * 9)
        time.sleep(0.1)

    def _get_pressed_buttons(self, data: List[int]) -> List[int]:
        """Get list of pressed button indices"""
        if len(data) < 3 or data[0] != 0x30:
            return []
        
        pressed = []
        for byte_idx in range(3, 6):
            if byte_idx >= len(data):
                continue
            byte = data[byte_idx]
            for bit in range(8):
                if byte & (1 << bit):
                    pressed.append((byte_idx - 3) * 8 + bit)
        return pressed

    def _get_stick_values(self, data: List[int]) -> Optional[Dict]:
        """Get calibrated stick values"""
        if len(data) < 12:
            return None
        
        lx = data[6] | ((data[7] & 0x0F) << 8)
        ly = (data[7] >> 4) | (data[8] << 4)
        rx = data[9] | ((data[10] & 0x0F) << 8)
        ry = (data[10] >> 4) | (data[11] << 4)
        
        if self.stick_calibration:
            return {
                'left': {
                    'x': (lx - self.stick_calibration['left']['center']['x']) / 
                         self.stick_calibration['left']['range']['x'],
                    'y': (ly - self.stick_calibration['left']['center']['y']) / 
                         self.stick_calibration['left']['range']['y']
                },
                'right': {
                    'x': (rx - self.stick_calibration['right']['center']['x']) / 
                         self.stick_calibration['right']['range']['x'],
                    'y': (ry - self.stick_calibration['right']['center']['y']) / 
                         self.stick_calibration['right']['range']['y']
                }
            }
        else:
            return {
                'left': {'x': (lx - 2048) / 2048.0, 'y': (ly - 2048) / 2048.0},
                'right': {'x': (rx - 2048) / 2048.0, 'y': (ry - 2048) / 2048.0}
            }

    def _load_mapping(self) -> Optional[Dict]:
        """Load button mapping from file"""
        if os.path.exists(self.button_mapping_file):
            try:
                with open(self.button_mapping_file, 'r') as f:
                    return {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                print(f"Error loading button mapping: {e}")
        return None

    def _load_stick_calibration(self) -> Optional[Dict]:
        """Load stick calibration from file"""
    
        if os.path.exists(self.stick_calibration_file):
            try:
                with open(self.stick_calibration_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading stick calibration: {e}")
        return None

    def _calibrate_buttons(self) -> Optional[Dict]:
        """Perform button calibration"""
        try:
            buttons_to_map = [
                'A', 'B', 'X', 'Y',
                'L', 'R', 'ZL', 'ZR',
                'Minus', 'Plus', 'Home', 'Capture',
                'L-Stick Press', 'R-Stick Press',
                'Up', 'Down', 'Left', 'Right'
            ]
            
            button_mapping = {}
            print("\nStarting button calibration...")
            print("Press Ctrl+C at any time to cancel.\n")
            
            for button in buttons_to_map:
                print(f"\nPress the {button} button...")
                
                # Wait for all buttons to be released
                time.sleep(0.5)
                while True:
                    pressed = self._get_pressed_buttons(self.dev.read(64, 1) or [])
                    if not pressed or all(idx == 15 for idx in pressed):
                        break
                    time.sleep(0.01)
                
                # Wait for a button press
                while True:
                    data = self.dev.read(64, 1)
                    if data:
                        pressed = self._get_pressed_buttons(data)
                        pressed = [idx for idx in pressed if idx != 15]
                        if pressed:
                            button_idx = pressed[0]
                            time.sleep(0.1)
                            
                            while True:
                                current = self._get_pressed_buttons(self.dev.read(64, 1) or [])
                                current = [idx for idx in current if idx != 15]
                                if not current:
                                    break
                                time.sleep(0.01)
                            
                            button_mapping[button_idx] = button
                            print(f"Mapped {button} to button index {button_idx}")
                            time.sleep(0.5)
                            break
                    time.sleep(0.01)
            
            # Save mapping
            with open(self.button_mapping_file, 'w') as f:
                json.dump(button_mapping, f, indent=2)
            print("\nButton mapping saved!")
            return button_mapping

        except KeyboardInterrupt:
            print("\nCalibration cancelled.")
            return None

    def _calibrate_sticks(self) -> Optional[Dict]:
        """Perform stick calibration"""
        try:
            calibration = {
                'left': {'center': {'x': 0, 'y': 0}, 'range': {'x': 0, 'y': 0}},
                'right': {'center': {'x': 0, 'y': 0}, 'range': {'x': 0, 'y': 0}}
            }

            print("\nCalibrating stick centers...")
            print("Leave both sticks centered and untouched for 5 seconds...")
            
            samples = []
            start_time = time.time()
            while time.time() - start_time < 5:
                data = self.dev.read(64, 1)
                if data and len(data) >= 12:
                    lx = data[6] | ((data[7] & 0x0F) << 8)
                    ly = (data[7] >> 4) | (data[8] << 4)
                    rx = data[9] | ((data[10] & 0x0F) << 8)
                    ry = (data[10] >> 4) | (data[11] << 4)
                    samples.append((lx, ly, rx, ry))
                time.sleep(0.01)

            if samples:
                lx_samples = sorted(s[0] for s in samples)
                ly_samples = sorted(s[1] for s in samples)
                rx_samples = sorted(s[2] for s in samples)
                ry_samples = sorted(s[3] for s in samples)
                
                n = len(samples)
                mid = n // 2
                
                calibration['left']['center'] = {
                    'x': lx_samples[mid],
                    'y': ly_samples[mid]
                }
                calibration['right']['center'] = {
                    'x': rx_samples[mid],
                    'y': ry_samples[mid]
                }
                
                print("Center calibration complete!")

            for stick in ['left', 'right']:
                print(f"\nCalibrating {stick} stick ranges...")
                print(f"Move the {stick} stick in circles at maximum range for 8 seconds...")
                print("Try to reach the edges in all directions...")
                
                samples_x = []
                samples_y = []
                
                start_time = time.time()
                while time.time() - start_time < 8:
                    data = self.dev.read(64, 1)
                    if data and len(data) >= 12:
                        if stick == 'left':
                            x = data[6] | ((data[7] & 0x0F) << 8)
                            y = (data[7] >> 4) | (data[8] << 4)
                        else:
                            x = data[9] | ((data[10] & 0x0F) << 8)
                            y = (data[10] >> 4) | (data[11] << 4)
                        
                        samples_x.append(x)
                        samples_y.append(y)
                    time.sleep(0.01)
                
                samples_x.sort()
                samples_y.sort()
                
                n = len(samples_x)
                p01_idx = max(0, int(n * 0.01))
                p99_idx = min(n - 1, int(n * 0.99))
                
                center_x = calibration[stick]['center']['x']
                center_y = calibration[stick]['center']['y']
                
                min_x = samples_x[p01_idx]
                max_x = samples_x[p99_idx]
                min_y = samples_y[p01_idx]
                max_y = samples_y[p99_idx]
                
                range_x = max(abs(max_x - center_x), abs(min_x - center_x))
                range_y = max(abs(max_y - center_y), abs(min_y - center_y))
                
                calibration[stick]['range'] = {'x': range_x, 'y': range_y}
                print(f"{stick} stick range calibration complete!")

            with open(self.stick_calibration_file, 'w') as f:
                json.dump(calibration, f, indent=2)
            print("\nStick calibration saved!")
            return calibration

        except KeyboardInterrupt:
            print("\nCalibration cancelled.")
            return None

# Example usage:
if __name__ == "__main__":
    controller = ProController()
    
    if not controller.connect():
        print("Failed to initialize controller")
        exit(1)
    
    print("\nController ready! Press Ctrl+C to exit.")
    print("- Move sticks to see positions")
    print("- Press buttons to see states")
    print("- Hold ZL+ZR for rumble\n")
    
    try:
        while True:
            state = controller.read_state()
            
            if state.get('changed', False):  # Only process if state changed
                # Show buttons if any are pressed
                if state['buttons']:
                    print(f"Buttons: {state['buttons']}")
                
                # Show sticks if they're moved significantly
                sticks = state['sticks']
                threshold = 0.08  # Reduced from 0.2 to 0.08
                
                lx, ly = sticks['left']['x'], sticks['left']['y']
                rx, ry = sticks['right']['x'], sticks['right']['y']
                
                if abs(lx) > threshold or abs(ly) > threshold:
                    print(f"Left Stick: x={lx:.2f}, y={ly:.2f}")
                if abs(rx) > threshold or abs(ry) > threshold:
                    print(f"Right Stick: x={rx:.2f}, y={ry:.2f}")
                
                # Handle rumble
                if 'ZL' in state['buttons'] and 'ZR' in state['buttons']:
                    controller.set_rumble(0xC0)
                else:
                    controller.set_rumble(0)
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        controller.disconnect()

if __name__ == "__main__":
    controller = ProController()
    controller.connect()
    print(controller.read_state())
    controller.disconnect()