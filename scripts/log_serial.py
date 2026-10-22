import serial
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
# CHANGE THIS TO YOUR PORT (e.g., 'COM3' or 'COM4')
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200

# The output file will be saved in the data/synthetic/ folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "synthetic" / "real_temperature_test.csv"

# =============================================================================
# Main Loop
# =============================================================================
def main():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for the ESP32 to reset
        
        print(f"Logging data to: {OUTPUT_FILE}")
        print("Press Ctrl+C to stop logging.\n")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Write the header
            f.write("timestamp_ms,temperature\n")
            
            while True:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8').strip()
                
                # Ignore empty lines or the header line from the ESP32
                if line and not line.startswith("timestamp"):
                    print(line) # Print to console so you see it working
                    f.write(line + "\n") # Write to the file
                    f.flush() # Force save to disk immediately
                    
    except serial.SerialException as e:
        print(f"\nERROR: Could not open {SERIAL_PORT}.")
        print("Make sure the Serial Monitor in PlatformIO is CLOSED.")
        print(f"Details: {e}")
    except KeyboardInterrupt:
        print("\n\nLogging stopped by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()