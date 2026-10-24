from pathlib import Path
import binascii

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TFLITE_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder_int8.tflite"
HEADER_PATH = PROJECT_ROOT / "src" / "inference" / "src" / "model_data.h"

# Read the tflite file
with open(TFLITE_PATH, 'rb') as f:
    model_data = f.read()

# Generate the C++ header content
header_content = f"""// Auto-generated from {TFLITE_PATH.name}
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char model_data[] = {{
"""

# Write bytes as hex
for i in range(0, len(model_data), 12):
    chunk = model_data[i:i+12]
    hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
    header_content += f"    {hex_str},\n"

header_content += f"""}};

const unsigned int model_data_len = {len(model_data)};

#endif // MODEL_DATA_H
"""

# Write the header file
with open(HEADER_PATH, 'w') as f:
    f.write(header_content)

print(f"Generated {HEADER_PATH.name} ({len(model_data)} bytes)")