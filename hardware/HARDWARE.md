# Hardware

SkinShield runs on a commodity edge board with a camera, a microphone and four skin/environment sensors.

## Board

| Item | Specification |
|---|---|
| Board | **Raspberry Pi 4B** (4 GB or 8 GB) |
| Alternative | NVIDIA Jetson Orin Nano (higher throughput) |
| Camera | Raspberry Pi Camera Module (e.g. Camera Module 3) on the CSI ribbon |
| Microphone | Analog **MIC module** (VCC / GND / OUT) on the ADC, or a USB microphone |
| Power | 5 V / 3 A USB-C supply, battery pack for field use |

## Sensors

| Sensor | Model (example) | Interface | Measures |
|---|---|---|---|
| Temperature | DHT22 / DS18B20 / MLX90614 | GPIO / I2C | Skin and ambient temperature |
| Moisture | Capacitive soil-moisture sensor (repurposed for skin surface) | ADC | Surface hydration |
| Light | BH1750 | I2C | Incident light intensity (lux) |
| Colour | TCS34725 | I2C | Red / green / blue reflectance |

## Wiring summary

```
Raspberry Pi 4B
├── CSI   → Camera Module (image input)
├── GPIO  → temperature sensor (data)
├── I2C   → BH1750 (light), TCS34725 (colour)
├── ADC   → capacitive moisture sensor, MIC module OUT (text input)
└── GND/VCC rail → all modules
```

All modules sit on a single breadboard, so the whole rig is one self-contained unit.

## Dashboard

A **Flask** web dashboard runs on the Pi and displays live readings:

| Metric | Example reading |
|---|---|
| Overall skin condition | Poor circulation |
| Temperature | 31.90 °C (LOW) |
| Light | 25.83 lux (LOW LIGHT) |
| Redness (R:G:B) | 45 : 45 : 45 (NORMAL) |
| Moisture | MOIST |

The dashboard is served locally, so it works without an internet connection.

## Why these sensors

Each sensor captures a signal that is clinically relevant but invisible in a single photograph:

- **Temperature** — local inflammation, infection and circulation.
- **Moisture** — barrier integrity, eczema and xerosis.
- **Light** — UV exposure and lesion visibility.
- **Colour (RGB)** — erythema, pigmentation and blanching.

Together with the photograph and the patient's description, they form the three-modality input that the model consumes.

## Photos

| File | Shows |
|---|---|
| `raspberry_pi_sensor_rig.png` | Full annotated rig: Raspberry Pi 4B, Camera Module, MIC module, moisture, temperature, light and colour sensors |
| `esp32_sensor_rig.jpeg` | Earlier ESP32 prototype on a perfboard |
| `dashboard.jpeg` | Live Flask dashboard running on the Pi |

## Status

- [x] Sensor rig assembled and reading live data
- [x] Camera Module and MIC module wired to the same Pi (bench setup)
- [x] Flask dashboard operational
- [ ] Camera + microphone integrated into a single enclosure
- [ ] Model deployed on-device (ONNX / TensorRT)
- [ ] Field pilot
