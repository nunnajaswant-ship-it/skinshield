# Hardware

SkinShield runs on a commodity edge board with a camera, a microphone and four skin/environment sensors.

## Board

| Item | Specification |
|---|---|
| Board | **Raspberry Pi 4B** (4 GB or 8 GB) |
| Alternative | NVIDIA Jetson Orin Nano (higher throughput) |
| Camera | Raspberry Pi Camera Module / USB camera |
| Microphone | USB microphone or I2S MEMS microphone |
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
├── GPIO  → temperature sensor (data)
├── I2C   → BH1750 (light), TCS34725 (colour)
├── ADC   → capacitive moisture sensor
├── CSI   → camera module
└── USB   → microphone
```

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
| `raspberry_pi_sensor_rig.jpeg` | Raspberry Pi 4B with the four labelled sensors |
| `esp32_sensor_rig.jpeg` | Earlier ESP32 prototype on a perfboard |
| `dashboard.jpeg` | Live Flask dashboard running on the Pi |

## Status

- [x] Sensor rig assembled and reading live data
- [x] Flask dashboard operational
- [ ] Camera + microphone integrated into the same enclosure
- [ ] Model deployed on-device (ONNX / TensorRT)
- [ ] Field pilot
