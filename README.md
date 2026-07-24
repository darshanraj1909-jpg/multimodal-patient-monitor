# Multimodal AI Patient Monitor

A real-time, browser-based dashboard for intelligent patient monitoring. This system captures live video from a standard webcam, streams it over WebSockets to a FastAPI backend, and uses Ultralytics YOLO models to instantly detect patient posture and environmental hazards (fire/smoke).

## Features

* **Zero-Latency Streaming:** Uses WebSockets (via FastAPI) to stream video frames without the overhead of HTTP requests.
* **Posture Recognition:** Detects patient states (e.g., sitting, standing, sleeping on bed) and flags dangerous states like "lying on floor".
* **Hazard Detection:** Constantly monitors the environment for fire and smoke.
* **Clinical Dashboard:** A dark-themed, responsive UI with glassmorphism effects and pulsating red alerts for immediate visual feedback during emergencies.
* **Edge-Ready:** Designed to run locally on a laptop camera or easily scale to edge AI hardware.

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Backend** | Python & FastAPI | High-performance asynchronous web server |
| **Communication**| WebSockets | Real-time bi-directional data streaming |
| **Computer Vision**| Ultralytics YOLO | Object detection and classification |
| **Frontend** | HTML5, CSS3, JS | Live video capture and alert dashboard |
| **Image Processing**| OpenCV & NumPy | Base64 frame decoding and matrix manipulation |

## Prerequisites

* Python 3.8+
* A webcam connected to your machine
* Trained YOLO models for your specific use cases (`.pt` format)

## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/darshanraj1909-jpg/multimodal-patient-monitor.git](https://github.com/darshanraj909-jpg/multimodal-patient-monitor.git)
cd multimodal-patient-monitor
