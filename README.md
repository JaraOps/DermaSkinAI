# DermaSkinAI 🩺🤖

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4)
![License](https://img.shields.io/badge/License-MIT-green)

> **Instant AI Skin Analysis for Early Detection.**
> *Tu primera opinión médica en segundos, no en meses.*

🌐 **Live Demo:** [www.dermaskinai.tech](https://www.dermaskinai.tech)

---

## 📖 About the Project

**DermaSkinAI** is a hybrid AI web application designed to act as a preliminary triage tool for skin lesions.

### 🌵 The Inspiration
I come from **Antofagasta, Chile**, a region known for the Atacama Desert and having some of the **highest solar radiation levels on Earth**. In my hometown, skin cancer is a daily reality, yet getting an appointment with a dermatologist can take **3 to 6 months**.

We built DermaSkinAI to bridge this gap: reducing patient anxiety and providing immediate, actionable guidance while they wait for a specialist.

---

## ⚙️ How it Works (The Hybrid Engine)

Unlike standard classifiers that just output a label, we use a **Hybrid AI Approach**:

1. **The Eye (Computer Vision):** A **TensorFlow/Keras** model (based on MobileNetV2) analyzes the lesion's visual patterns to detect 7 common pathologies (Melanoma, Nevus, etc.).
2. **The Brain (LLM Reasoning):** The raw probability data is sent to **Google Gemini**.
3. **The Output:** Gemini acts as a "Medical Translator," generating an empathetic, easy-to-understand explanation with triage recommendations.

```mermaid
graph LR
    A[📷 User Photo] -->|Upload| B(🐍 Flask Server)
    subgraph "Hybrid AI Engine"
    B -->|1. Vision Analysis| C{👁️ TensorFlow}
    C -->|Class Probabilities| D[🧠 Google Gemini]
    D -->|Medical Context & Triage| D
    end
    D -->|2. Final Report| E[📱 Client App]
    style C fill:#ff9900,stroke:#333,stroke-width:2px
    style D fill:#4285F4,stroke:#333,stroke-width:2px,color:white
