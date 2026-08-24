<p align="center">
  <img src="Assets/alpha build banner.png" alt="EchoHands Banner" width="100%">
</p>

# EchoHands — Alpha Build

EchoHands is an AI-powered real-time American Sign Language (ASL) recognition system that uses a webcam to detect supported hand signs and convert recognized gestures into digital text.

The Alpha Build is the first complete working milestone of EchoHands. It combines hand detection, landmark processing, static sign recognition, dynamic gesture recognition, recognition control, and text building into one real-time application.

> **Alpha Build status:** The core recognition system is working and serves as the stable technical baseline for future development.

---

# ✨ What EchoHands Can Do

The current Alpha Build can:

- Detect a hand in real time through a webcam
- Extract hand landmarks using MediaPipe
- Recognize supported static ASL signs
- Recognize dynamic signs such as **J** and **Z**
- Analyze hand movement across multiple frames
- Filter and control predictions before adding them to text
- Prevent repeated recognition when a gesture is held
- Build words and text from accepted signs
- Support space, clear, and backspace controls

---

# Supported Signs

EchoHands currently works with the trained signs included in the project.

The system handles two main categories of gestures:

## Static Gestures

Static gestures are recognized primarily from the configuration or pose of the hand in a captured frame.

The current Alpha Build supports the trained static ASL alphabet and numeric signs available in the included model.

## Dynamic Gestures

Some ASL signs involve motion and cannot be reliably recognized from a single frame.

EchoHands currently includes dedicated sequence-based recognition for:

- **J**
- **Z**

These gestures are recognized by analyzing hand movement across multiple frames.

<p align="center">
  <img src="sign description/sign letters.png" alt="Supported ASL Signs" width="750">
</p>

<p align="center">
  <em>Figure 1. ASL alphabet and numeric signs supported by the EchoHands recognition system.</em>
</p>

---

# Installation & Requirements

## Recommended Python Version

The Alpha Build is recommended to run with:

```text
Python 3.10
```

Using the same Python version helps avoid dependency and compatibility issues.

## Clone the Repository

Clone the Alpha repository and move into the project directory:

```bash
git clone https://github.com/shivanshu43/EchoHands_Alpha-build.git
cd EchoHands
```

If the Alpha repository is private or access-restricted, permission from the project owner is required.

## Create a Virtual Environment

### Windows

```bash
py -3.10 -m venv venv
```

If Python 3.10 is already the active interpreter:

```bash
python -m venv venv
```

## Activate the Virtual Environment

### Windows Command Prompt

```bash
venv\Scripts\activate
```

Verify the active Python interpreter:

```bash
where python
```

Also verify the Python version:

```bash
python --version
```

## Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The exact Python dependencies and versions are defined in:

```text
requirements.txt
```

---

# How to Operate EchoHands

Start EchoHands from the project root:

```bash
python -m src.app
```

If the environment, dependencies, models, and camera access are correctly configured, EchoHands will initialize the recognition system and open the real-time application window.

<!-- Replace the image path below with the final Alpha UI screenshot stored in Assets/. -->
<p align="center">
  <img src="Assets/alpha UI.png" alt="EchoHands Alpha Interface" width="650">
</p>

<p align="center">
  <em>Figure 2. EchoHands Alpha Build real-time recognition interface.</em>
</p>

## Understanding the Interface

The EchoHands Alpha interface provides a real-time view of the recognition process. As the user interacts with the system through hand gestures and keyboard controls, the interface continuously updates to show what EchoHands is detecting, predicting, processing, and finally accepting.

The different sections are connected to the user's interaction with the system rather than being independent displays.

### Mode

```text
Mode: NONE
```

The **Mode** indicator shows the recognition mode currently active in EchoHands.

When the application is idle or waiting for a new gesture, it displays:

```text
NONE
```

As the user performs gestures, EchoHands can switch its internal recognition behaviour depending on the type of gesture being processed. This is especially relevant for motion-based signs, where the system needs to collect and analyze a sequence of frames instead of relying on a single hand pose.

### Prediction

```text
Prediction: No Hand Detected
```

The **Prediction** area shows what EchoHands is currently detecting or predicting.

When the user is not showing a hand to the webcam, the interface displays:

```text
No Hand Detected
```

As soon as a hand becomes visible, EchoHands detects the hand and processes its landmarks. If the gesture can be recognized, this area updates with the current predicted sign.

Because the camera continuously processes frames, the prediction can change while the user moves their hand or transitions between gestures.

### Confidence

```text
Confidence: 0.0%
```

The **Confidence** value shows how strongly the recognition model supports the current prediction.

As the user presents a gesture, the confidence value updates according to the model's current prediction. A stable and clearly performed sign is expected to produce a stronger prediction than an unclear or transitioning hand pose.

However, EchoHands does not automatically add every prediction to the text output. The prediction must also satisfy the application's recognition-control logic.

### Sequence Frames

```text
Sequence frames: 0
```

The **Sequence Frames** indicator shows how many consecutive frames are currently being collected for dynamic gesture analysis.

This becomes important when the user performs a motion-based gesture. Unlike a static sign, which can be recognized mainly from the current hand configuration, a dynamic sign requires EchoHands to observe movement over time.

As the user performs the gesture, frames are collected into a sequence. Once the sequence contains the required information, it can be analyzed by the dynamic recognition pipeline.

### Recognition State

```text
NONE — Waiting for next gesture
```

The **Recognition State** explains what EchoHands is currently doing with the user's gesture.

When the interface displays:

```text
NONE — Waiting for next gesture
```

the system is ready to observe and process the next valid gesture.

After EchoHands accepts a gesture, the recognition-control logic prevents the same held gesture from being added repeatedly. For example, if the user holds the sign `A` in front of the camera for several frames:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → A
Frame 4 → A
```

the system should not produce:

```text
AAAA
```

for a single intended gesture.

Instead, once the gesture is accepted, EchoHands waits for the appropriate transition before allowing another gesture to be added. This is why the recognition state is important during normal interaction with the application.

### Text Output

```text
Text: HELLO
```

The **Text** area displays the final characters and words accepted by EchoHands.

This output behaves differently from the live prediction area:

```text
Prediction → What the model is currently seeing

Text → What EchoHands has accepted as input
```

For example, the prediction may temporarily identify a sign while the user is moving between gestures, but the text changes only after EchoHands accepts a valid gesture through its recognition-control process.

As the user performs accepted signs one by one, the text is built gradually.

### Keyboard Controls

The controls displayed at the bottom of the interface allow the user to interact directly with the generated text.

| Key | Behaviour |
|---|---|
| `[SPACE]` | Adds a space to the current text |
| `[DOUBLE SPACE]` | Clears the complete text output |
| `[BACKSPACE]` | Removes the last accepted character |
| `[Q]` | Closes the EchoHands application |

For example, after recognizing:

```text
H E L L O
```

the user can press:

```text
[SPACE]
```

to continue with the next word.

If the user wants to remove the last accepted character:

```text
[BACKSPACE]
```

removes it from the text output.

A double press of:

```text
[SPACE]
```

clears the complete current text.

### Typical User Interaction Flow

A normal interaction with EchoHands follows this sequence:

```text
User shows hand to camera
        ↓
Hand is detected
        ↓
Interface updates the prediction and confidence
        ↓
Static pose or motion sequence is processed
        ↓
Recognition control decides whether to accept the gesture
        ↓
Accepted sign is added to the text output
        ↓
System waits for the next gesture
```

For dynamic gestures, the interaction additionally involves collecting multiple frames:

```text
User performs hand movement
        ↓
Sequence Frames increases
        ↓
Movement sequence is analyzed
        ↓
Dynamic gesture is predicted
        ↓
Recognition control accepts the gesture
        ↓
Character is added to Text
```

The interface therefore gives the user feedback throughout the interaction — from hand detection, to live prediction, to sequence processing, to final text generation.


# Project Structure

```text
EchoHands/
│
├── Assets/
│   ├── Project images
│   └── Detailed structural documentation
│
├── data/
│   └── Project datasets and processed data retained for development
│
├── models/
│   ├── Static recognition model artifacts
│   ├── Dynamic recognition model artifacts
│   ├── Label encoders
│   └── Related model metadata
│
├── sign description/
│   └── Supported sign reference images
│
├── src/
│   │
│   ├── app.py
│   │   Main real-time application entry point
│   │
│   ├── core/
│   │   Core real-time recognition components
│   │
│   ├── dataset/
│   │   Dataset collection, preparation and analysis modules
│   │
│   ├── training/
│   │   Training, evaluation and analysis modules
│   │
│   └── utils/
│       Utility components
│
├── tests/
│   └── Development and validation scripts
│
├── requirements.txt
│   Python dependencies
│
└── README.md
```

## Core Recognition Components

### `app.py`

`src/app.py` is the main entry point of EchoHands. It connects the major parts of the application, including camera input, hand detection, landmark processing, prediction, recognition control, sequence detection, text building, keyboard controls, and the real-time display.

The application is started with:

```bash
python -m src.app
```

### `camera.py`

Handles webcam access and frame retrieval for the real-time recognition loop.

### `hand_detector.py`

Uses MediaPipe to detect the user's hand and obtain hand landmark information from each webcam frame. It can also draw detected landmarks and hand connections on the displayed camera frame.

### `landmark_processor.py`

Processes raw hand landmark coordinates into a consistent representation that can be used by the recognition models.

### `predictor.py`

Handles static gesture prediction by loading the trained static recognition model and using processed hand features to predict the corresponding sign.

### `dynamic_predictor.py`

Handles dynamic gesture prediction from a sequence of hand information collected across multiple frames. This is used for motion-based signs such as **J** and **Z**.

### `recognition_controller.py`

Manages the real-time recognition state and helps coordinate how predictions are accepted.

### `sequence_detector.py`

Collects and manages sequences of hand information needed for dynamic gesture recognition.

### `word_builder.py`

Maintains the text currently constructed by EchoHands. It supports adding recognized characters and spaces, removing characters, clearing text, and resetting the text state.

---

## Models

The `models/` directory contains the trained model artifacts required by the Alpha application.

These include resources related to:

- Static sign recognition
- Dynamic sign recognition
- Label encoding
- Model class information
- Other model metadata

The required model files must remain in the locations expected by the source code.

Do not rename, move, or delete model artifacts unless the related source code is updated accordingly.

---

## Data, Dataset & Training Modules

The Alpha repository retains modules used during the development of the recognition system.

These include functionality related to:

- Data collection
- Duplicate detection
- Dataset quality checking
- Variation management
- Sequence generation
- Dataset preparation
- Feature generation
- Data augmentation
- Model training
- Model evaluation
- Validation and cross-validation
- Error analysis
- Recognition experiments

These modules were used to build, train, test, and improve EchoHands.

They are **not required for a normal user to run the real-time application**:

```bash
python -m src.app
```

They are retained in the Alpha build because the repository also serves as the technical development baseline for future improvement, retraining, debugging, and experimentation.

---

## Tests & Development Scripts

The `tests/` directory contains scripts used during development and validation.

These scripts were used to verify different parts of the recognition pipeline, including:

- Core recognition behavior
- Model behavior
- Dynamic recognition
- Sequence handling
- Feature handling
- Training-related experiments

They are not required for normal application usage.

Before major development changes, the source code can be checked using:

```bash
python -m compileall src tests
```

The stable application itself should then be tested using:

```bash
python -m src.app
```

# How EchoHands Works

The main recognition pipeline is:

```text
Camera Input
      ↓
Hand Detection
      ↓
Hand Landmark Extraction
      ↓
Landmark / Feature Processing
      ↓
Recognition Controller
      ↓
Static or Dynamic Recognition
      ↓
Gesture Acceptance
      ↓
WordBuilder
      ↓
Digital Text Output
```

The webcam continuously captures frames. MediaPipe detects the hand and extracts its landmarks. These landmarks are processed into the feature representation required by the recognition pipeline.

The system then decides whether the input should be handled through the static recognition pipeline or analyzed as part of a dynamic sequence.

For stable hand poses, the static model predicts the corresponding sign. For motion-based signs such as **J** and **Z**, the system analyzes movement across multiple frames.

Once a valid gesture is accepted, it is passed to the `WordBuilder` and added to the digital text output.

## Why Recognition Control Is Important

A webcam processes many frames every second. If a user holds the same gesture in front of the camera, the model may predict the same sign repeatedly:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → A
Frame 4 → A
```

Without recognition control, this could incorrectly produce:

```text
AAAA
```

even though the user intended to enter only one **A**.

EchoHands therefore uses gesture acceptance and recognition-state logic to prevent a held gesture from being repeatedly added to the text. The system waits for the appropriate transition or release behavior before accepting another gesture.

This is important because a model prediction alone should not automatically become a new character in the final text.

---

# Static Recognition

Static gestures are recognized from the hand configuration detected in a frame.

The general process is:

```text
Camera Frame
→ Hand Detection
→ Landmark Extraction
→ Feature Processing
→ Static Model
→ Predicted Sign
→ Recognition Acceptance
→ WordBuilder
```

The static recognition model produces a predicted sign and associated confidence. The application then applies recognition logic before allowing that sign to affect the final text output.

---

# Dynamic Recognition

Some gestures contain meaningful motion, so a single frame does not provide enough information.

EchoHands therefore uses a sequence-based workflow:

```text
Hand Movement
→ Landmark Frames
→ Sequence Collection
→ Movement / Sequence Detection
→ Dynamic Model
→ Dynamic Prediction
```

The current dynamic recognition system supports:

- **J**
- **Z**

The dynamic model analyzes information collected across multiple frames to determine which motion-based sign was performed.

---



---

# Alpha Build Purpose

The Alpha Build is the stable technical and development baseline of EchoHands.

Its purpose is to preserve the completed recognition pipeline in a working state while keeping the project suitable for future experimentation, retraining, debugging, and architectural development.

The Alpha Build includes both the working real-time application and the development-oriented components used to build and improve it.

It is therefore intended primarily as the project's technical foundation rather than the final public-facing product.

---

# Future Development

The Alpha Build is designed as a flexible recognition foundation rather than a final language-specific system.

## Expanding to Other Sign Languages

The current recognition pipeline can be adapted to other sign languages by collecting appropriate data, preparing language-specific datasets, and training suitable recognition models.

One important future direction is **Indian Sign Language (ISL)**.

The goal is not to assume that an ASL-trained model can directly recognize ISL. Instead, the existing EchoHands pipeline can serve as the technical foundation for building and training a separate recognition system using ISL-specific signs, datasets, labels, and models.

Conceptually:

```text
EchoHands Recognition Pipeline
            │
            ├── ASL Dataset + Models
            │        ↓
            │     ASL Recognition
            │
            └── ISL Dataset + Models
                     ↓
                  ISL Recognition
```

This makes EchoHands suitable for future expansion into a broader, modular sign-language recognition platform.

## Modular Mobile Application Direction

A future version can separate the recognition system into modular parts so that the mobile application does not need to contain the complete training or heavy model-development environment.

A possible structure is:

```text
Mobile Application
        │
        ├── Camera Input
        ├── User Interface
        └── Communication Layer
                │
                ▼
        Remote Recognition Service
                │
                ├── Hand / Feature Processing
                ├── Static Recognition
                ├── Dynamic Recognition
                └── Trained Models
                │
                ▼
           Prediction Result
                │
                ▼
          Mobile Interface
```

In this direction, the mobile device would mainly provide the camera and user interface, while the recognition models could run remotely.

## Cloud-Based Recognition Architecture

The longer-term idea is to support a cloud or remote architecture where model inference is performed on infrastructure controlled by the project.

Instead of requiring every mobile device to run the complete recognition stack locally:

```text
Mobile Phone
      ↓
Camera / Recognition Input
      ↓
Remote or Cloud Service
      ↓
EchoHands Models
      ↓
Prediction
      ↓
Result Returned to Phone
```

This could make the mobile application lighter and make model updates easier to manage centrally.

The exact architecture is a future development goal and would require further work on networking, latency, privacy, security, scalability, and deployment.

---

# Assets and Documentation

The `Assets/` directory contains project images and detailed structural documentation related to the EchoHands architecture and development.

The `sign description/` directory contains supported sign reference images.

---

# License and Access

This Alpha Build may be kept restricted to approved collaborators and development use.

If access to the repository is private or controlled, contributors should not redistribute the source code, datasets, or model artifacts without permission from the project owner.

---

# Author

**Shivanshu Khode**

**EchoHands — AI-powered real-time sign language recognition.**
