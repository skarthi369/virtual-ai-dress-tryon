# AR Virtual Try-On Application

This application allows users to virtually try on different combinations of shirts and pants using AR technology. The app uses your computer's camera to track your body and overlay virtual clothing items in real-time.

## Features

- Real-time body tracking using MediaPipe
- Virtual shirt and pants overlay
- Multiple clothing options
- Recommended outfit combinations
- Interactive UI with CustomTkinter

## Requirements

- Python 3.8 or higher
- Webcam
- Required packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python virtual_tryon.py
```

2. Allow camera access when prompted
3. Use the dropdown menus to select different shirts and pants
4. Click on recommended combinations to try pre-selected outfits
5. Stand in front of the camera to see the virtual clothing overlay

## Controls

- Use the "Select Shirt" dropdown to choose different shirts
- Use the "Select Pants" dropdown to choose different pants
- Click on recommended combinations for quick outfit changes

## Note

The application uses basic color overlays for demonstration purposes. For production use, you may want to add actual clothing textures and more sophisticated AR rendering.
        

        python virtual_tryon.py