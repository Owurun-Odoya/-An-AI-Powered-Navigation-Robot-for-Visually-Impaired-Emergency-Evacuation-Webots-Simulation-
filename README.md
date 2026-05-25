# -An-AI-Powered-Navigation-Robot-for-Visually-Impaired-Emergency-Evacuation-Webots-Simulation-

This project implements an autonomous e‑puck robot controller in Webots. The robot navigates a maze using wall‑following, arrow detection, and EXIT sign recognition.
 Features

-Wall‑following using proximity sensors

-Arrow detection (turn left / right) using OpenCV

-EXIT detection using OCR (Tesseract)

-Voice feedback (“Turn left”, “Exit reached”)

-State‑machine navigation


**How it Works**

-Robot follows walls to explore

-Detects red arrow signs → turns accordingly

-Detects green EXIT sign → moves toward it

-Stops and announces when EXIT is reached


**Tools Used**

-Python

-Webots

-OpenCV

-NumPy

-pytesseract


**How to Run**

Install dependencies:

pip install numpy opencv-python pytesseract

Install Tesseract OCR and set path in the code

Open project in Webots and run the simulation

**Outcome**

The robot successfully navigates the maze by:

-following walls

-obeying arrows

-reaching the EXIT autonomously
