import cv2
import pickle
import numpy as np
import pyttsx3
import time

model = pickle.load(open("sign_model.pkl", "rb"))

engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Phrases mapping
phrases = {
    "travelling_A": "Where is the ticket counter?",
    "travelling_B": "At what time will the train or bus come?",
    "travelling_C": "Which platform or gate should I go to?",
    "travelling_D": "How much is the fare?",
    "new_place_A": "Can you guide me around?",
    "new_place_B": "Where is the hotel?",
    "new_place_C": "Is this place safe?",
    "new_place_D": "How far is the main market?",
    "food_place_A": "Can I have the menu please?",
    "food_place_B": "Is this seat available?",
    "food_place_C": "Please bring water.",
    "food_place_D": "How much is the bill?"
}

cap = cv2.VideoCapture(0)
prev_label = ""
last_speak_time = 0

print("🎥 Showing predictions... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (100, 100)).flatten().reshape(1, -1)

    pred = model.predict(img)[0]

    text = phrases.get(pred, "Unknown")
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Real-Time Translation", frame)

    # Speak only if label changes or 5 seconds passed
    if text != prev_label or time.time() - last_speak_time > 5:
        engine.say(text)
        engine.runAndWait()
        prev_label = text
        last_speak_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
