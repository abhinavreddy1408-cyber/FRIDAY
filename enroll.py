"""
F.R.I.D.A.Y. Voice Enrollment Wizard
====================================
Guides the user through capturing 5 distinct phrases.
Extracts speaker embeddings using SpeechBrain and creates an average fingerprint
that serves as the 'Golden Signature' for Voice Identity validation.
"""

import os
import sys
import time

# Ensure the root folder is added so we can import 'friday'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import speech_recognition as sr

from friday.config import config
from friday.security import init_security, save_voice_profile

PHRASES = [
    "I am Iron Man.",
    "Friday, run a full system diagnostic.",
    "Sometimes you gotta run before you can walk.",
    "Friday, open Google Chrome and search.",
    "Initialising override protocol Alpha."
]


def main():
    print("=" * 60)
    print("      F.R.I.D.A.Y. VOICE ENROLLMENT WIZARD")
    print("=" * 60)

    init_security()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n[ERROR] PyTorch/SpeechBrain not installed.")
        print("Please ensure your dependencies finished installing and try again.")
        return

    print("\nI need to learn your unique voice signature.")
    print("I will give you 5 phrases to read. Wait for the '[*] GO:' prompt,")
    print("then speak the phrase clearly into your microphone.\n")

    input("Press Enter when you are in a quiet room and ready to begin...")
    print("\nCalibrating microphone...")

    recogniser = sr.Recognizer()
    recogniser.energy_threshold = config.audio.mic_energy_threshold
    recogniser.pause_threshold = 0.8
    recogniser.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        recogniser.adjust_for_ambient_noise(source, duration=1.0)

    print("Microphone ready.\n")

    captured_audios = []

    for i, phrase in enumerate(PHRASES, 1):
        print("-" * 60)
        print(f"Phrase {i}/5:")
        print(f'  "{phrase}"\n')

        # Give a second to read it
        time.sleep(1.5)
        print("  [*] GO (Listening)...")

        with sr.Microphone() as source:
            try:
                audio = recogniser.listen(source, timeout=10, phrase_time_limit=10)
                captured_audios.append(audio)
                print("  [âœ“] Captured successfully.")
            except sr.WaitTimeoutError:
                print("  [X] Timed out. We'll proceed with fewer samples.")

        time.sleep(1)

    print("\n" + "=" * 60)
    print("Processing voice embeddings... this may take a moment.")

    success = save_voice_profile(captured_audios)

    if success:
        print("\n[SUCCESS] Voice Identity signature created!")
        print("F.R.I.D.A.Y. will now securely isolate commands to your voice.")
    else:
        print("\n[FAILED] Could not generate fingerprint.")


if __name__ == "__main__":
    main()
