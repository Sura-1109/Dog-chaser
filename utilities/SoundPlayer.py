import os
import time

#see if pygame works first, if not fall back and use winsound on windows instead
class SoundPlayer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mode = None
        try:
            import pygame
            pygame.mixer.init()
            self.pygame = pygame
            self.sounds = {
                "low": pygame.mixer.Sound(os.path.join(base_dir, "sounds", "low.wav")),
                "MiD": pygame.mixer.Sound(os.path.join(base_dir, "sounds", "mid.wav")),
                "HIGH": pygame.mixer.Sound(os.path.join(base_dir, "sounds", "high.wav"))                
            }
            self.mode = "pygame"
        except Exception as e:
            try:
                import winsound
                self.winsound = winsound
                self.mode = "winsound"
            except Exception:
                self.mode = "none"

    def play(self, level):
        if self.mode == "pygame":
            self.sounds[level].play()
        elif self.mode == "winsound":
            #winsound doesn't really play wav files asynchronously without a window; use a Beep as a fallback
            freq = 4000 if level == "low" else (10000 if level == "MiD" else 15000)
            dur = 600
            try:
                self.winsound.Beep(freq, dur)
            except RuntimeError:
                pass
        else:
            #if no sound backend is available
            pass