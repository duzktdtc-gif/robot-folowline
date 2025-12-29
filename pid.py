
# -*- coding: utf-8 -*-
import time

class PIDController:
    def __init__(self, kp, ki, kd, windup=1000.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup = float(windup)
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, error, now=None):
        if now is None:
            now = time.time()
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = now - self.prev_time

        # P
        p = self.kp * error

        # I
        if dt > 0:a
            self.integral += error * dt
            # anti-windup
            if self.integral > self.windup:
                self.integral = self.windup
            elif self.integral < -self.windup:
                self.integral = -self.windup
        i = self.ki * self.integral

        # D
        if self.prev_error is None or dt == 0:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / dt

        self.prev_error = error
        self.prev_time = now
        return p + i + d
