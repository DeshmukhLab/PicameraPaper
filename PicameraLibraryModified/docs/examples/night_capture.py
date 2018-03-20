from picamera import PiCamera
from time import sleep
from fractions import Fraction

# Set a framerate of 1/6fps, then set shutter
# speed to 6s and ISO to 800
camera = PiCamera(resolution=(1280, 720), framerate=Fraction(1, 6))
camera.shutter_speed = 6000000
camera.iso = 800
# Give the camera a good long time to set gains and
# measure AWB (you may wish to use fixed AWB instead)
sleep(30)
camera.exposure_mode = 'off'
# Finally, capture an image with a 6s exposure. Due
# to mode switching on the still port, this will take
# longer than 6 seconds
camera.capture('dark.jpg')
