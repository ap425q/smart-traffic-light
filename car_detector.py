#!/usr/bin/env python3
import jetson.inference
import jetson.utils
import RPi.GPIO as GPIO
import time

# set up the GPIO pin for the LED
led_pin = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(led_pin, GPIO.OUT)


display = jetson.utils.videoOutput("display://0") 
camera = jetson.utils.videoSource("csi://0")
#camera = jetson.utils.videoSource("cars1.mp4")
net = jetson.inference.detectNet("trafficcamnet", threshold=0.5)

info = jetson.utils.cudaFont()

human = ['person','car','bicycle']

# timestamp for last detection
last_detection_time = time.time()

while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(img)
    current_time = time.time()
    if current_time - last_detection_time >= 2:  # check if 2 seconds have passed since last
        car_detected = False
        for detection in detections:
            class_desc = net.GetClassDesc(detection.ClassID)
            if class_desc in human:
                print("[+]" ,class_desc ," detected")
                car_detected = True
                last_detection_time = current_time  # update timestamp for last detection
                break
        if car_detected:
            print("[*] Turing led on")
            GPIO.output(led_pin, GPIO.HIGH) # turn on the LED
        else:
            print("[-] Turing Led off")
            GPIO.output(led_pin, GPIO.LOW) # turn off the LED
    display.Render(img)	
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
