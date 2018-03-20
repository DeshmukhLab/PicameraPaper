<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.

Follow the steps in following order:

* **DevelopmentEnvironment**: instruction to install Python and OpenCV on Windows and Linux systems

* **TTLandZmodo**: instructions to set up TTL wiring for synchronizing the Picamera system with the neural recording system. Also contains information about setting up Zmodo multi display system. These instructions will work with minor modifications for any cheap security camera system capable of displaying videos from multiple cameras, which accepts composite video inputs from multiple cameras.

* **PicameraTrackingSystem**: information about the Picamera system, operating the system and changes required to be made to Picamera library for accurate timestamping.

* **PicameraLibraryModified**: modified Picamera (http://picamera.readthedocs.io/en/release-1.13/) Python library code which allows accurate timestamping

* **PicameraVideoAcquistionScripts**: codes used in capturing video data, frame timestamps and Transistor-Transistor Logic (TTL) pulse timestamps with difference in logging TTL timestamp.
    * StartAcquisition_gpio.py: uses Rpi.GPIO (https://pypi.python.org/pypi/RPi.GPIO) and Picamera python library to log timestamp for each transition.
    * killrecording.py: kills Picamera video acqusition script (StartAcquisition_gpio.py).
    
* **LinuxStitchingCodes**: codes used in calculating the camera intrinsic parameter and position tracking from multiple cameras
    * getBlendedImage: generates single blended stitched image from a list of input images.
    * getStitchingParams: calculates the registration data for each camera.
    * getTransformedImages: generates the transformed images from each camera using the previously calculated registration data.
    * stitchedVideoTracking: calculates position in each video camera frame from given set of video inputs
    * stitching&RegistrationData: calculates the registration data and saves single blended image from multiple input images.
    
(Compilation and running instruction are present within each sub folder)
