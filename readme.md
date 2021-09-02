Face Detection and Hand Gesture Detection and Recognition
 
install dependencies with
pip install -r requirements.txt

Dlib installation
conda install -c conda-forge dlib
(for conda based virtual environments)

Structure of the current directory (only important ones are included)
+----gestureandface //master git directory
	+-----facial_images (DIR)
	+-----model (DIR)
	+-----main.py
	+-----config.py
	+-----requirements.txt
	+-----readme.md
	
facial_images (DIR)
consists images with unique identifier and/or name of the known person. this identifier is returned by the program when the face in a live video stream matches. program at its initial run creates facial encodings of each person in the directory, which it continuously matches with the encodings of the face in the video stream.

model (DIR)

main.py
is the main codebase/program of this whole system which continously reads on the video stream to perform recognition of face and hand gestures and calls a function with the parameters identifier, gesture recognized and time stamp when a gesture is seen

config.py
consists of various parameters that helps define and/or customize the program. these parameters can be linked up with GUI for further enhancements. the parameters are:
	
	=> face_match_threshold is the percentage threshold for face match accuracy
	=> display_window states if to display image or not, will only display video window when true
	=> display_bounding_box if to display bound box or not, true is only valid when display_window is true
	=> display_hand_landmarks if to display hand landmarks or not, true is only valid when display_window is true
	=> threshold defines pixel threshold between hand and face from 150 (close) to 300 (far)

requirements.txt
cosists of dependencies needed to run the program

readme.md
the file for documentation