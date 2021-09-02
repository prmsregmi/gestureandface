#parameter definition
buffer = True #if buffer, only gives response after buffer time else continuous
buffer_duration = 20 #define in number of loops
unknown_buffer = 20
process_this_frame = True
face_match_threshold = 65 #percentage threshold for face match accuracy
display_window = True #display image or not, will only display result when false
display_bounding_box = True #display bound box or not, true is only valid when display_window is true
display_hand_landmarks = True #display hand landmarks or not, true is only valid when display_window is true
threshold = 200 #define pixel threshold between hand and face from 150 (close) to 300 (far)