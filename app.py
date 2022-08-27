# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:19:44 2021
@author: Sonu
"""
import time
from playsound import playsound
import numpy as np
import cv2
from flask import Flask, render_template, Response, request
import mediapipe as mp
workout = "squats"
md_drawing=mp.solutions.drawing_utils
md_drawing_styles=mp.solutions.drawing_styles
md_pose = mp.solutions.pose

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    #if angle >180.0:
        #angle = 360-angle
        
    return angle 

def bp_coordinates(body_parts, idx):
    """
    Convenience method for getting (x, y) coordinates for
    a given body part id.
    returns tuple of (x, y) coordinates
    """

    return (body_parts[idx].x, body_parts[idx].y)

def gen(exercice):
    position = None
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()
    counter = 0
    stage = None
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        # print(result.pose_landmarks)
        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                try:
                    shoulder = [landmarks[md_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[md_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[md_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[md_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[md_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[md_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[md_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[md_pose.PoseLandmark.LEFT_HIP.value].y]
                    shoulderr = [landmarks[md_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[md_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbowr = [landmarks[md_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[md_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wristr = [landmarks[md_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[md_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angler = calculate_angle(shoulderr,elbowr,wristr)
 
                    knee = [landmarks[md_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[md_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[md_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[md_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    legs_angle = calculate_angle(hip,knee,ankle)

                    angle = calculate_angle(shoulder, elbow, wrist)
                
                     
                    lmList = []
                    for id,im in enumerate(result.pose_landmarks.landmark):
                        h,w,_= frame.shape
                        X = int(im.x*w)
                        Y = int(im.y*h)
                        print(len(lmList))
                        lmList.append([id,X,Y])
                        if len(lmList) !=0:
                            oh = (lmList[12][2])
                            print(oh)
                            #print(lmList[12][2])
                            '''
                            if ((lmList[12][2] - lmList[14][2])>=15 and (lmList[11][2] - lmList[13][2])>=15):
                                position = "down"
                            if ((lmList[12][2] - lmList[14][2])<=5 and (lmList[11][2] - lmList[13][2])<=5) and position == "down":
                                position = "up"
                                count +=1 
                                print(count)
                            '''
                except:
                    pass
                #print(str('GENOTIV')  + str(bp_coordinates(landmarks,md_pose.PoseLandmark.LEFT_WRIST.value)))
                #array_shoulder_l  = (bp_coordinates(landmarks,md_pose.PoseLandmark.LEFT_SHOULDER.value))
                #array_forearm_l   = (bp_coordinates(landmarks,md_pose.PoseLandmark.LEFT_ELBOW.value))
                #array_wrist_l  = (bp_coordinates(landmarks,md_pose.PoseLandmark.LEFT_WRIST.value))
                # Curl counter logic
                workout = exercice
                if workout == "squats": 
                    if (legs_angle > 160):
                        stage = "down"
                    if (legs_angle < 90 and stage =='down'):
                        stage="up"
                        counter +=1
                        playsound('sound.mp3',block=False)
                        print(counter)
                else:
                    if workout == "pushups":
                            if (angle > 160 and angler > 160):
                                stage = "down"
                            if (angle < 50 and stage =='down'and angler < 90):
                                stage="up"
                                counter +=1
                                playsound('sound.mp3',block=False)
                                print(counter)
                #print(str("ZERO :") +str(array_shoulder_l[0])+" "+str(array_shoulder_l[1]))
                #insh = (str(calculate_angle(array_shoulder_l,array_forearm_l,array_wrist_l)))
                #print(insh)
                cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)
                cv2.putText(img, str(legs_angle), 
                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 5.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                cv2.putText(img, (str(stage) + str(" ") + str(counter)+ str("")), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        # checking video frame rate
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Writing FrameRate on video
        cv2.putText(img, str(int(fps)), (280, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed_pushups',methods=['GET', 'POST'])
def video_feed():
    if request.method == 'POST':
        # Then get the data from the form
        tag = request.form['tag']
        print(tag)

        # Get the username/password associated with this tag
        #user, password = tag_lookup(tag)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen("pushups"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_squats',methods=['GET', 'POST'])
def squats():
    if request.method == 'POST':
        # Then get the data from the form
        tag = request.form['tag']
        print(tag)

        # Get the username/password associated with this tag
        #user, password = tag_lookup(tag)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen("squats"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')






if __name__=="__main__":
    app.run(debug=True)







