import cv2
import numpy as np
import time


def get_frame(webcam):
    ret, frame = webcam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def combine(old_frame, new_frame):
    alpha = 0.3
    return cv2.addWeighted(new_frame, alpha, old_frame, 1-alpha, 0.0)


def mutate_video(video, out):
    all_video = video
    while True:
        for t in range(50):
            next_frame = get_frame(webcam)
            video[t] = combine(video[t], next_frame)
            cv2.imshow('frame', video[t])
            out.write(video[t])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return out


webcam = cv2.VideoCapture(0)
img = get_frame(webcam)
height, width = img.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width,height), isColor=False)


while True:
    next_frame = get_frame(webcam)
    cv2.imshow('frame', next_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video = np.stack([get_frame(webcam) for _ in range(50)])
out = mutate_video(video, out)


"""frame = (frame + next_frame)/2.0
# frame = next_frame


print(np.mean(frame), np.mean(next_frame))

# Our operations on the frame come here
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the resulting frame
cv2.imshow('frame', frame)
#print(gray)"""
