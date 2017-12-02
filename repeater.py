import cv2
import numpy as np
import time
import pygame
from pygame.locals import *

SAVE = None
THRESHOLD_VAL = 170
N_SECTIONS = 3
FRAMES = 50
ACTIVE = [True, True, True]

def get_frame(webcam):
    ret, frame = webcam.read()
    """_, frame = cv2.threshold(frame,
                             THRESHOLD_VAL, 255,
                             cv2.THRESH_BINARY)"""

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def combine(old_frame, new_frame):
    alpha = 0.2
    #new_frame = noisy("s&p", new_frame).astype(np.uint8)
    return cv2.addWeighted(new_frame, alpha, old_frame, 1-alpha, 0.0)


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.1
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def mutate_video(video, width, height, out=None):
    all_video = video
    cutoffs = [int(np.floor(i / N_SECTIONS * width)) for i in range(N_SECTIONS + 1)]
    while True:
        for t in range(FRAMES):
            next_frame = get_frame(webcam)
            for i in range(N_SECTIONS):
                if ACTIVE[i]:
                    part_old_frame = video[t, :, cutoffs[i]: cutoffs[i+1]]
                    part_new_frame = next_frame[:, cutoffs[i]: cutoffs[i+1]]

                    video[t, :, cutoffs[i]: cutoffs[i+1]] = combine(part_old_frame, part_new_frame)

            # video[t] = combine(video[t], next_frame)
            display_frame(video[t])

            if out:
                out.write(video[t])

            key = cv2.waitKey(1) & 0xFF

            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == pygame.K_q:
                    return 0
                for i, key in enumerate([pygame.K_1, pygame.K_2, pygame.K_3]):
                    if event.key == key:
                        print("CHANGING")
                        ACTIVE[i] = not ACTIVE[i]
                        print(ACTIVE)


def display_frame(frame):
    screen.fill([0,0,0])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))
    pygame.display.update()

webcam = cv2.VideoCapture(0)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([1280,720], pygame.FULLSCREEN)

img = get_frame(webcam)
height, width, depth = img.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')

if SAVE:
    out_obj = cv2.VideoWriter(SAVE, fourcc, 20.0, (width, height),
                              isColor=False)
else:
    out_obj = None

def initial_run():
    while True:
        frame = get_frame(webcam)

        display_frame(frame)

        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == pygame.K_q:
                return 0

initial_run()
print("run ok")

    #cv2.imshow("test", next_frame)
    #frame = pygame.surfarray.make_surface(frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

video = np.stack([get_frame(webcam) for _ in range(50)])
out_obj = mutate_video(video, width, height, out_obj)


"""frame = (frame + next_frame)/2.0
# frame = next_frame


print(np.mean(frame), np.mean(next_frame))

# Our operations on the frame come here
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the resulting frame
cv2.imshow('frame', frame)
#print(gray)"""
