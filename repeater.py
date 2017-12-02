import cv2
import numpy as np
import time
import pygame
from pygame.locals import *

SAVE = None
THRESHOLD_VAL = 170
N_SECTIONS = 3
MAX_N_FRAMES = 1000 # This should be longer than the loop you want, but too long may degrade performance.
ALPHA = 0.5
ACTIVE = [True, True, True]


def get_frame(webcam):
    ret, frame = webcam.read()
    """_, frame = cv2.threshold(frame,
                             THRESHOLD_VAL, 255,
                             cv2.THRESH_BINARY)"""

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def combine(old_frame, new_frame):

    #new_frame = noisy("s&p", new_frame).astype(np.uint8)
    #return cv2.addWeighted(new_frame, alpha, old_frame, 1-alpha, 0.0)
    return cv2.addWeighted(old_frame, 1-ALPHA, new_frame, ALPHA, 0.0)

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


def mutate_video(video, original_video, out=None):
    cutoffs = [int(np.floor(i / N_SECTIONS * width))
               for i in range(N_SECTIONS + 1)]

    # Capture original video
    #original_video = np.expand_dims(get_frame(webcam), 0)

    #video_list = []
    nframes = 0
    while True:
        #np.stack([get_frame(webcam) for _ in range(FRAMES)])
        frame = get_frame(webcam)
        display_frame(frame)
        original_video[nframes] = frame
        video[nframes] = frame
        #original_video = np.append(original_video, np.expand_dims(frame, 0), axis=0)
        #video_list.append(frame)
        nframes += 1

        def wait_for_space():
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                    return True
            return False
        if wait_for_space():
            break

    # nframes = original_video.shape[0]
    #nframes = len(video_list)
    #original_video = np.stack(video_list, 0)
    # original_video = np.stack([get_frame(webcam) for _ in range(FRAMES)])


    while True:
        for t in range(nframes):
            next_frame = get_frame(webcam)
            frame_to_display = np.zeros(shape=next_frame.shape)

            for i in range(N_SECTIONS):
                if ACTIVE[i]:
                    part_old_frame = video[t, :, cutoffs[i]: cutoffs[i+1]]
                    part_new_frame = next_frame[:, cutoffs[i]: cutoffs[i+1]]

                    this_bit = combine(part_old_frame, part_new_frame)
                    frame_to_display[:, cutoffs[i]: cutoffs[i+1]] = video[t, :, cutoffs[i]: cutoffs[i+1]]
                    video[t, :, cutoffs[i]: cutoffs[i+1]] = this_bit

                else:
                    frame_to_display[:, cutoffs[i]: cutoffs[i+1]] = original_video[t, :, cutoffs[i]: cutoffs[i+1]]
            # video[t] = combine(video[t], next_frame)
            display_frame(frame_to_display)

            if out:
                out.write(frame_to_display)

            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                    print("run ended!")
                    return 0
                for i, key in enumerate([pygame.K_3, pygame.K_2, pygame.K_1]):
                    if event.type == KEYDOWN and event.key == key:
                        print("CHANGING")
                        ACTIVE[i] = not ACTIVE[i]
                        print(ACTIVE)

    return 1 # Execution shouldn't get to here


def display_frame(frame):
    screen.fill([0, 0, 0])
    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
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
            if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                return 0


# Do expensive setup (creating big arrays) FIRST
first_frame = get_frame(webcam)
original_video_zeros = np.zeros([MAX_N_FRAMES] + list(first_frame.shape), dtype=np.uint8)
video_zeros = np.zeros([MAX_N_FRAMES] + list(first_frame.shape), dtype=np.uint8)

initial_run()
print("Initial run ok.")

out_obj = mutate_video(video_zeros, original_video_zeros, out_obj)
print("Exiting...")
