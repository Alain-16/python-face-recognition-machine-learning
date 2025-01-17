import cv2

def count_frame(video_path):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: couldn't open the video")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()

    return total_frames


video_path='/home/alain/Documents/images/trialVD.mp4'
frame_count=count_frame(video_path)

if frame_count is not None:
    print(f"Total number of frames in the video:{frame_count}")