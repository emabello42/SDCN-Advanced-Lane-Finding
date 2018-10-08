from .camera import Camera
from .lines_detector import LinesDetector
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def test_calibration(src_img):
    cam = Camera()
    cam.calibrate()
    cam.save_calibration()
    cam2 = Camera()
    cam2.load()
    dst_img = cam2.undistort(src_img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(src_img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dst_img)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


if __name__ == '__main__':
    camera = Camera()
    print("Calibrating camera...")
    #camera.calibrate(nx=9,ny=6, path_pattern="../camera_cal/*.jpg")
    camera.load_calibration("../camera_cal/calibration_info.p")
    print("     DONE!")
    print("Generating perspective transform...")
    camera.generate_perspective_transform()
    print("     DONE!")
    print("Detecting lane lines...")
    detector = LinesDetector(camera)
    output = '../output_harder.mp4'
    #clip1 = VideoFileClip("../challenge_video.mp4")#.subclip(2,9)
    #clip1 = VideoFileClip("../project_video.mp4")#.subclip(15,20)
    clip1 = VideoFileClip("../harder_challenge_video.mp4")#.subclip(0,8)
    white_clip = clip1.fl_image(detector.process_frame) #NOTE: this function expects color images!!
    white_clip.write_videofile(output, audio=False)
    print("     DONE!")
