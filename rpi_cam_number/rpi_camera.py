from picamera import PiCamera
from time import sleep
import numpy as np


##camera = PiCamera()
##img_width = 64
##img_height = 64
##img_size = img_width*img_height
##
##camera.rotation = 180
##camera.resolution = (img_width, img_height)
##camera.framerate = 15
##camera.color_effects = (128, 128)
##
##camera.start_preview()
##sleep(5)
##output = np.zeros(img_size+(img_size>>1), dtype=np.uint8)
##camera.capture(output, format='yuv')
##camera.stop_preview()

##for line in range(0, img_height):
##    for pixel in range(0, img_width):
##
##        print("%d "%(output[line*img_width + pixel]), end='')
##
##    print("")
##
##print("\n")
##
##for line in range(0, (img_height>>1)):
##    for pixel in range(0, (img_width>>1)):
##
##        print("%d "%(output[img_size + line*(img_width>>1) + pixel]), end='')
##
##    print("")
##
##print("\n")
##
##for line in range(0, (img_height>>1)):
##    for pixel in range(0, (img_width>>1)):
##
##        print("%d "%(output[img_size + (img_size>>2) + line*(img_width>>1) + pixel]), end='')
##
##    print("")
##
##print("\n")

class rpi_camera:

    def __init__(self, img_width, img_height, img_rotation):
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = img_width*img_height
        self.output = np.zeros(self.img_size+(self.img_size>>1), dtype=np.uint8)

        self.camera = PiCamera()
        self.camera.rotation = img_rotation
        self.camera.resolution = (img_width, img_height)
        self.camera.framerate = 15

    def capture(self, standby_sec):
        self.camera.start_preview()
        sleep(standby_sec)
        self.camera.capture(self.output, format='yuv')
        self.camera.stop_preview()

        return self.output


    def display_image_data(self, img_buf, img_width, img_height, show_uv = False):

        print(type(img_width))
        print(type(img_height))

        print(img_width)
        print(img_height)

        ##  Y(Luminance), 420
        for line in range(0, img_height):
            for pixel in range(0, img_width):

                print("%3d "%(img_buf[line*img_width + pixel]), end='')

            print("")

        print("\n")

        if(show_uv == True):
            ##  U(Chrominance, blue), 420
            for line in range(0, (img_height>>1)):
                for pixel in range(0, (img_width>>1)):

                    print("%3d "%(img_buf[img_size + line*(img_width>>1) + pixel]), end='')

                print("")

            print("\n")

            ##  V(Chrominance, red), 420
            for line in range(0, (img_height>>1)):
                for pixel in range(0, (img_width>>1)):

                    print("%3d "%(img_buf[img_size + (img_size>>2) + line*(img_width>>1) + pixel]), end='')

                print("")

            print("\n")

    def crop(self, img_buf, img_width, img_height, crop_width, crop_height):

        if(crop_width <= img_width and crop_height <= img_height):
            v_margin = (img_height - crop_height)>>1
            h_margin = (img_width - crop_width)>>1
            crop_size = crop_width*crop_height
            img_size = img_width*img_height
            self.cropout = np.zeros(crop_size+(crop_size>>1), dtype=np.uint8)
            
            for line in range(0, crop_height):
                for pixel in range(0, crop_width):
                    self.cropout[line*crop_width + pixel] = img_buf[(line+v_margin)*img_width + (pixel + h_margin)]

            for line in range(0, (crop_height>>1)):
                for pixel in range(0, (crop_width>>1)):
                    self.cropout[crop_size + line*(crop_width>>1) + pixel] = img_buf[img_size + (line+(v_margin>>1))*(img_width>>1) + (pixel + (h_margin>>1))]

                    self.cropout[crop_size + (crop_size>>2) + line*(crop_width>>1) + pixel] = img_buf[img_size + (img_size>>2) + (line+(v_margin>>1))*(img_width>>1) + (pixel + (h_margin>>1))]

            return self.cropout

        else:
            return -1


    def min_filtering_2x2(self, img_buf, img_width, img_height, v_pos, h_pos):

        pixel_tl = img_buf[v_pos*img_width + h_pos]
        pixel_tr = img_buf[v_pos*img_width + (h_pos + 1)]
        pixel_bl = img_buf[(v_pos + 1)*img_width + h_pos]
        pixel_br = img_buf[(v_pos + 1)*img_width + (h_pos + 1)]

        candidate = [pixel_tl, pixel_tr, pixel_bl, pixel_br]

        return min(candidate)

    def half_sampling(self, img_buf, img_width, img_height):

        img_size = img_width*img_height
        
        smpl_width = img_width>>1
        smpl_height = img_height>>1
        smpl_size = smpl_width*smpl_height
        self.smpl_out = np.zeros(smpl_size + (smpl_size>>1), dtype=np.uint8)

        for line in range(0, smpl_height):
            for pixel in range(0, smpl_width):

                img_v_pos = line<<1
                img_h_pos = pixel<<1

                sample_pixel = self.min_filtering_2x2(img_buf, img_width, img_height, img_v_pos, img_h_pos)
                
##                self.smpl_out[line*smpl_width + pixel] = img_buf[(line<<1)*img_width + (pixel<<1)]
                self.smpl_out[line*smpl_width + pixel] = sample_pixel

        for line in range(0, (smpl_height>>1)):
            for pixel in range(0, (smpl_width>>1)):
                
                self.smpl_out[smpl_size + line*(smpl_width>>1) + pixel] = img_buf[img_size + (line<<1)*(img_width>>1) + (pixel<<1)]

                self.smpl_out[smpl_size + (smpl_size>>2) + line*(smpl_width>>1) + pixel] = img_buf[img_size + (img_size>>2) + (line<<1)*(img_width>>1) + (pixel<<1)]

        return self.smpl_out

    def pixel_inversion(self, img_buf):

        inv_out = ~img_buf

        return inv_out

    def normalize(self, img_buf, img_width, img_height, uv_data = False):

        norm_width = img_width
        norm_height = img_height
        norm_size = img_width*img_height

        norm_out = np.zeros(norm_size + (norm_size>>1), dtype=np.float32)

        avg = np.average(img_buf)
##        stdev = np.std(img_buf)

        for line in range(0, norm_height):
            for pixel in range(0, norm_width):
##                norm_out[line*norm_width + pixel] = (img_buf[line*norm_width + pixel] - avg)/stdev
                norm_out[line*norm_width + pixel] = img_buf[line*norm_width + pixel] - avg
        
        for line in range(0, (norm_height>>1)):
            for pixel in range(0, (norm_width>>1)):
##                norm_out[norm_size + line*(norm_width>>1) + pixel] = (img_buf[norm_size + line*(norm_width>>1) + pixel] - avg)/stdev
##                norm_out[norm_size + (norm_size>>2) + line*(norm_width>>1) + pixel] = (img_buf[norm_size + (norm_size>>2) + line*(norm_width>>1) + pixel] - avg)/stdev
                norm_out[norm_size + line*(norm_width>>1) + pixel] = img_buf[norm_size + line*(norm_width>>1) + pixel] - avg
                norm_out[norm_size + (norm_size>>2) + line*(norm_width>>1) + pixel] = img_buf[norm_size + (norm_size>>2) + line*(norm_width>>1) + pixel] - avg

        return norm_out

    def y_data_only(self, yuv_buf, img_width, img_height):

        img_size = img_width*img_height

        y_buf = np.zeros(img_size, dtype=np.float32)

        for line in range(0, img_height):
            for pixel in range(0, img_width):
                y_buf[line*img_height + pixel] = yuv_buf[line*img_height + pixel]

        return y_buf

    def cast_uint8(self, img_buf):

        return img_buf.astype(np.uint8)
    
