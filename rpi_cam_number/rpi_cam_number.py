import rpi_camera as rpi_cam

def display_image_data(img_buf, img_width, img_height):

    ##  Y(Luminance), 420
    for line in range(0, img_height):
        for pixel in range(0, img_width):

            print("%3d "%(img_buf[line*img_width + pixel]), end='')

        print("")

    print("\n")

##    ##  U(Chrominance, blue), 420
##    for line in range(0, (img_height>>1)):
##        for pixel in range(0, (img_width>>1)):
##
##            print("%3d "%(img_buf[img_size + line*(img_width>>1) + pixel]), end='')
##
##        print("")
##
##    print("\n")
##
##    ##  V(Chrominance, red), 420
##    for line in range(0, (img_height>>1)):
##        for pixel in range(0, (img_width>>1)):
##
##            print("%3d "%(img_buf[img_size + (img_size>>2) + line*(img_width>>1) + pixel]), end='')
##
##        print("")
##
##    print("\n")

img_width = 32
img_height = 32
img_size = img_width*img_height

crop_width = 28
crop_height = 28
crop_size = crop_width*crop_height

camera = rpi_cam.rpi_camera(img_width, img_height, 180)

repeat = 1

while repeat > 0:

    output = camera.capture()
    display_image_data(output, img_width, img_height)

    crop_out = camera.crop(output, img_width, img_height, crop_width, crop_height)
    display_image_data(crop_out, crop_width, crop_height)

    half_smpl_img = camera.half_sampling(crop_out, crop_width, crop_height)
    display_image_data(half_smpl_img, (crop_width>>1), (crop_height>>1))

    inv_out = camera.pixel_inversion(half_smpl_img)
    display_image_data(inv_out, (crop_width>>1), (crop_height>>1))

    norm_out = camera.normalize(inv_out, (crop_width>>1), (crop_height>>1))
    display_image_data(norm_out, (crop_width>>1), (crop_height>>1))

    uint8_out = camera.cast_uint8(norm_out)
    display_image_data(uint8_out, (crop_width>>1), (crop_height>>1))

    response = input('Repeat? (y/n) ')

    if(response == 'n'):
        repeat = -1
    else:
        repeat = 1

