import rpi_camera as rpi_cam
import digit_class_cnn as dcnn
import tensorflow as tf
##import matplotlib.pyplot as plt

img_width = 96
img_height = 96
img_size = img_width*img_height

crop_width = 56
crop_height = 56
crop_size = crop_width*crop_height

camera = rpi_cam.rpi_camera(img_width, img_height, 0)
cnn = dcnn.digit_classifier()

repeat = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

while repeat > 0:

    if repeat == 1:

        ##  1. Training
        ##  2. Restore data
        print('1. Training, 2. Restore weights')
        mode = input('Mode? (1/2) ')

        print('selected: %s'%(mode))

        if mode == '1':
            cnn.train(sess, 2000)
            save_path = cnn.save_weights(sess, './digit_class_cnn.wgt')
            print('model weights saved at %s'%(save_path))
         
        else:
            cnn.restore_weights(sess, './digit_class_cnn.wgt')

        response = input('Test? (y/n) ')

        if response == 'y':
            repeat = 2
        else:
            repeat = -1          

    if repeat > 1:

        camera.overlay_rect_mark(img_width, img_height, crop_width, crop_height)

        output = camera.capture(6)
##        camera.display_image_data(output, img_width, img_height)

        camera.remove_overlay()

        crop_out = camera.crop(output, img_width, img_height, crop_width, crop_height)
##        camera.display_image_data(crop_out, crop_width, crop_height)

        half_smpl_img = camera.half_sampling(crop_out, crop_width, crop_height)
##        camera.display_image_data(half_smpl_img, (crop_width>>1), (crop_height>>1))

        inv_out = camera.pixel_inversion(half_smpl_img)
##        camera.display_image_data(inv_out, (crop_width>>1), (crop_height>>1))

        norm_out = camera.normalize(inv_out, (crop_width>>1), (crop_height>>1))
##        camera.display_image_data(norm_out, (crop_width>>1), (crop_height>>1))

        y_data = camera.y_data_only(norm_out, (crop_width>>1), (crop_height>>1))

        uint8_out = camera.cast_uint8(y_data)
##        camera.display_image_data(uint8_out, (crop_width>>1), (crop_height>>1))
        
        prediction = cnn.run(uint8_out, sess)
        print(prediction)

        predict_num = sess.run(tf.argmax(prediction, 1))
        print("Number = %d"%(predict_num))

        uint8_out_boost = camera.boost_contrast(uint8_out, (crop_width>>1), (crop_height>>1))

##        uint8_out_RGB = camera.y_data_to_RGB(uint8_out, (crop_width>>1), (crop_height>>1))
##        uint8_out_boost_RGB = camera.y_data_to_RGB(uint8_out_boost, (crop_width>>1), (crop_height>>1))

##        plt.imshow(uint8_out_boost_RGB)
##        plt.show()

        response = input('Repeat? (y/n) ')

        if response == 'n':
            repeat = -1
        else:
            repeat = 2
