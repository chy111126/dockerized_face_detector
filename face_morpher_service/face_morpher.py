import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import math
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from dfc_vae import *
from utils import *
#from vgg_loss import *

'''
Tensorlayer implementation of DFC-VAE
'''

flags = tf.app.flags
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 2, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent representation vector from. [2048]")
flags.DEFINE_integer("interpol_steps", 5, "Number of interpolation images to be created (included original two images) ")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

class FaceMorpher:
    def __init__(self):
        ##========================= DEFINE MODEL ===========================##
        # the input_imgs are input for both encoder and discriminator
        input_imgs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim],
                                    name='real_images')
        self.input_imgs = input_imgs

        # normal distribution for generator
        z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
        # normal distribution for reparameterization trick
        eps = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
        lr_vae = tf.placeholder(tf.float32, shape=[])

        # ---------------------- encoder : still used to find latent vector ----------------------
        net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False)
        self.z_mean = z_mean

        # ---------------------- decoder : original structure ----------------------
        # decode z 
        # z = z_mean + z_sigma * eps
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # using reparameterization tricks
        gen0, gen0_logits = generator(z, is_train=True, reuse=False) # reconstruction

        # ---------------------- decoder : sample interpolated result, while reusing original decoder ----------------------
        interpolated_z = tf.placeholder(tf.float32, [FLAGS.interpol_steps, FLAGS.z_dim], name='interpol_images')
        self.interpolated_z = interpolated_z

        gen5, gen5_logits = generator(interpolated_z, is_train=False, reuse=True)
        self.gen5 = gen5

        # Start TF session
        self.sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(self.sess)

        # prepare file under checkpoint_dir
        model_dir = "dfc-vae3"
        save_dir = os.path.join("checkpoint", model_dir) #'./checkpoint/vae_0808'
        # Restore model weights
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(save_dir, 'weights.ckpt'))

        return

    def inference_model(self, face1_data, face2_data):
        return_dict = {
            "interpol_steps": FLAGS.interpol_steps,
            "interpol_faces": []
        }

        # ------------------------- Inference ------------------------------
        test_from_img = get_image_from_base64(face1_data, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)
        test_to_img = get_image_from_base64(face2_data, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)

        #print("test_from_img", test_from_img.shape)
        #print("test_to_img", test_to_img.shape)

        to_transform_img_set = np.array([test_from_img, test_to_img]).astype(np.float32)

        # Extract two latent-space vectors for the two faces
        img1 = self.sess.run([self.z_mean], feed_dict={self.input_imgs: to_transform_img_set})
        img1 = img1[0]

        # interpolation step
        # Linear interpolation is used
        time_step = FLAGS.interpol_steps - 1
        diff_img = (img1[1] - img1[0]) / time_step
        new_img = []
        for x in range(time_step+1):
            new_img.append(img1[0] + diff_img * x)
        inter_z = np.array(new_img)

        # Get reconstructed image
        img1 = self.sess.run([self.gen5.outputs], feed_dict={self.interpolated_z: inter_z})

        faces = img1[0]
        for face_arr in faces:
            result_face = inverse_transform(face_arr) * 255 // 1
            #print(result_face)
            result_face = cv2.cvtColor(result_face, cv2.COLOR_RGB2BGR)
            new_face = from_image_to_base64(result_face)
            return_dict['interpol_faces'].append(new_face)

        return return_dict

    def inference_model_local_test(self):

        # ------------------------- Inference ------------------------------
        test_from_img = get_image(os.path.join("./data/celebA/1017.jpg"), FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)
        test_to_img = get_image(os.path.join("./data/celebA/1027.jpg"), FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)

        to_transform_img_set = np.array([test_from_img, test_to_img]).astype(np.float32)

        # Extract two latent-space vectors for the two faces
        img1 = self.sess.run([self.z_mean], feed_dict={self.input_imgs: to_transform_img_set})
        img1 = img1[0]
        print(img1.shape)

        # interpolation step
        time_step = FLAGS.interpol_steps - 1
        diff_img = (img1[1] - img1[0]) / time_step
        new_img = []
        for x in range(time_step+1):
            new_img.append(img1[0] + diff_img * x)
        inter_z = np.array(new_img)
        print(inter_z)

        # Get reconstructed image
        img1 = self.sess.run([self.gen5.outputs], feed_dict={self.interpolated_z: inter_z})
        print(img1[0].shape)
        save_images(img1[0], [8, 8], './inference.png')

        return

if __name__ == '__main__':
    # Unit test    
    # Accepts two 28*28 images input
    face1_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD4m/4Jgfsg+DvifokPxV+LWnM0NvdH7LDcgCORsZ8w7uCFzxnjPNfsT+xN8EPgr4NB1fwkvhObaf3iJdRSXA5O0nGTgdMcDOcAV8Ifs2fs4eI/HX7Pvhz4SHXjpsh0uLezQ5ijmDrMQVXa2CRtbDDIJBOCRXrn7OP7CPxV+HXxt0nU/HfxItbiBL2Fyul2iRs8PnKZlISONVVkLcfOdxGTnmvzqvVp4ytUqSqNNN2XQ/ZMLgqmEw1OkoLVe8+t/Q6X/g4c/ZsP7QnwC8O+IfCOv6HBqui6qkF28l9FCkdlL826YkjCo8ahc9PMbHXnz39jbwR8Cvh9+zX4U8E+NL0Weo6bpiQ3KWl0TG79XkBXIbc5Y5HXNa/7RP8AwTT+JXh/wf8AFvxdrfjnSfEUSie7t7aS2L3VxB58bSQcJuANuHQKGbIKgL0I+Yv2YPFvi6x+AvhvT/H2gi01G1s3t3jSJY1KRyvGhVEAVF2qMAAAY44xWlVxWAVNTvaW3qvxS2MsPBwx3O4PWNk+lk9F2T1vue2/DPxrrsemzaJpGvR2WoQsVZymQGGQeKveB7L9oK7N5B4P1nxZpviW8u9tzqNlfaax1GNX3IYop5xujwBhVCsMkZzmvN/ipfT+Atb0Lxr4dIjutQu4YL2JxmKZWwMleoYeoI9819G6b4C0P4iW2l6Prct1DHPGjF7C5MLqT/dYcivMpWjLbc9pTU36HAftO+MPjD8K9bfW/iHrfiA3mreF203+y9ceyL6vJFFN9od4kleNYhH0BO442kcjPzP4A+Oel+EfDMHhjXtCGnyWXEVukEaqI3AkBAT5cfORx6V6B/wWO1F/gt+0TYfD7wzaW17pOj/CWwtrK11u3W72/aru8E0hLjPmEInzdto9K+AvF/7SHjK9i0vUV8N+Hbd57CQypZaQIEJW6nQHahAztRRn2r2KGAnXhzrZu34X/U8jFZjSoPklvq16Xt+aP//Z"
    face2_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD4H/ZE/ZL8PeJddvX1u7jXUdClCT6S7YlDcqSytgheCM46+4r9f/2BNX+DPwwSDQ9K0/TtM1L7KN9ndwCOaRegKHoQD/DwR718n3Xwa+G/jGxmbw7E9hqMVtLJoutafcD7WHZADiU8ybwMEMcEnJ55pNd/Yk8TeGfEwi8D/tAanJqmoTRroYuGea5hy4UzyRXEYeOFN4Lv5zBgABtLAV+c4if9oVPaSk1bo76fd+qP1zB4b6jS9lCKa7qyb379fmfqR8Wte8L/ABR+FmseAfEWs2b22s6ZcWE9o1wpeaKWNo2UL16MR071+NVtpOseF7ePwx4jiMeoaaotL9GOSs0fyOP++lNfVX7PXwY/aO+Dvx58S2PxJ/aBgkXT5Li3luzo8bR7Ekk2K6vdM0ZkiEUgARwQ+dwBUj5P1rV/GWq+JNY1j4h2sVprd9rd7dapbQB1jjmkuJHdVVyWVcscKSSowM8VtgacqdWaunt38+6XQ7sLy3tGLS+Xl5v5HSWGq6p4Y8N2mlauht7yJJFZIWziRCQwUnGeQSD3Fbngy7+J+qeJTpvgPU7xb28th9l8R/2/Ol4ysFL2xRbaQBAR8qk7OCTzWf8AHTSbWfUbyEF48RRXMbxNtaORi4JH/fA/HOetdr/wT68A+EP2nhDoHxW0fzhZyO0F5p9zLa3ClW4+eNh/npiufHUXhsbUi9k/z1RhlOLWIwVOrK/vK79ev4m/8Qf2hP2hfhV4603xR4on1myMHh5tN1ePxXc212Ne8qQtBcRRoiGIHzXTLorbY1PzDBr5r1vxPqPiHWbvX9Xu/Pu765kuLqZzkvI7FmY/UkmvYP8AgpJJF4f+PMXw50O1S20nQdHhhsLdXdm5LEs7OxZ2OByT2+teAQ5kTcx5zXXg6acOddfyPfpQhSpc76n/2Q=="

    fm = FaceMorpher()
    return_dict = fm.inference_model(face1_data, face2_data)
    print(return_dict['interpol_steps'])
    for img in return_dict['interpol_faces']:
        print(img[:100])
