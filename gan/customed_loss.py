
import tensorflow as tf

def wassertein_d_loss(output,target):
	d_x = - output[0][0]
	d_g_z = output[0][0]
	return (d_x * target[0][0]) + (d_g_z * target[0][1])

def wassertein_g_loss(output,target):
	return - output[0][0]