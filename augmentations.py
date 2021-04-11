import numpy as np
from math import floor, ceil
import cv2
from skimage import filters


def reflect_x_image(image): # okay for 0, 8 (maybe 1)
    N = len(image)
    ref_mat = []
    for i in range(N):
        ref_mat.append([[-1,0],[0,1]])
    new_img = np.zeros((N,28,28))
    for i in range(28):
        for j in range(28):
            data = image[:,i,j]
            new_idx = np.array([i,j])@ref_mat
            if np.all((new_idx[:,0]< 28) & (new_idx[:,1]<28)):
                new_img[new_idx[:,0],new_idx[:,1]] = data
    result = new_img
    return result

def reflect_y_image(image, axis): # okay for 0, 1, 8
    N = len(image)
    ref_mat = []
    for i in range(N):
        ref_mat.append([[1,0],[0,-1]])
    new_img = np.zeros((N,28,28))
    for i in range(28):
        for j in range(28):
            data = image[:,i,j]
            new_idx = np.array([i,j])@ref_mat
            if np.all((new_idx[:,0]< 28) & (new_idx[:,1]<28)):
                new_img[new_idx[:,0],new_idx[:,1]] = data
    result = new_img
    return result

def rotate_cv_image(image):
    angle = np.random.randint(-20,20,1)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotate_image(image):
    N = len(image)
    angle = np.random.randint(-20,20,N)
    theta = angle/180*np.pi
    rot = []
    for i in range(N):
        rot.append(np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]))
    new_img = np.zeros((N,28,28))
    for i in range(28):
        for j in range(28):
            data = image[:,i,j]
            new_idx = np.round(np.array([i,j])@rot).astype(int)
            if np.all((new_idx[:,0]< 28) & (new_idx[:,1]<28)):
                new_img[:,new_idx[:,0],new_idx[:,1]] = data
    result = new_img
    return result



def translate_image(image):
    N = len(image)
    translation = [] 
    for i in range(N):
        translation.append(np.random.randint(1,7,(N,2)))
    new_img = np.zeros((N,28,28))
    for i in range(28):
        for j in range(28):
            data = image[:,i,j]
            new_idx = np.array([N,i,j]) + translation
            if np.all((new_idx[:,0]< 28) & (new_idx[:,1]<28)):
                new_img[:,new_idx[:,0],new_idx[:,1]] = data
    result = new_img
    return result



def crop_image(image):  # How do we resize after crop??
    pixels_crop = np.random.randint(1,4)
    result = image[pixels_crop:-pixels_crop,pixels_crop:-pixels_crop]
    return result


def sqeeze_image(image):
    N = len(image)
    frac = np.random.randint(70,100,N)/100
    mat = []
    for i in range(N):
        mat.append(frac[i]*np.array([[1,0],[0,1]]))
    new_imgs = np.zeros((N,28,28))
    for i in range(28):
        for j in range(28):
            data = image[:,i,j]
            new_idx = np.round(np.array([i,j])@mat).astype(int)
            if np.all((new_idx[:,0]< 28) & (new_idx[:,1]<28)):
                new_imgs[:,new_idx[:,0],new_idx[:,1]] = data
    result = new_imgs
    return result


def noise_image(image):
    N = len(image)
    scale = np.random.randint(1,10,N)
    noise = []
    for s in scale:
        noise.append(np.random.normal(0,s,(28,28)))
    result = image + noise
    return result

def blur_image(image):
    t = np.random.randint(3,11,1)/10
    def gauss2D(n_x,n_y,t):
        const = 1/(2*t*np.pi)
        ran_x = range(-floor(n_x/2),ceil(n_x/2))
        ran_y = range(-floor(n_y/2),ceil(n_y/2))
        return [[const*np.exp(-(x**2+y**2)/(2*t)) for x in ran_x] for y in ran_y]
    filter_gauss2 = gauss2D(image.shape[0],image.shape[1],t)
    result = filters.edges.convolve(image, filter_gauss2)
    return result


def non_linear_image(image):
    N = len(image)
    non_lin_funcs = {}
    non_lin_funcs[0] = lambda img, x: np.sqrt(img)*x
    non_lin_funcs[1] = lambda img, x: np.sqrt(x)*img
    
    func = non_lin_funcs[np.random.randint(0,len(non_lin_funcs))]
    new_img = np.zeros((N,28,28))
    for i in range(28):
        new_img[:,i] = func(image[:,i],i)
    for j in range(28):
        new_img[:,:,j] = func(image[:,:,j],j)
    result = new_img
    return result