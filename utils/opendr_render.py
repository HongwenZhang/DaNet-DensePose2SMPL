## Could also use matplotlib to display
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from skimage.transform import resize


#  https://github.com/classner/up/blob/master/up_tools/camera.py
def rotateY(points, angle):
    """Rotate all points in a 2D array around the y axis."""
    ry = np.array([
        [np.cos(angle),     0.,     np.sin(angle)],
        [0.,                1.,     0.           ],
        [-np.sin(angle),    0.,     np.cos(angle)]
    ])
    return np.dot(points, ry)

def rotateX( points, angle ):
    """Rotate all points in a 2D array around the x axis."""
    rx = np.array([
        [1.,    0.,                 0.           ],
        [0.,    np.cos(angle),     -np.sin(angle)],
        [0.,    np.sin(angle),     np.cos(angle) ]
    ])
    return np.dot(points, rx)

def rotateZ( points, angle ):
    """Rotate all points in a 2D array around the z axis."""
    rz = np.array([
        [np.cos(angle),     -np.sin(angle),     0. ],
        [np.sin(angle),     np.cos(angle),      0. ],
        [0.,                0.,                 1. ]
    ])
    return np.dot(points, rz)

class opendr_render(object):
    def __init__(self, ratio=2, color=None):
        """
        Initialize the ratio.

        Args:
            self: (todo): write your description
            ratio: (todo): write your description
            color: (bool): write your description
        """
        self.ratio = ratio
        self.color = color

    def render(self, image, cam, K, verts, face):
        """
        Render the image as an rgb image.

        Args:
            self: (todo): write your description
            image: (array): write your description
            cam: (todo): write your description
            K: (todo): write your description
            verts: (str): write your description
            face: (todo): write your description
        """
        ## Create OpenDR renderer
        rn = ColoredRenderer()

        ## Assign attributes to renderer
        w, h = (224 * self.ratio, 224 * self.ratio)

        f = np.array([K[0, 0], K[1, 1]]) * float(self.ratio)
        c = np.array([K[0, 2], K[1, 2]]) * float(self.ratio)
        t = np.array([cam[1], cam[2], 2 * K[0, 0] / (224. * cam[0] + 1e-9)])
        rn.camera = ProjectPoints(v=verts, rt=np.zeros(3), t=t, f=f, c=c, k=np.zeros(5))

        rn.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}

        albedo = np.ones_like(verts)*.9

        if self.color is not None:
            color0 = self.color
            color1 = self.color
            color2 = self.color
        else:
            # white
            color0 = np.array([1, 1, 1])
            color1 = np.array([1, 1, 1])
            color2 = np.array([0.7, 0.7, 0.7])

        # rn.set(v=m*np.array([1,-1,1]), f=m.f, bgcolor=np.zeros(3))
        # rn.set(v=verts, f=face, vc=color, bgcolor=np.zeros(3))
        rn.set(v=verts, f=face, bgcolor=np.zeros(3))

        yrot = np.radians(120)

        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=color0)

        # Construct Left Light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=color1)

        # Construct Right Light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=color2)

        img_orig = np.transpose(image, (1, 2, 0))
        img_resized = resize(img_orig, (img_orig.shape[0] * self.ratio, img_orig.shape[1] * self.ratio), anti_aliasing=True)

        img_smpl = img_resized.copy()
        img_smpl[rn.visibility_image != 4294967295] = rn.r[rn.visibility_image != 4294967295]

        rn.set(v=rotateY(verts, np.radians(90)), f=face, bgcolor=np.zeros(3))
        render_smpl = rn.r

        render_smpl_rgba = np.zeros((render_smpl.shape[0], render_smpl.shape[1], 4))
        render_smpl_rgba[:, :, :3] = render_smpl
        render_smpl_rgba[:, :, 3][rn.visibility_image != 4294967295] = 255

        return img_orig, img_resized, img_smpl, render_smpl_rgba
