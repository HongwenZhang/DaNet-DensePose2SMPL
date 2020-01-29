## Could also use matplotlib to display
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
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
    def __init__(self, model_path, scale_ratio=1):
        super(opendr_render, self).__init__()

        self.ratio = scale_ratio

        self.m = load_model(model_path)

    def render(self, image, K, verts):

        ## Create OpenDR renderer
        rn = ColoredRenderer()

        ## Assign attributes to renderer
        w, h = (224 * self.ratio, 224 * self.ratio)

        f = np.array([K[0, 0, 0], K[0, 1, 1]]) * float(self.ratio)
        c = np.array([K[0, 0, 2], K[0, 1, 2]]) * float(self.ratio)

        rn.camera = ProjectPoints(v=verts, rt=np.zeros(3), t=np.array([0, 0, 5.]), f=f, c=c, k=np.zeros(5))

        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}

        albedo = np.ones_like(self.m)*.9

        color1 = np.array([0.85490196, 0.96470588, 0.96470588])
        # light steel blue
        color2 = np.array([i / 255. for i in [181, 178, 146]])
        color3 = np.array([i / 255. for i in [190, 178, 167]])
        # beige
        # color4 = np.array([i / 255. for i in [245, 245, 220]])
        # wheat
        color4 = np.array([i / 255. for i in [245, 222, 179]])
        # thistle
        # color5 = np.array([i / 255. for i in [216, 191, 216]])
        color5 = np.array([i / 255. for i in [183, 166, 173]])

        # aqua marine
        color6 = np.array([i / 255. for i in [127, 255, 212]])
        # turquoise
        color7 = np.array([i / 255. for i in [64, 224, 208]])
        # medium turquoise
        color8 = np.array([i / 255. for i in [72, 209, 204]])
        # honeydew
        color9 = np.array([i / 255. for i in [240, 255, 240]])
        # burly wood
        color10 = np.array([i / 255. for i in [222, 184, 135]])
        # sandy brown
        color11 = np.array([i / 255. for i in [244, 164, 96]])
        # floral white
        color = np.array([i / 255. for i in [255, 250, 240]])

        rn.set(v=verts, f=self.m.f, vc=color, bgcolor=np.zeros(3))

        yrot = np.radians(120)
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Right Light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        img_orig = np.transpose(image, (1, 2, 0))
        img_resized = resize(img_orig, (img_orig.shape[0] * self.ratio, img_orig.shape[1] * self.ratio), anti_aliasing=True)

        img_smpl = img_resized.copy()
        img_smpl[rn.visibility_image != 4294967295] = rn.r[rn.visibility_image != 4294967295]

        rn.set(v=rotateY(verts, np.radians(90)), f=self.m.f, bgcolor=np.zeros(3))
        render_smpl = rn.r

        smpl_rgba = np.zeros((render_smpl.shape[0], render_smpl.shape[1], 4))
        smpl_rgba[:, :, :3] = render_smpl
        smpl_rgba[:, :, 3][rn.visibility_image != 4294967295] = 255

        return img_orig, img_resized, img_smpl, smpl_rgba
