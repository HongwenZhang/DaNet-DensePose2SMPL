import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import trimesh

try:
    import pyrender
except:
    pass

try:
    from opendr.renderer import ColoredRenderer
    from opendr.lighting import LambertianPointLight, SphericalHarmonics
    from opendr.camera import ProjectPoints
except:
    pass

import neural_renderer as nr

from utils.densepose_methods import DensePoseMethods

from skimage.transform import resize

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy().copy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img


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
    def __init__(self, ratio=1, color=None):
        self.ratio = ratio
        self.color = color

    def render(self, image, cam, K, verts, face):
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


class IUV_Renderer(object):
    '''
    Renderer for generating IUV maps
    Ref: H. Zhang et al. Learning 3D Human Shape and Pose from Dense Body Parts
    '''
    def __init__(self, orig_size=224, out_size=56, focal_length=5000.):

        self.orig_size = orig_size
        self.out_size = out_size
        self.focal_length = focal_length

        K = np.array([[self.focal_length, 0., self.orig_size / 2.],
                      [0., self.focal_length, self.orig_size / 2.],
                      [0., 0., 1.]])

        R = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        t = np.array([0, 0, 5])

        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])

        self.coco_plus2coco = [14, 15, 16, 17, 18, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

        DP = DensePoseMethods()

        vert_mapping = DP.All_vertices.astype('int64') - 1
        self.vert_mapping = torch.from_numpy(vert_mapping)

        faces = DP.FacesDensePose
        faces = faces[None, :, :]
        self.faces = torch.from_numpy(faces.astype(np.int32))

        num_part = float(np.max(DP.FaceIndices))
        textures = np.array(
            [(DP.FaceIndices[i] / num_part, np.mean(DP.U_norm[v]), np.mean(DP.V_norm[v])) for i, v in
             enumerate(DP.FacesDensePose)])

        textures = textures[None, :, None, None, None, :]
        self.textures = torch.from_numpy(textures.astype(np.float32))

        self.renderer = nr.Renderer(camera_mode='projection', image_size=self.out_size, fill_back=False, anti_aliasing=False,
                                    dist_coeffs=torch.FloatTensor([[0.] * 5]), orig_size=self.orig_size)
        self.renderer.light_intensity_directional = 0.0
        self.renderer.light_intensity_ambient = 1.0

    def verts2uvimg(self, verts, cam):
        ''' render IUV images of given SMPL vertices and camera.

        Args:
            verts (tensor): [B, 6890, 3] SMPL vertices
            cam (tensor): [B, 3] camera (s, x, y)
        Return:
            iuv_image (tensor): [B, 3, self.out_size, self.out_size] IUV images
        '''
        batch_size = verts.size(0)

        K, R, t = self.camera_matrix(cam)

        # map verts to its DensePose version
        vertices = verts[:, self.vert_mapping, :]

        iuv_image = self.renderer(vertices, self.faces.to(verts.device).expand(batch_size, -1, -1),
                               self.textures.to(verts.device).expand(batch_size, -1, -1, -1, -1, -1).clone(),
                               K=K, R=R, t=t,
                               mode='rgb',
                               dist_coeffs=torch.FloatTensor([[0.] * 5]).to(verts.device))

        return iuv_image

    def camera_matrix(self, cam):
        '''
        Args:
            cam (tensor): [B, 3] camera (s, x, y)
        '''
        batch_size = cam.size(0)

        K = self.K.repeat(batch_size, 1, 1)
        R = self.R.repeat(batch_size, 1, 1)
        t = torch.stack([cam[:, 1], cam[:, 2], 2 * self.focal_length/(self.orig_size * cam[:, 0] + 1e-9)], dim=-1)
        t = t.unsqueeze(1)

        if cam.is_cuda:
            device_id = cam.get_device()
            K = K.cuda(device_id)
            R = R.cuda(device_id)
            t = t.cuda(device_id)

        return K, R, t
