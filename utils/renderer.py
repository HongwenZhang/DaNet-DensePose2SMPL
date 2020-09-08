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
    def __init__(self, ratio=2, color='white'):
        self.ratio = ratio
        self.color = color

        # self.m = load_model('./data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

        ## Assign random pose and shape parameters

        # m.pose[:] = np.random.rand(m.pose.size) * .2
        # m.betas[:] = np.random.rand(m.betas.size) * .03


    def render(self, image, cam, K, verts, face, draw_id=''):

        # roll_axis = torch.Tensor([1, 0, 0]).unsqueeze(0)  # .expand(1, -1)
        # alpha = torch.Tensor([np.pi] * 1).unsqueeze(1) * 0.5
        # pose[0, :3] = axis_angle_add(pose[0, :3].unsqueeze(0), roll_axis, alpha)
        # pose[:3] *= torch.Tensor([1, -1, -1])

        # self.m.betas[:] = shape.numpy()[0]
        # self.m.pose[:] = pose.numpy()[0]

        # m.betas[:] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # m.betas[:] = np.array([0.]*10)
        # m.pose[:] = np.array([0.]*72)
        # m.pose[0] = -np.pi
        # m.pose[2] = 0.5
        # m.pose[2] = np.pi
        # m.betas[0] = 0.5

        ## Create OpenDR renderer
        rn = ColoredRenderer()

        # print(rn.msaa)
        #
        # rn.msaa = True

        ## Assign attributes to renderer
        w, h = (224 * self.ratio, 224 * self.ratio)
        # w, h = (1000, 1000)

        f = np.array([K[0, 0], K[1, 1]]) * float(self.ratio)
        c = np.array([K[0, 2], K[1, 2]]) * float(self.ratio)
        t = np.array([cam[1], cam[2], 2 * K[0, 0] / (224. * cam[0] + 1e-9)])
        # t = np.array([0, 0, 5.])

        # c = np.array([K[0, 0, 2], 112 - K[0, 1, 1] * float(cam[0, 2])]) * float(self.ratio)

        # rn.camera = ProjectPoints(v=m*np.array([1,-1,-1]), rt=np.zeros(3), t=np.array([0, 0, 5.]), f=f, c=c, k=np.zeros(5))
        rn.camera = ProjectPoints(v=verts, rt=np.zeros(3), t=t, f=f, c=c, k=np.zeros(5))

        rn.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}

        # [:, [1, 0, 2]]

        albedo = np.ones_like(verts)*.9
        # albedo(6890, 3)(6890, 3)(13776, 3)

        color1 = np.array([0.85490196, 0.96470588, 0.96470588])
        # light steel blue
        # color1 = np.array([i / 255. for i in [176, 196, 222]])
        # color1 = np.array([i / 255. for i in [168, 173, 180]])
        # color2 = np.array([i / 255. for i in [255, 244, 229]])
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
        # floral white Ours
        color12 = np.array([i / 255. for i in [255, 250, 240]])
        # medium slate blue SPIN
        color13 = np.array([i / 255. for i in [72 * 2.5, 61 * 2.5, 255]])


        # color_list = [color1, color2, color3, color4, color5]
        color_list = [color6, color7, color8, color9, color10, color11, color12, color13]
        # color_list = color_list + [color13]

        # color = color_list[int(len(color_list) * float(np.random.rand(1)))]
        # color = color_list[-1]
        if self.color in ['white']:
            color = color12
            color0 = np.array([1, 1, 1])
            color1 = np.array([1, 1, 1])
            color2 = np.array([0.7, 0.7, 0.7])
        elif self.color in ['blue']:
            color = color13
            color0 = color
            color1 = color
            color2 = color

        # rn.set(v=m*np.array([1,-1,1]), f=m.f, bgcolor=np.zeros(3))
        rn.set(v=verts, f=face, vc=color, bgcolor=np.zeros(3))
        # rn.set(v=rotateY(verts, np.radians(90)), f=self.m.f, bgcolor=np.zeros(3))

        ## Construct point light source
        # rn.vc = LambertianPointLight(
        #     f=m.f,
        #     v=rn.v,
        #     num_verts=len(m),
        #     light_pos=np.array([-1000,-1000,-2000]),
        #     vc=np.ones_like(m)*.9,
        #     light_color=np.array([1., 1., 1.]))
        yrot = np.radians(120)
        '''
        rn.vc = LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([-200, -100, -100]),
            vc=albedo,
            light_color=color)

        # Construct Left Light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([500, 10, -200]),
            vc=albedo,
            light_color=color)

        # Construct Right Light
        rn.vc += LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=np.array([-300, 100, 600]),
            vc=albedo,
            light_color=color)
        '''
        # 1. 1. 0.7
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

        # render_smpl = rn.r

        ## Construct point light source
        # rn.vc += SphericalHarmonics(light_color=np.array([1., 1., 1.]))

        img_orig = np.transpose(image, (1, 2, 0))
        img_resized = resize(img_orig, (img_orig.shape[0] * self.ratio, img_orig.shape[1] * self.ratio), anti_aliasing=True)

        # ax_smpl = plt.subplot(2, 2, 2)
        # plt.imshow(rn.r)
        # plt.axis('off')

        # print(max(rn.r))
        # print(min(rn.r))
        # fig = plt.figure()

        img_smpl = img_resized.copy()
        img_smpl[rn.visibility_image != 4294967295] = rn.r[rn.visibility_image != 4294967295]

        '''
        ax_stack = plt.subplot(2, 2, 3)
        ax_stack.imshow(img_smpl)
        plt.axis('off')
        '''

        rn.set(v=rotateY(verts, np.radians(90)), f=face, bgcolor=np.zeros(3))
        render_smpl = rn.r

        # rn.set(v=rotateY(verts, np.radians(90)), f=self.m.f, bgcolor=np.zeros(3))

        render_smpl_rgba = np.zeros((render_smpl.shape[0], render_smpl.shape[1], 4))
        render_smpl_rgba[:, :, :3] = render_smpl
        render_smpl_rgba[:, :, 3][rn.visibility_image != 4294967295] = 255

        '''
        ax_img = plt.subplot(2, 2, 1)
        ax_img.imshow(np.transpose(image, (1, 2, 0)))
        plt.axis('off')
        ax_smpl = plt.subplot(2, 2, 2)
        ax_smpl.imshow(render_smpl_rgba)
        plt.axis('off')
        '''

        return img_orig, img_resized, img_smpl, render_smpl_rgba

        # img_uv = np.transpose(uvimage_front[0].cpu().numpy(), (1, 2, 0))
        # # img_uv = resize(img_uv, (img_uv.shape[0], img_uv.shape[1]), anti_aliasing=True)
        # img_uv[img_uv == 0] = img_show[img_uv == 0]

        # plt.show()

        # save_path = './notebooks/output/upimgs/'
        save_path = './notebooks/output/demo_results-v2/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        matplotlib.image.imsave(save_path + 'img_' + draw_id + '.png', img_orig)
        matplotlib.image.imsave(save_path + 'img_smpl_' + draw_id + '.png', img_smpl)
        matplotlib.image.imsave(save_path + 'smpl_' + draw_id + '.png', render_smpl_rgba)

        # output_dir = os.path.split('./notebooks/output/demo_results-v1/' + draw_id + '.pdf')[0]
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #
        # plt.savefig('./notebooks/output/demo_results-v1/' + draw_id + '.pdf', format='pdf', bbox_inches='tight')

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # import pdb; pdb.set_trace()

