import os.path as osp
import argparse

import numpy as np
import torch

import smplx


def main(
    model_folder,
    model_type="smplx",
    ext="npz",
    gender="neutral",
    plot_joints=False,
    num_betas=10,
    sample_shape=True,
    sample_expression=True,
    num_expression_coeffs=10,
    plotting_module="pyrender",
    use_face_contour=False,
    pose_index=0,
):

    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext,
    )
    print(model)

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

    pose_path = "./data/reachy/raw/xyzs+reps_{:03}.npz".format(pose_index)

    with np.load(pose_path) as data:
        poses = data["xyzs4smpl"]
        # poses = data['xyzs']
        reachy_pose = poses[100]
        reachy_pose = torch.tensor(reachy_pose, dtype=torch.float32)
        reachy_pose = reachy_pose.unsqueeze(0)

        pose = torch.zeros_like(reachy_pose)
        pose[:, :, 0] = reachy_pose[:, :, 1]
        pose[:, :, 1] = reachy_pose[:, :, 2]
        pose[:, :, 2] = reachy_pose[:, :, 0]

    print(pose)
    print(pose.shape)
    print(pose.dtype)

    output = model.forward(betas=betas, body_pose=pose, return_verts=True)

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print("Vertices shape =", vertices.shape)
    print("Joints shape =", joints.shape)

    if plotting_module == "pyrender":
        import pyrender
        import trimesh

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-X Demo")

    parser.add_argument("--model-folder", required=True, type=str, help="The path to the model folder")
    parser.add_argument(
        "--model-type",
        default="smplx",
        type=str,
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="The type of model to load",
    )
    parser.add_argument("--gender", type=str, default="neutral", help="The gender of the model")
    parser.add_argument(
        "--num-betas",
        default=10,
        type=int,
        dest="num_betas",
        help="Number of shape coefficients.",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        default=10,
        type=int,
        dest="num_expression_coeffs",
        help="Number of expression coefficients.",
    )
    parser.add_argument(
        "--plotting-module",
        type=str,
        default="pyrender",
        dest="plotting_module",
        choices=["pyrender", "matplotlib", "open3d"],
        help="The module to use for plotting the result",
    )
    parser.add_argument("--ext", type=str, default="npz", help="Which extension to use for loading")
    parser.add_argument(
        "--plot-joints",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="The path to the model folder",
    )
    parser.add_argument(
        "--sample-shape",
        default=True,
        dest="sample_shape",
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random shape",
    )
    parser.add_argument(
        "--sample-expression",
        default=True,
        dest="sample_expression",
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random expression",
    )
    parser.add_argument(
        "--use-face-contour",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Compute the contour of the face",
    )
    parser.add_argument("--pose-index", default=0, type=int, dest="pose_index")

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression
    pose_index = args.pose_index

    main(
        model_folder,
        model_type,
        ext=ext,
        gender=gender,
        plot_joints=plot_joints,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        sample_shape=sample_shape,
        sample_expression=sample_expression,
        plotting_module=plotting_module,
        use_face_contour=use_face_contour,
        pose_index=pose_index,
    )
