import kinpy as kp
from kinpy.chain import Chain
import vtk
from vtk import vtkPNGWriter
import os.path as osp
from typing import List

# from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter


def draw_imgs(
    angles: List[dict],
    chain: Chain,
    tmp_fp: str,
    resolution: int,
    smooth: bool,
    joint_keys: List[str],
    camera_dist: float = 0.75,  # distance from the camera to the robot
):
    viz = kp.Visualizer(win_size=(resolution, resolution // 4 * 3))
    viz._ren.ResetCamera(
        -camera_dist,
        camera_dist,
        -camera_dist,
        camera_dist,
        camera_dist / 3,
        camera_dist,
        # 0,
        # 1,
    )
    viz._ren.GetActiveCamera().Azimuth(90)
    viz._ren.GetActiveCamera().Roll(-90)
    # viz._win.SetOffScreenRendering(1)
    if smooth:
        print("median filtering....")
        for ki, k in enumerate(joint_keys):
            values = np.array([th[k] for th in angles])
            values = savgol_filter(values, 25, 2)
            for thi, th in enumerate(angles):
                th[k] = values[thi]
                # angles[thi] = th
        print("filtering done")

    for i, th in enumerate(angles):
        if i % 1 == 0:
            ret = chain.forward_kinematics(th)

            viz.add_robot(ret, chain.visuals_map())

            viz._win.Render()
            viz._w2if = vtk.vtkWindowToImageFilter()
            viz._w2if.SetInput(viz._win)
            viz._w2if.Update()

            writer = vtkPNGWriter()
            writer.SetFileName(osp.join(tmp_fp, "{:04}.png".format(i)))
            writer.SetInputConnection(viz._w2if.GetOutputPort())
            writer.Write()

            viz._ren.RemoveAllViewProps()

    del viz._ren, viz._win, viz._w2if, viz


def render_ret(ret, chain, img_fp, resolution):
    viz = kp.Visualizer(win_size=(resolution, resolution // 4 * 3))
    viz._ren.ResetCamera(-0.75, 0.75, -0.75, 0.75, 0, 0.75)
    viz._ren.GetActiveCamera().Azimuth(90)
    viz._ren.GetActiveCamera().Roll(-90)
    viz._win.SetOffScreenRendering(1)
    viz._w2if = vtk.vtkWindowToImageFilter()
    viz._w2if.SetInput(viz._win)

    viz.add_robot(ret, chain.visuals_map())

    viz._win.Render()
    viz._w2if.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(img_fp)
    writer.SetInputConnection(viz._w2if.GetOutputPort())
    writer.Write()

    viz._ren.RemoveAllViewProps()

    del viz
