import kinpy as kp
import vtk
from vtk import vtkPNGWriter
import os.path as osp

def draw_imgs(angles, chain, tmp_fp, resolution):
    for i, th in enumerate(angles):
        viz = kp.Visualizer(win_size=(resolution, resolution//4*3))
        viz._ren.ResetCamera(-0.75, 0.75, -0.75, 0.75, 0, 0.75)
        viz._ren.GetActiveCamera().Azimuth(90)
        viz._ren.GetActiveCamera().Roll(-90)
        viz._win.SetOffScreenRendering(1)
        viz._w2if = vtk.vtkWindowToImageFilter()
        viz._w2if.SetInput(viz._win)

        ret = chain.forward_kinematics(th)

        viz.add_robot(ret, chain.visuals_map())

        viz._win.Render()
        viz._w2if.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(osp.join(tmp_fp, '{:04}.png'.format(i)))
        writer.SetInputConnection(viz._w2if.GetOutputPort())
        writer.Write()

        viz._ren.RemoveAllViewProps()

        del viz._ren, viz._win, viz._w2if, viz

def render_ret(ret, chain, img_fp, resolution):
    viz = kp.Visualizer(win_size=(resolution, resolution//4*3))
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