from paraview.simple import *
from paraview import servermanager
import numpy as np
import argparse
import os

paraview.simple._DisableFirstRenderCameraReset()

pm = servermanager.vtkProcessModule.GetProcessModule()
print("PARAVIEW plotting")
print("Number of processes:", pm.GetNumberOfLocalPartitions())

parser = argparse.ArgumentParser()
parser.add_argument('-I', dest="Root_folder")
parser.add_argument('-O', dest="Out_folder")
args = parser.parse_args()

Root_folder = args.Root_folder
Out_folder = args.Out_folder

if Root_folder is None:
    raise("Need to specifiy -I argument. Path of Root folder")

if Out_folder is None:
    Out_folder = "Figures_anim"

if os.path.exists(f"{Root_folder}/{Out_folder}") is False:
    os.mkdir(f"{Root_folder}/{Out_folder}")
    print(f"Created out folder at : {Root_folder}/{Out_folder}")


# print(Out_folder)
print("Reading data")
########################
# --- Load data ---
reader = PVDReader(registrationName='output.pvd', FileName=f'{Root_folder}/output.pvd')
reader.PointArrays = ['phase [ ]','velocity [cm/yr]']
# --- Load surface data ---
reader_surf = PVDReader(registrationName='output_surf.pvd', FileName=f'{Root_folder}/output_surf.pvd')
reader_surf.PointArrays = ['amplitude [km]']
########################

# get the time-keeper
timeKeeper = GetTimeKeeper()
# get times values (in My) for each timesteps
times = timeKeeper.TimestepValues

# get animation scene
animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()
# set the scene at timestep 25
animationScene1.AnimationTime = times[10]

########################
# --- Create view ---
########################
view = GetActiveViewOrCreate('RenderView')
view.CameraPosition = [0, -7.5e3, 0]
view.CameraFocalPoint = [0, 0, -310]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = 3020.3563455796325
view.Update()

# Show box
BoxDisplay = Show(reader, view, 'UniformGridRepresentation')
BoxDisplay.Representation = 'Outline'
########################
# If netcdf with pressure dyn exists, use it 
reader_nc = None
if os.path.exists(f"{Root_folder}/Netcdf/output.nc") is True:
    reader_nc = NetCDFReader(registrationName='output.nc', FileName=f'{Root_folder}/Netcdf/output.nc')
    reader_nc.Dimensions = '(z, y, x)'
    reader_nc.SphericalCoordinates = 0
    Mantle_nc = IsoVolume(registrationName='Mantle', Input=reader_nc)
    Mantle_nc.InputScalars = ['POINTS', 'phase [ ]']
    Mantle_nc.ThresholdRange = [0.7, 1.5]
    # Mantle_nc.UpperThreshold = 1.5
    Mantle_ncDisplay = Show(Mantle_nc, view, 'StructuredGridRepresentation')
    Mantle_ncDisplay.SetRepresentationType('Surface')
    ColorBy(Mantle_ncDisplay, ('POINTS', 'pressure_dyn [MPa]'))
print("Computing figure")

########################
# Create thresholds 
########################
# Lithos
Lithos = IsoVolume(registrationName='Lithos', Input=reader)
Lithos.InputScalars = ['POINTS', 'phase [ ]']
Lithos.ThresholdRange = [1.5, 10]

# show Lithos in view
LithosDisplay = Show(Lithos, view, 'UnstructuredGridRepresentation')
LithosDisplay.Representation = 'Surface'
LithosDisplay.SelectInputVectors = ['POINTS', 'velocity [cm/yr]']
ColorBy(LithosDisplay, ('POINTS', 'velocity [cm/yr]', 'Magnitude'))

# Mantle
Mantle = Threshold(registrationName='Mantle', Input=reader)
Mantle.Scalars = ['POINTS', 'phase [ ]']
Mantle.LowerThreshold = 0.7
Mantle.UpperThreshold = 1.5
# show data in view
# MantleDisplay = Show(Mantle, view, 'UnstructuredGridRepresentation')
# MantleDisplay.Representation = 'Surface'
# MantleDisplay.SelectInputVectors = ['POINTS', 'velocity [cm/yr]']
# ColorBy(MantleDisplay, ('POINTS', 'velocity [cm/yr]', 'Magnitude'))
#############################
view.Update()
######################

########################
### Stream tracer 
########################
y_slice = 0
slice1 = Slice(Input=Mantle)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [0.0, y_slice, 0.0] 
slice1.SliceType.Normal = [0.0, 1.0, 0.0]      # Vertical slice

slice1.UpdatePipeline()

# Transform function to put offsets
########################
slice1_translate = Transform(registrationName='slice1_translate', Input=slice1)
slice1_translate.Transform = 'Transform'
slice1_translate.Transform.Scale = [1.0, 1.0, 1.0]
y_translate = -2.0
slice1_translate.Transform.Translate = [0.0, y_translate, 0.0]

calculator2 = Calculator(Input=slice1_translate)
calculator2.ResultArrayName = "vel_proj"
# Subtract lithostatic pressure
calculator2.Function = '("velocity [cm/yr]_X")*iHat + ("velocity [cm/yr]_Z")*kHat'
calculator2.UpdatePipeline()

streamTracerSlice = StreamTracer(registrationName='streamTracerSlice', Input=calculator2,
    SeedType='Line')
streamTracerSlice.Vectors = ['POINTS', 'vel_proj']
streamTracerSlice.MaximumStreamlineLength = 6000.0
streamTracerSlice.SeedType.Resolution = 15

# init the 'Line' selected for 'SeedType'
streamTracerSlice.SeedType.Point1 = [-3000.0, y_slice+y_translate, -330]
streamTracerSlice.SeedType.Point2 = [3000.0, y_slice+y_translate, -330]

tube = Tube(Input=streamTracerSlice)
tube.Radius = 2.0          # controls thickness
tube.NumberofSides = 12     # smoothness
tube.UpdatePipeline()

tubeDisplay = Show(tube, view)
tubeDisplay.SetRepresentationType('Surface')
ColorBy(tubeDisplay, ('POINTS', 'vel_proj', 'Magnitude'))
# tubeDisplay.DiffuseColor = [0.0, 0.0, 0.0]

######################
view.Update()

########################
# Surface plot
########################
# Transform function to put offsets
########################
transform1 = Transform(registrationName='Transform1', Input=reader_surf)
transform1.Transform = 'Transform'
transform1.Transform.Scale = [1.0, 1.0, 20.0]
transform1.Transform.Translate = [0.0, 0.0, 300.0]
transform1Display = Show(transform1, view, 'GeometryRepresentation')
transform1Display.SetRepresentationType('Wireframe')
ColorBy(transform1Display, ('POINTS', 'amplitude [km]'))
transform1Display.LineWidth = 5.0


########################
# Modify colorbars and min max values
########################

ImportPresets("/home/bouchary/Crameri_color_maps/bam/bam_PARAVIEW.xml")
ImportPresets("/home/bouchary/Crameri_color_maps/cork/cork_PARAVIEW.xml")
ImportPresets("/home/bouchary/Crameri_color_maps/vik/vik_PARAVIEW.xml")
ImportPresets("/home/bouchary/Crameri_color_maps/davos/davos_PARAVIEW.xml")
ImportPresets("/home/bouchary/Crameri_color_maps/batlow/batlow_PARAVIEW.xml")

# Velocity
# get 2D transfer function for 'velocitycmyr'
velocitycmyrTF2D = GetTransferFunction2D('velocitycmyr')

# get color transfer function/color map for 'velocitycmyr'
velocitycmyrLUT = GetColorTransferFunction('velocitycmyr')
velocitycmyrLUT.TransferFunction2D = velocitycmyrTF2D
velocitycmyrLUT.ScalarRangeInitialized = 1.0
velocitycmyrLUT.RescaleTransferFunction(0.0, 3.5)
velocitycmyrLUTColorBar = GetScalarBar(velocitycmyrLUT, view)
velocitycmyrLUTColorBar.Title = 'velocity [cm/yr]'
velocitycmyrLUTColorBar.TitleFontSize = 25
velocitycmyrLUTColorBar.LabelFontSize = 25
velocitycmyrLUTColorBar.WindowLocation = 'Any Location'
velocitycmyrLUTColorBar.Position = [0.6, 0.7]
velocitycmyrLUTColorBar.ScalarBarLength = 0.25
velocitycmyrLUTColorBar.ScalarBarThickness = 20
velocitycmyrLUTColorBar.ComponentTitle = ''
velocitycmyrLUT.ApplyPreset("vik", True)

# Topography
# Optional: show colorbar
transform1Display.SetScalarBarVisibility(view, True)
# get 2D transfer function for 'amplitudekm'
amplitudekmTF2D = GetTransferFunction2D('amplitudekm')
# get color transfer function/color map for 'amplitudekm'
amplitudekmLUT = GetColorTransferFunction('amplitudekm')
amplitudekmLUT.TransferFunction2D = amplitudekmTF2D
amplitudekmLUT.RescaleTransferFunction(-3.5, 1.5)
amplitudekmLUTColorBar = GetScalarBar(amplitudekmLUT, view)
amplitudekmLUTColorBar.Title = 'amplitude [km]'
amplitudekmLUTColorBar.TitleFontSize = 25
amplitudekmLUTColorBar.LabelFontSize = 25
amplitudekmLUTColorBar.WindowLocation = 'Any Location'
amplitudekmLUTColorBar.Position = [0.8, 0.7]
amplitudekmLUTColorBar.ScalarBarLength = 0.25
amplitudekmLUTColorBar.ScalarBarThickness = 20
# amplitudekmLUT.ApplyPreset('Blue - Green - Orange', True)
amplitudekmLUT.ApplyPreset("batlow", True)

if reader_nc is not None:
    # Dynamic pressure
    # get 2D transfer function for 'pressure_dynMPa'
    pressure_dynMPaTF2D = GetTransferFunction2D('pressure_dynMPa')
    # get color transfer function/color map for 'pressure_dynMPa'
    pressure_dynMPaLUT = GetColorTransferFunction('pressure_dynMPa')
    pressure_dynMPaLUT.TransferFunction2D = pressure_dynMPaTF2D
    pressure_dynMPaLUT.ScalarRangeInitialized = 1.0
    pressure_dynMPaLUT.RescaleTransferFunction(-20, 20)
    pressure_dynMPaLUTColorbar = GetScalarBar(pressure_dynMPaLUT, view)
    pressure_dynMPaLUTColorbar.Title = 'Dynamic pressure [MPa]'
    pressure_dynMPaLUTColorbar.TitleFontSize = 25
    pressure_dynMPaLUTColorbar.LabelFontSize = 25
    pressure_dynMPaLUTColorbar.WindowLocation = 'Any Location'
    pressure_dynMPaLUTColorbar.Position = [0.4, 0.7]
    pressure_dynMPaLUTColorbar.ScalarBarLength = 0.25
    pressure_dynMPaLUTColorbar.ScalarBarThickness = 20
    pressure_dynMPaLUTColorbar.ComponentTitle = ''
    pressure_dynMPaLUT.ApplyPreset("bam", True)


view.Update()
########################
# Set grid
# ########################
view.AxesGrid.Visibility = 1

view.AxesGrid.UseCustomBounds = 1
view.AxesGrid.CustomBounds = [-3000.0, 3000.0, 0.0, 0.0, -660.0, 40.0]

# No Y label for 2D plots
view.AxesGrid.YAxisUseCustomLabels = 1

view.AxesGrid.ZAxisUseCustomLabels = 1
view.AxesGrid.ZAxisLabels = [-660.0, -330.0, 0.0]

view.AxesGrid.XAxisUseCustomLabels = 1
view.AxesGrid.XAxisLabels = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]

view.AxesGrid.XLabelFontSize = 50
view.AxesGrid.ZLabelFontSize = 50
view.AxesGrid.XTitleFontSize = 50
view.AxesGrid.ZTitleFontSize = 50

print("Saving figures")
########################
# --- Render & save ---
########################
view.StillRender()
view.ViewSize = [1920, 1080]
# SaveScreenshot("phase_y_view_test.png", view, ImageResolution=[1920*2, 1080*2])

# save animation
SaveAnimation(f"{Root_folder}/{Out_folder}/frame_2D.png", view, ImageResolution=[1920*2, 1080*2])
# Adjust camera
#####################
