from paraview.simple import *
from paraview import servermanager
import argparse
import os

paraview.simple._DisableFirstRenderCameraReset()

pm = servermanager.vtkProcessModule.GetProcessModule()
print("Number of processes:", pm.GetNumberOfLocalPartitions(), flush=True)


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
    print(f"Created out folder at : {Root_folder}/{Out_folder}", flush=True)

print("Reading data", flush=True)
########################
# --- Load data ---
reader = PVDReader(registrationName='output.pvd', FileName=f'{Root_folder}/output.pvd')
reader.PointArrays = ['phase [ ]','velocity [cm/yr]', 'pressure [MPa]']
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
view.CameraPosition = [0, -7.5e3, 2000]
view.CameraFocalPoint = [0, 0, -310]
view.CameraViewUp = [0, 0, 0.5]
view.CameraParallelScale = 3020.3563455796325
view.Update()

# Show box
BoxDisplay = Show(reader, view, 'UniformGridRepresentation')
BoxDisplay.Representation = 'Outline'
########################
# If netcdf with pressure dyn exists, use it 
reader_nc = None
if os.path.exists(f"{Root_folder}/Netcdf/output_pdyn.nc") is True:
    reader_nc = NetCDFReader(registrationName='output_pdyn.nc', FileName=f'{Root_folder}/Netcdf/output.nc')
    reader_nc.Dimensions = '(z, y, x)'
    reader_nc.SphericalCoordinates = 0
    Mantle_nc = IsoVolume(registrationName='Mantle', Input=reader_nc)
    Mantle_nc.InputScalars = ['POINTS', 'phase [ ]']
    Mantle_nc.ThresholdRange = [0.7, 1.5]
    Lithos_nc = IsoVolume(registrationName='Mantle', Input=reader_nc)
    Lithos_nc.InputScalars = ['POINTS', 'phase [ ]']
    Lithos_nc.ThresholdRange = [1.5, 12]
    # # Mantle_nc.UpperThreshold = 1.5
    # Mantle_ncDisplay = Show(Mantle_nc, view, 'StructuredGridRepresentation')
    # Mantle_ncDisplay.SetRepresentationType('Surface')
    # ColorBy(Mantle_ncDisplay, ('POINTS', 'pressure_dyn [MPa]'))
print("Computing figure", flush=True)

########################
# Surface plot
########################
# Transform function to put offsets
########################
# surfDisplay = Show(reader_surf, view, 'StructuredGridRepresentation')
# surfDisplay.SetRepresentationType('Surface')
# ColorBy(surfDisplay, ('POINTS', 'amplitude [km]'))


surf = Transform(registrationName='surf', Input=reader_surf)
surf.Transform = 'Transform'
surf.Transform.Scale = [1.0, 1.0, 1.0]
surf.Transform.Translate = [0.0, 0.0, 20.0]
surfDisplay = Show(surf, view, 'GeometryRepresentation')
surfDisplay.SetRepresentationType('Surface')
ColorBy(surfDisplay, ('POINTS', 'amplitude [km]'))


sliceSurf = Slice(Input=reader_surf)
sliceSurf.SliceType = 'Plane' 
sliceSurf.SliceType.Origin = [0.0, -740.0, 0.0]
sliceSurf.SliceType.Normal = [0.0, 1.0, 0.0]
sliceSurf.UpdatePipeline()
sliceSurf_translate = Transform(registrationName='sliceSurf_translate', Input=sliceSurf)
sliceSurf_translate.Transform = "Transform"
sliceSurf_translate.Transform.Scale = [1.0, 1.0, 10.0]
sliceSurf_translate.Transform.Translate = [0.0, 0.0, 250.0]
sliceSurf_translateDisplay = Show(sliceSurf_translate, view, 'GeometryRepresentation')
sliceSurf_translateDisplay.SetRepresentationType('Wireframe')
ColorBy(sliceSurf_translateDisplay, ('POINTS', 'amplitude [km]'))
sliceSurf_translateDisplay.LineWidth = 8.0

sliceSurf2 = Slice(Input=reader_surf)
sliceSurf2.SliceType = 'Plane' 
sliceSurf2.SliceType.Origin = [0.0, -370.0, 0.0]
sliceSurf2.SliceType.Normal = [0.0, 1.0, 0.0]
sliceSurf2.UpdatePipeline()
sliceSurf2_translate = Transform(registrationName='sliceSurf2_translate', Input=sliceSurf2)
sliceSurf2_translate.Transform = "Transform"
sliceSurf2_translate.Transform.Scale = [1.0, 1.0, 10.0]
sliceSurf2_translate.Transform.Translate = [0.0, 0.0, 250.0]
sliceSurf2_translateDisplay = Show(sliceSurf2_translate, view, 'GeometryRepresentation')
sliceSurf2_translateDisplay.SetRepresentationType('Wireframe')
ColorBy(sliceSurf2_translateDisplay, ('POINTS', 'amplitude [km]'))
sliceSurf2_translateDisplay.LineWidth = 8.0

########################
# Create thresholds 
########################
# Lithos
# Lithos = Threshold(registrationName='Lithos', Input=reader)
# Lithos.Scalars = ['POINTS', 'phase [ ]']
# Lithos.LowerThreshold = 1.5
# Lithos.UpperThreshold = 10.0

Lithos = IsoVolume(registrationName='Lithos', Input=reader)
Lithos.InputScalars = ['POINTS', 'phase [ ]']
Lithos.ThresholdRange = [1.5, 10]
# Lithos.LowerThreshold = 1.5
# Lithos.UpperThreshold = 10.0

# surfaceLithos = ExtractRegionSurface(Input=Lithos)
# smoothLithos = Smooth(Input=surfaceLithos)
# smoothLithos.NumberofIterations = 30
# smoothLithos.RelaxationFactor = 0.1

# show smooth Lithos in view
# smoothLithosDisplay = Show(smoothLithos, view, 'UnstructuredGridRepresentation')
# smoothLithosDisplay.Representation = 'Surface'
# smoothLithosDisplay.SelectInputVectors = ['POINTS', 'velocity [cm/yr]']
# ColorBy(smoothLithosDisplay, ('POINTS', 'velocity [cm/yr]', 'Magnitude'))

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
streamTracer1 = StreamTracer(registrationName='StreamTracer1', Input=Mantle,
    SeedType='Line')
streamTracer1.Vectors = ['POINTS', 'velocity [cm/yr]']
streamTracer1.MaximumStreamlineLength = 6000.0
streamTracer1.SeedType.Resolution = 40

# init the 'Line' selected for 'SeedType'
streamTracer1.SeedType.Point1 = [-3000.0, -500.0, -330.0]
streamTracer1.SeedType.Point2 = [3000.0, -500.0, -330.0]

tube = Tube(Input=streamTracer1)
tube.Radius = 4.0          # controls thickness
tube.NumberofSides = 12     # smoothness
tube.UpdatePipeline()

tubeDisplay = Show(tube, view)
tubeDisplay.SetRepresentationType('Surface')
ColorBy(tubeDisplay, ('POINTS', 'velocity [cm/yr]', 'Magnitude'))

# streamTracer1Display = Show(streamTracer1, view, 'GeometryRepresentation')
# streamTracer1Display.Representation = 'Surface'
# ColorBy(streamTracer1Display, ('POINTS', 'velocity [cm/yr]', 'Magnitude'))
######################
view.Update()


########################
# Compute dynamic pressure at z=-120
########################
# Create slice at z = -200 km
z_slice = -300.0
slice1 = Slice(Input=reader)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [0.0, 0.0, z_slice]   # z = -120 km
slice1.SliceType.Normal = [0.0, 0.0, 1.0]      # Horizontal slice

slice1.UpdatePipeline()

z_translate = -1000.0
slice1_translate = Transform(registrationName='slice1_translate', Input=slice1)
slice1_translate.Transform = 'Transform'
slice1_translate.Transform.Scale = [1.0, 1.0, 1.0]
slice1_translate.Transform.Translate = [0.0, 0.0, z_translate]
# Transform function to put offsets
########################

if reader_nc is not None:
    slice_pdyn = Slice(Input=Mantle_nc)
    slice_pdyn.SliceType = 'Plane'
    slice_pdyn.SliceType.Origin = [0.0, 0.0, z_slice]   # z = -120 km
    slice_pdyn.SliceType.Normal = [0.0, 0.0, 1.0]      # Horizontal slice

    slice_pdyn_translate = Transform(registrationName='slice_pdyn_translate', Input=slice_pdyn)
    slice_pdyn_translate.Transform = 'Transform'
    slice_pdyn_translate.Transform.Scale = [1.0, 1.0, 1.0]
    slice_pdyn_translate.Transform.Translate = [0.0, 0.0, z_translate]
    # Show result
    dyn_pressureDisplay = Show(slice_pdyn_translate, view, 'GeometryRepresentation')
    dyn_pressureDisplay.SetRepresentationType('Surface')
    ColorBy(dyn_pressureDisplay, ('POINTS', 'pressure_dyn [MPa]'))



else:
    # Compute dynamic pressure on slice
    # --- Integrate variables ---
    integrate = IntegrateVariables(Input=slice1_translate)
    integrate.UpdatePipeline()
    # --- Fetch result to Python ---
    data = servermanager.Fetch(integrate)
    # --- Extract values ---
    point_data = data.GetPointData()
    # Pressure integral
    pressure_array = point_data.GetArray("pressure [MPa]")
    sum_pressure = pressure_array.GetTuple(0)[0]
    # Area
    cell_data = data.GetCellData()
    area_array = cell_data.GetArray("Area")
    area = area_array.GetTuple(0)[0]
    # Litho pressure
    litho_pressure = sum_pressure/area
    # Dyn pressure
    # Create Calculator filter
    calculator = Calculator(Input=slice1_translate)
    # Define new field
    calculator.ResultArrayName = "pressure_dyn [MPa]"
    # Subtract lithostatic pressure
    calculator.Function = f'"pressure [MPa]" - {litho_pressure}'
    calculator.UpdatePipeline()
    # Show result
    dyn_pressureDisplay = Show(calculator, view, 'GeometryRepresentation')
    dyn_pressureDisplay.SetRepresentationType('Surface')
    ColorBy(dyn_pressureDisplay, ('POINTS', 'pressure_dyn [MPa]'))


# Put stream tracers on the Pdyn slice 
# First : project velocity on the plane
calculator2 = Calculator(Input=slice1_translate)
calculator2.ResultArrayName = "vel_proj"
calculator2.Function = '"velocity [cm/yr]_X"*iHat+"velocity [cm/yr]_Y"*jHat'
calculator2.UpdatePipeline()

streamTracerSlice = StreamTracer(registrationName='streamTracerSlice', Input=calculator2,
    SeedType='Line')
streamTracerSlice.Vectors = ['POINTS', 'vel_proj']
streamTracerSlice.MaximumStreamlineLength = 6000.0
streamTracerSlice.SeedType.Resolution = 10

# init the 'Line' selected for 'SeedType'
streamTracerSlice.SeedType.Point1 = [-3000.0, -500.0, z_slice+z_translate]
streamTracerSlice.SeedType.Point2 = [3000.0, -500.0, z_slice+z_translate]

tube = Tube(Input=streamTracerSlice)
tube.Radius = 2.0          # controls thickness
tube.NumberofSides = 12     # smoothness
tube.UpdatePipeline()

tubeDisplay = Show(tube, view)
tubeDisplay.SetRepresentationType('Surface')
tubeDisplay.DiffuseColor = [0.0, 0.0, 0.0]

######################
view.Update()


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
velocitycmyrLUT.RescaleTransferFunction(0.0, 5.0)
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
surfDisplay.SetScalarBarVisibility(view, True)
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

# Dynamic pressure
# Import colormap
# Optional: show colorbar
dyn_pressureDisplay.SetScalarBarVisibility(view, True)
# get 2D transfer function for 'amplitudekm'
dyn_pressureTF2D = GetTransferFunction2D('pressure_dyn [MPa]')
# get color transfer function/color map for 'amplitudekm'
dyn_pressureLUT = GetColorTransferFunction('pressure_dyn [MPa]')
dyn_pressureLUT.TransferFunction2D = dyn_pressureTF2D
dyn_pressureLUT.RescaleTransferFunction(-15.0, 15.0)
dyn_pressureLUTColorBar = GetScalarBar(dyn_pressureLUT, view)
dyn_pressureLUTColorBar.Title = 'Dynamic pressure [MPa]'
dyn_pressureLUTColorBar.TitleFontSize = 25
dyn_pressureLUTColorBar.LabelFontSize = 25
dyn_pressureLUTColorBar.WindowLocation = 'Any Location'
dyn_pressureLUTColorBar.Position = [0.4, 0.7]
dyn_pressureLUTColorBar.ScalarBarLength = 0.25
dyn_pressureLUTColorBar.ScalarBarThickness = 20
# dyn_pressureLUT.ApplyPreset('Blue Orange (divergent)', True)
dyn_pressureLUT.ApplyPreset("bam", True)

view.Update()
########################
# Set grid
# ########################
view.AxesGrid.Visibility = 1

view.AxesGrid.UseCustomBounds = 1
view.AxesGrid.CustomBounds = [-3000.0, 3000.0, -750.0, 750.0, -660.0, 0.0]

# No Y label for 2D plots
view.AxesGrid.YAxisUseCustomLabels = 1
view.AxesGrid.YAxisLabels = [-750.0, 0.0, 750.0]


view.AxesGrid.ZAxisUseCustomLabels = 1
view.AxesGrid.ZAxisLabels = [-660.0, -330.0, 0.0]

view.AxesGrid.XAxisUseCustomLabels = 1
view.AxesGrid.XAxisLabels = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]

view.AxesGrid.XLabelFontSize = 50
view.AxesGrid.YLabelFontSize = 50
view.AxesGrid.ZLabelFontSize = 50
view.AxesGrid.XTitleFontSize = 50
view.AxesGrid.YTitleFontSize = 50
view.AxesGrid.ZTitleFontSize = 50

print("Saving plots", flush=True)
########################
# --- Render & save ---
########################
view.StillRender()
view.ViewSize = [1920, 1080]
# SaveScreenshot("phase_y_view_test_3D_pointcloud.png", view, ImageResolution=[1920*2, 1080*2])

# save animation
SaveAnimation(f"{Root_folder}/{Out_folder}/frame_3D.png", view, ImageResolution=[1920*2, 1080*2])
# Adjust camera
# convert to video : ffmpeg -framerate 5 -i frame_3D.%04d.png -c:v libx264 -pix_fmt yuv420p animation.mp4
# or : ffmpeg -framerate 5 -i frame_3D.%04d.png -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p animation.mp4
#####################
