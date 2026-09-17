import vtk
import numpy as np
import xarray as xr
import os
from lxml import etree as ET
import dask
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import pyvista as pv
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
import cmcrameri.cm as cmc
# Safe rule of thumb
import psutil
available_gb = psutil.virtual_memory().available / 1e9
peak_per_file_gb = 4.0  # measure this for your data
max_workers = max(1, int(available_gb / peak_per_file_gb))
print("Cluster specs :")
print("Available RAM :", available_gb, "GB. Maximum workers :", max_workers)


# self.pvd_to_xarray_ds(max_workers=max_workers)


class LaMEM_POST:

    def __init__(self, filename, fieldnames=None):
        """
        Class initialization

        :param filename: str, filename of the pvtr file
        :param fieldnames: str or list of str, fields to extract.
            If None, all fields are extracted.
        """
        self.surface = False
        self.filename = filename
        if filename.split(".")[-1] == "vtr":
            self.reader = vtk.vtkXMLRectilinearGridReader()
        elif filename.split(".")[-1] == "pvtr":
            self.reader = vtk.vtkXMLPRectilinearGridReader()
        elif filename.split(".")[-1] == "vtu":
            self.reader = vtk.vtkXMLUnstructuredGridReader()
        elif filename.split(".")[-1] == "pvtu":
            self.reader = vtk.vtkXMLPUnstructuredGridReader()
        elif filename.split(".")[-1] == "vts":
            self.reader = vtk.vtkXMLStructuredGridReader()
            self.surface = True
        elif filename.split(".")[-1] == "pvts":
            self.reader = vtk.vtkXMLPStructuredGridReader()
            self.surface = True

        else:
            raise RuntimeError(f"Did not recognize file extension {filename.split('.')[-1]}")

        if os.path.isfile(self.filename) is True:
            self.reader.SetFileName(self.filename)
        else:
            raise RuntimeError(f"File not found: {self.filename}")

        self.reader.UpdateInformation()

        if isinstance(fieldnames, str):
            self.fieldnames = np.array([fieldnames])
        elif isinstance(fieldnames, (list, np.ndarray)):
            self.fieldnames = np.array(fieldnames)
        else:
            self.fieldnames = fieldnames

        # print("fieldnames", self.fieldnames)
        self.pdata_sel = self.reader.GetPointDataArraySelection()
        self.point_fields = np.array([self.pdata_sel.GetArrayName(i)
                        for i in range(self.pdata_sel.GetNumberOfArrays())])

        if isinstance(self.fieldnames, (str, list, np.ndarray)):
            self.extract_field()

        elif not fieldnames:
            self.fieldnames = np.copy(self.point_fields)

        else:
            raise TypeError("fieldnames variable need to be a str or list of str")

        self.reader.Update()

        self.output = self.reader.GetOutput()
        self.pdata = self.output.GetPointData()

        self.N = self.output.GetNumberOfPoints()

        if self.surface is False:
            self.nx, self.ny, self.nz = self.output.GetDimensions()
            self.x = self.vtk_to_numpy_array(self.output.GetXCoordinates())
            self.y = self.vtk_to_numpy_array(self.output.GetYCoordinates())
            self.z = self.vtk_to_numpy_array(self.output.GetZCoordinates())

        else:
            # Need to convert structured grid to rectilinear grid
            dims = [0, 0, 0]
            self.output.GetDimensions(dims)
            self.nx, self.ny, self.nz = dims
            pts = self.output.GetPoints()
            points = self.vtk_to_numpy_array(pts.GetData())
            points = points.reshape((self.ny, self.nx, 3))
            self.x = points[0, :, 0]
            self.y = points[:, 0, 1]
            self.z = [0.0]
            # Put a dummy z coord : 1 cell
            assert self.nz == 1

            rg = vtk.vtkRectilinearGrid()
            rg.SetDimensions(self.nx, self.ny, 2)
            rg.SetXCoordinates(self.numpy_to_vtk_array(self.x))
            rg.SetYCoordinates(self.numpy_to_vtk_array(self.y))
            rg.SetZCoordinates(self.numpy_to_vtk_array([0.0, 1.0]))

            self.output = rg

        self.ds = None
        self.comp = "all"

    def extract_field(self, comp="all"):

        self.pdata_sel.DisableAllArrays()
        if isinstance(self.fieldnames, str):
            if not self.fieldnames in self.point_fields:
                raise AttributeError(f"Field {self.fieldnames} does not exist." \
                f"Available fields : {self.point_fields}")
            else:
                self.pdata_sel.EnableArray(self.fieldnames)

        elif isinstance(self.fieldnames, (list, np.ndarray)):
            for fieldname in self.fieldnames:
                if not fieldname in self.point_fields:
                    raise AttributeError(f"Field {fieldname} does not exist." \
                    f"Available fields : {self.point_fields}")
                else:
                    self.pdata_sel.EnableArray(fieldname)

        if comp == "all":
            pass
        elif isinstance(comp, dict):
            self.comp = comp
            for key in self.comp.keys():
                if key not in self.fieldnames:
                    print(f"Did not find {key} in fieldnames")
        else:
            raise RuntimeError('comp must be a dict, or set to "all"')


    def vtk_to_numpy_array(self, vtk_array, dims=None, n_comp=None):
        """
        Converts a single vtk array to a numpy array

        :param vtk_array: vtk array
        :param dims: tuple, dimensions of the array.
                    If None, 1D, otherwise, 3D (nx, ny, nz)
        """
        np_arr = np.array([vtk_array.GetValue(i) for i in range(vtk_array.GetNumberOfValues())])
        if isinstance(dims, tuple):
            if isinstance(n_comp, int):
                np_arr = np_arr.reshape((n_comp, self.N))
                np_arr = np_arr.reshape((dims[-1], dims[1], dims[0], n_comp))
            else:
                np_arr = np_arr.reshape(dims[::-1])
        return np_arr

    def vtk_to_xarray(self, transpose=True):
        self.transpose = transpose
        point_data = {}        
        for i in range(self.pdata.GetNumberOfArrays()):
            name = self.pdata.GetArrayName(i)
            arr = self.pdata.GetArray(i)
            n_comp = arr.GetNumberOfComponents()
            filt_comp=False
            if self.comp == "all":
                pass
            elif isinstance(self.comp, dict):
                if name in self.comp.keys():
                    filt_comp=True
                    if np.any(np.array(self.comp[name])>n_comp):
                        print(f"Field {name} does not have components > {n_comp}")

            if n_comp>1:
                data4d = self.vtk_to_numpy_array(arr,
                                                 dims=(self.nx, self.ny, self.nz),
                                                 n_comp=n_comp)
                for comp in range(n_comp):
                    if filt_comp==True:
                        if comp in self.comp[name]:
                            point_data[name+"[" +str(comp)+"]"] = data4d[..., comp]
                        else:
                            continue
                    else:
                        point_data[name+"[" +str(comp)+"]"] = data4d[..., comp]
            else:
                point_data[name] = self.vtk_to_numpy_array(arr,
                                                           dims=(self.nx, self.ny, self.nz))


        # Transpose for plotting directly with xarray
        if self.transpose:
            self.ds = xr.Dataset(
                {name: (("x", "y", "z"), data.transpose(2,1,0)) for name, data in point_data.items()},
                coords={
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                }
            )
        # Do not transpose to read with Paraview
        else:
            self.ds = xr.Dataset(
                {name: (("z", "y", "x"), data) for name, data in point_data.items()},
                coords={
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                }
            )
        return self.ds

    def numpy_to_vtk_array(self, arr):
        """
        Docstring for numpy_to_vtk_array

        :param arr: 1D numpy array
        """
        vtk_arr = vtk.vtkDoubleArray()
        vtk_arr.SetNumberOfValues(len(arr))
        for i, v in enumerate(arr):
            vtk_arr.SetValue(i, float(v))
        return vtk_arr

    def xarray_to_vtk_array(self, data_arr):
        """
        Convert an xarray data_var to vtk array

        :param ds: xarray dataarray
        :param data_var: str, name of the xarray data_var
        """
        Nx, Ny, Nz = len(data_arr.x), len(data_arr.y), len(data_arr.z)
        if (self.nx, self.ny, self.ny) != (Nx, Ny, Nz):
            UserWarning("" \
            "Xarray dataarray to convert has different coords indices than vtk" \
            f"Dataarray : Nx {Nx}, Ny {Ny}, Nz {Nz}" \
            f"VTK : Nx {self.nx}, Ny {self.ny}, Nz {self.nz}"
            )
        # If it was transposed, need to retranspose for vtk writing
        if self.transpose:
            # Transpose ds xyz coords to vtk zyx
            data_xyz = data_arr.values
            data_zyx = data_xyz.transpose(2, 1, 0)
        else:
            data_zyx = data_arr.values
        # Convert in flat array
        flat = data_zyx.ravel()
        # Create vtk data
        vtk_data = vtk.vtkDoubleArray()
        vtk_data.SetName(data_arr.name)
        vtk_data.SetNumberOfComponents(1)
        vtk_data.SetNumberOfTuples(Nx * Ny * Nz)
        # Fill in values
        for i, v in enumerate(flat):
            vtk_data.SetValue(i, float(v))

        return vtk_data

    def add_arr_to_vtk(self, arr, fieldname, n_comp=1, type="xarray"):
        """
        Docstring for add_arr_to_vtk

        :param arr: 1D numpy array or Xarray dataarray
        :param arrname: str, name of the field
        :param vtk_in: vtk dataset to update
        :param type: str, "xarray" or "numpy", depending on arr encoding
        """
        if type=="xarray":
            vtk_arr = self.xarray_to_vtk_array(arr)
        elif type=="numpy":
            vtk_arr = self.numpy_to_vtk_array(arr)
        vtk_arr.SetNumberOfComponents(n_comp)
        r = self.pdata.AddArray(vtk_arr)
        self.pdata.GetArray(r).SetName(fieldname)

    def write_xarray_to_pvtr(self, ds, retranspose=True, ofilename="output_processed"):
        """
        Docstring for write_xarray_to_pvtr

        :param ds: xarray dataset
        :param retranspose: bool, retranspose xarray dataset to read vtk out in paraview
        :param ofilename: str, filename of output pvtr
        """
        vtk_grid = self.xarray_to_vtk(ds, retranspose)

        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName(ofilename)
        writer.SetInputData(vtk_grid)
        writer.SetInputData()
        writer.Write()
        return

    def write_vtk_to_pvtr(self, ofilename):
        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName(ofilename)
        writer.SetInputData(self.output)
        writer.Write()
        return

    def compute_P_litho(self):
        """
        Compute lithostatic pressure from an xarray dataset

        :param ds: xarray dataset
        """
        return self.ds["pressure [MPa]"].mean(dim=["x","y"])

    def compute_P_dyn(self, merge=False):
        """
        Compute dynamic pressure from an xarray dataset

        :param ds: xarray dataset
        """
        P_litho = self.compute_P_litho()
        self.P_dyn = self.ds["pressure [MPa]"] - P_litho
        self.P_dyn = self.P_dyn.rename("pressure_dyn [MPa]")
        if merge:
            self.ds = xr.merge([
                self.ds,
                self.P_dyn
                ])
        return self.P_dyn

class VTUIO:
    def __init__(self, filename):
        """
        Class initialization
        :param filename: str, filename of the vtu file
        """
        self.filename = filename
        if not os.path.isfile(self.filename):
            raise RuntimeError(f"File not found: {self.filename}")
        self.grid = None
        # Store only phase IDs, not the thresholded meshes
        self._phase_ids = []
        self.ds = xr.Dataset()
 
    def read(self):
        reader = pv.get_reader(self.filename)
        reader.disable_all_cell_arrays()
        self.grid = reader.read()
 
    def extract_phases(self, phases="all"):
        """
        Store which phases to process — does NOT threshold yet.
        :param phases: "all" or list of ints
        """
        if phases == "all":
            self._phase_ids = list(set(self.grid.point_data["Phase"]))
        else:
            self._phase_ids = list(phases)
 
    def define_surface_grid(self, xmin, xmax, dx, ymin, ymax, dy):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.ymin = ymin
        self.ymax = ymax
        self.dy = dy
 
        self.nx = int((xmax - xmin) / dx) + 1
        self.ny = int((ymax - ymin) / dy) + 1
        self.x = np.arange(xmin, xmax + 1, dx)
        self.y = np.arange(ymin, ymax + 1, dy)
 
        self.ds = self.ds.assign_coords(
            x=("x", self.x),
            y=("y", self.y),
        )
 
    def use_grid_from(self, ds_surf):
        """
        Use coordinates from a lighter xarray dataset.
        :param ds: xarray dataset
        """
        self.x = ds_surf.x
        self.y = ds_surf.y
        del ds_surf
        gc.collect()
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.xmin = float(self.x.min())
        self.xmax = float(self.x.max())
        self.dx = float(self.x.diff("x").mean())
        self.ymin = float(self.y.min())
        self.ymax = float(self.y.max())
        self.dy = float(self.y.diff("x").mean())
 
        self.ds = self.ds.assign_coords(
            x=("x", self.x),
            y=("y", self.y),
        )
 
    def compute_surfaces(self, norm="min_max"):
        """
        Compute top/bottom surfaces per phase.
 
        Key memory optimisations vs. the original:
        - Phases are thresholded one at a time and immediately discarded
          (no self.extracted_phases dict holding all meshes simultaneously).
        - Only the three coordinate columns (x, y, z) are extracted as
          read-only float32 arrays — cutting point-array RAM by ~2×.
        - Both bin statistics for a phase are computed from the same
          already-sliced arrays; temporaries are deleted explicitly.
        """
        if norm == "min_max":
            norm_bot = "min"
            norm_top = "max"
        elif norm == "percentiles":
            norm_bot = lambda z: np.percentile(z, 2.5)
            norm_top = lambda z: np.percentile(z, 97.5)
        else:
            raise ValueError(f"Unknown norm: {norm!r}")
 
        # Pre-extract the phase array once so we can mask without threshold()
        all_phases = np.asarray(self.grid.point_data["Phase"])
        # Read all points once as float32 to halve the coordinate RAM
        all_points = np.asarray(self.grid.points, dtype=np.float32)

        del self.grid
        gc.collect()
 
        bin_x = [self.nx, self.ny]
 
        for phase_id in self._phase_ids:
            key = f"phase_{phase_id}"
 
            # Boolean mask — tiny compared to a full threshold copy
            mask = all_phases == phase_id
 
            # Slice only the rows we need, keep as float32
            x_pts = all_points[mask, 0]
            y_pts = all_points[mask, 1]
            z_pts = all_points[mask, 2]
 
            top_surface, _, _, _ = binned_statistic_2d(
                x_pts, y_pts, z_pts,
                statistic=norm_top,
                bins=bin_x,
            )
            self.ds[f"{key}_top [km]"] = (
                ("x", "y"),
                top_surface.astype(np.float32),   # store as float32 too
            )
 
            bot_surface, _, _, _ = binned_statistic_2d(
                x_pts, y_pts, z_pts,
                statistic=norm_bot,
                bins=bin_x,
            )
            self.ds[f"{key}_bot [km]"] = (
                ("x", "y"),
                bot_surface.astype(np.float32),
            )
 
            # Free per-phase temporaries before the next iteration
            del x_pts, y_pts, z_pts, top_surface, bot_surface, mask
 
        # Free the large intermediate arrays
        del all_phases, all_points
 
        return self.ds
    
    def compute_thicknesses(self, phases="all", from_surface=None):
        """
        Docstring for compute_thicknesses
        
        :param phases: list of ints, or "all"
        :param from_surface: xr dataarray, the dataarray of the top surface
        """
        if phases=="all":
            phases=[]
            keys = self.extracted_phases.keys()
            for key in keys:
                phases.append(key.split("_")[-1])

        for key in self.extracted_phases.keys():
            if key.split("_")[-1] in phases:
                if from_surface is None:
                    thickness = self.ds[f"{key}_top [km]"] - self.ds[f"{key}_bot [km]"]
                else:
                    thickness = from_surface - self.ds[f"{key}_bot [km]"]
                self.ds[f"{key}_thick [km]"] = thickness

# class VTUIO:

#     def __init__(self, filename):        
#         """
#         Class initialization

#         :param filename: str, filename of the pvtr file
#         """
#         self.filename = filename

#         if os.path.isfile(self.filename) is False:
#             raise RuntimeError(f"File not found: {self.filename}")
        
#         self.grid = None
#         self.extracted_phases = {}
#         self.ds = xr.Dataset()

#     def read(self):
#         self.grid = pv.read(self.filename)

#     def extract_phases(self, phases="all"):
#         """
#         Docstring for extract_phases
        
#         :param phases: array or list of ints 
#         """
#         if phases == "all":
#             phases = list(set(self.grid.point_data["Phase"]))
#         for phase in phases:
#             self.extracted_phases[f"phase_{phase}"] = (
#                 self.grid
#                 .threshold([phase, phase], scalars="Phase", preference="points")
#             )

#     def define_surface_grid(self, xmin, xmax, dx, ymin, ymax, dy):

#         self.xmin = xmin
#         self.xmax = xmax
#         self.dx = dx
#         self.ymin = ymin
#         self.ymax = ymax
#         self.dy = dy

#         len_x = self.xmax - self.xmin
#         len_y = self.ymax - self.ymin     
#         self.nx = int(len_x/dx)+1
#         self.ny = int(len_y/dy)+1
#         self.x = np.arange(self.xmin, self.xmax+1, self.dx)
#         self.y = np.arange(self.ymin, self.ymax+1, self.dy)

#         self.ds = self.ds.assign_coords(
#             x=("x", self.x),
#             y=("y", self.y),
#         )

#     def use_grid_from(self, ds):
#         """
#         Docstring for use_grid_from
        
#         :param ds: xarray dataset, will use its coords to define surface grid
#         """
#         self.x = ds.x
#         self.y = ds.y
#         self.nx = len(self.x)
#         self.ny = len(self.y)
#         self.xmin = self.x.min()
#         self.xmax = self.x.max()
#         self.dx = self.x.diff("x").mean().item()
#         self.ymin = self.y.min()
#         self.ymax = self.y.max()
#         self.dy = self.y.diff("x").mean().item()

#         self.ds = self.ds.assign_coords(
#             x=("x", self.x),
#             y=("y", self.y),
#         )
        
#     def compute_surfaces(self, norm="min_max"):
#         if norm=="min_max":
#             norm_bot="min"
#             norm_top="max"

#         elif norm=="percentiles":
#             norm_bot = lambda z: np.percentile(z, 2.5)
#             norm_top = lambda z: np.percentile(z, 97.5)

#         for key, phase in self.extracted_phases.items():
#             points = phase.points
#             x_points = points[:,0]
#             y_points = points[:,1]
#             z_points = points[:,2]

#             top_surface, _, _, _ = binned_statistic_2d(
#                 x_points,
#                 y_points,
#                 z_points,
#                 statistic=norm_top,
#                 bins=[self.nx, self.ny]
#             )
#             self.ds[f"{key}_top [km]"] = (("x", "y"), top_surface)

#             bot_surface, _, _, _ = binned_statistic_2d(
#                 x_points,
#                 y_points,
#                 z_points,
#                 statistic=norm_bot,
#                 bins=[self.nx, self.ny]
#             )
#             self.ds[f"{key}_bot [km]"] = (("x", "y"), bot_surface)

#         return self.ds

#     def compute_thicknesses(self, phases="all", from_surface=None):
#         """
#         Docstring for compute_thicknesses
        
#         :param phases: list of ints, or "all"
#         :param from_surface: xr dataarray, the dataarray of the top surface
#         """
#         if phases=="all":
#             phases=[]
#             keys = self.extracted_phases.keys()
#             for key in keys:
#                 phases.append(key.split("_")[-1])

#         for key in self.extracted_phases.keys():
#             if key.split("_")[-1] in phases:
#                 if from_surface is None:
#                     thickness = self.ds[f"{key}_top [km]"] - self.ds[f"{key}_bot [km]"]
#                 else:
#                     thickness = from_surface - self.ds[f"{key}_bot [km]"]
#                 self.ds[f"{key}_thick [km]"] = thickness

class PVDIO:


    def __init__(self, filename):
        if os.path.isfile(filename) is True:
            self.folder, self.filename = os.path.split(filename)
        else:
            raise RuntimeError(f"File not found: {filename}")
        self.vtrfilenames = []
        self.timesteps = np.array([])
        self.vtu = False
        self.read_pvd()
        # Memory of timesteps in pvd
        self.__timesteps = np.copy(self.timesteps)
        self.fields = None
        self.ds = None
        self.vtr_pdata = np.array([])
        self.comp = "all"

    def read_pvd(self):
        """
        Read in PVD file

        """
        self.tree = ET.parse(self.folder+"/"+self.filename)
        root = self.tree.getroot()
        for collection in root.getchildren():
            for dataset in collection.getchildren():
                self.timesteps = np.append(self.timesteps, [float(dataset.attrib['timestep'])])
                self.vtrfilenames.append(dataset.attrib['file'])
                # print(dataset.attrib['file'])
                # print(dataset.attrib['file'].split(".")[-1])
                if dataset.attrib['file'].split(".")[-1] == "vtu":
                    self.vtu=True
                    self.ds = xr.Dataset()
                    self.phases = "all"
                elif dataset.attrib['file'].split(".")[-1] == "pvtu":
                    self.vtu=True
                    self.ds = xr.Dataset()
                    self.phases = "all"

        ## TO DO : init xarray ds with timesteps and return it, after we keep what we want.
        ## def set_timesteps, def set_fields, def update -> take all arguments

    def set_ts_and_fields(self, timesteps="all", fields="all", comp="all"):
        """
        Tells which timesteps and fields to keep

        :param timesteps: int or list of int, indices of the timesteps to keep
        :param fields: str or list of str, fields to keep
        :param comp: dict, of the form {"fieldname":[int1, int2...]}, with ints the component number to keep 
        """

        if isinstance(timesteps, (int,list,np.ndarray)):
            self.timesteps = self.timesteps[timesteps]
            self.vtrfilenames = [self.vtrfilenames[ts] for ts in timesteps]
        elif timesteps == "all":
            pass
        else:
            raise TypeError("timesteps must be a int or list of int")

        if fields == "all":
            pass
        elif isinstance(fields, (str,list)):
            self.fields = np.array(fields)
        else:
            raise TypeError("fields must be a str or list of str")
        
        if comp=="all":
            return
        elif isinstance(comp, dict):
            self.comp = comp
        else:
            raise TypeError("comp must be a dict")
        
    def set_marker_grid(self, xmin, xmax, dx, ymin, ymax, dy):
        if self.vtu==False:
            print("This function should be used only for markers vtu files")
            print("Nothing has been done")
            return
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.ymin = ymin
        self.ymax = ymax
        self.dy = dy

        len_x = self.xmax - self.xmin
        len_y = self.ymax - self.ymin     
        self.nx = int(len_x/dx)+1
        self.ny = int(len_y/dy)+1
        self.x = np.arange(self.xmin, self.xmax+1, self.dx)
        self.y = np.arange(self.ymin, self.ymax+1, self.dy)

        # self.ds = self.ds.assign_coords(
        #     x=("x", self.x),
        #     y=("y", self.y),
        # )

    def use_grid_from(self, ds_surf):
        """
        Docstring for use_grid_from
        
        :param ds: xarray dataset, will use its coords to define surface grid
        """
        if self.vtu==False:
            print("This function should be used only for markers vtu files")
            print("Nothing has been done")
            return
        self.x = ds_surf.x.data
        self.y = ds_surf.y.data
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.dx = ds_surf.x.diff("x").mean().item()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.dy = ds_surf.y.diff("x").mean().item()
        del ds_surf
        gc.collect()

        # self.ds = self.ds.assign_coords(
        #     x=("x", self.x),
        #     y=("y", self.y),
        # )

    def set_marker_phases(self, phases="all"):
        self.phases = phases

    def _load_vtr(self, vtr_filename):
        vtr = LaMEM_POST(f"{self.folder}/{vtr_filename}", self.fields)
        vtr.comp = self.comp
        ds = vtr.vtk_to_xarray()
        del vtr
        gc.collect()
        return ds
    
    def _load_vtu(self, vtu_filename, norm="min_max"):
        vtu_data = VTUIO(f"{self.folder}/{vtu_filename}")
        vtu_data.read()
        vtu_data.extract_phases(phases=self.phases)
        vtu_data.ds = vtu_data.ds.assign_coords(
            x=("x", self.x),
            y=("y", self.y),
        )     
        vtu_data.nx, vtu_data.ny = self.nx, self.ny
        ds = vtu_data.compute_surfaces(norm=norm)  
        ds = ds.load()
        del vtu_data 
        gc.collect()
        return ds


    # def pvd_to_xarray_ds(self, save_vtr=True, norm="min_max", max_workers=4):
    #     """
    #     Convert all VTU/VTR files to a single xarray dataset along the time axis.
 
    #     :param max_workers: int
    #         1  -> fully sequential, minimum RAM (default, recommended on clusters)
    #         N  -> process N files in parallel sub-processes
    #              Rule of thumb: N = floor(node_RAM_gb / peak_RAM_per_file_gb)
    #              On a cluster, request enough memory for N files when submitting
    #              your job (e.g. #SBATCH --mem=N*peak_RAM_per_file).
    #     """
 
    #     if self.vtu:
    #         def loader(fname):
    #             return self._load_vtu(fname, norm=norm)
    #     else:
    #         def loader(fname):
    #             return self._load_vtr(fname)
            
    #     filenames = self.vtrfilenames
 
    #     if max_workers == 1:
    #         # Sequential: one file at a time, GC between each
    #         datasets = []
    #         for fname in filenames:
    #             datasets.append(loader(fname))
    #             gc.collect()
 
    #     else:
    #         # Bounded parallel: at most max_workers files in memory at once.
    #         # ProcessPoolExecutor gives true memory isolation (separate processes)
    #         # and works on any machine or cluster node without extra daemons.
    #         datasets = [None] * len(filenames)
    #         with ProcessPoolExecutor(max_workers=max_workers) as pool:
    #             futures = {
    #                 pool.submit(loader, fname): i
    #                 for i, fname in enumerate(filenames)
    #             }
    #             for future in as_completed(futures):
    #                 i = futures[future]
    #                 datasets[i] = future.result()
    #                 gc.collect()
 
    #     self.ds = xr.concat(
    #         datasets,
    #         dim=xr.DataArray(self.timesteps, dims='t', name='time'),
    #     )
    #     return self.ds

    
    def pvd_to_xarray_ds(self, save_vtr=True, norm="min_max", batch_size=1, vtuio=0):
        
        batch_size = batch_size
        if batch_size==1:
            print("Converting to xarray in sequential.")
            print("To allow threading, consider using batch_size=<num_threads>")

        def make_delayed(filename):
            if self.vtu:
                return dask.delayed(self._load_vtu)(filename, norm=norm)
            else:
                return dask.delayed(self._load_vtr)(filename)

        filenames = self.vtrfilenames

        # if self.vtu == True:
             # Split into batches of at most batch_size
        batches = [filenames[i:i+batch_size] for 
                   i in range(0, len(filenames), batch_size)]
        print("batches :", batches, flush=True)
        datasets = []
        for batch in batches:
            print("Processing batch :", batch, flush=True)
            delayed_batch = [make_delayed(fname) for fname in batch]
            results = dask.compute(*delayed_batch)  # at most batch_size files at once
            datasets.extend(results)
            gc.collect()

        self.ds = xr.concat(
            datasets,
            dim=xr.DataArray(self.timesteps, dims='t', name='time')
        )

        # else:
        #     delayed_datasets = [
        #         dask.delayed(self._load_vtr)(
        #             vtr_filename
        #         )
        #         for vtr_filename in self.vtrfilenames
        #     ]

        #     # Tell xarray these are lazy datasets
        #     datasets = dask.compute(*delayed_datasets)

        #     self.ds = xr.concat(
        #         datasets,
        #         dim=xr.DataArray(self.timesteps, dims='t', name='time')
        #     )

        return self.ds

    def compute_P_litho(self, P_arr_name):
        """
        Compute lithostatic pressure from an xarray dataset

        :param P_arr_name: str, name of the total pressure field
        """
        return self.ds[P_arr_name].mean(dim=["x","y"])

    def compute_P_dyn(self, P_arr_name="pressure [MPa]", merge=False):
        """
        Compute dynamic pressure from an xarray dataset

        :param P_arr_name: str, name of the total pressure field
        :merge: bool, merge the output array in
        """
        P_litho = self.compute_P_litho(P_arr_name)
        self.P_dyn = self.ds[P_arr_name] - P_litho
        self.P_dyn = self.P_dyn.rename("pressure_dyn [MPa]")
        if merge:
            self.ds = xr.merge([
                self.ds,
                self.P_dyn
                ])
        return self.P_dyn

    def numpy_to_vtk_array(self, arr):
        """
        Docstring for numpy_to_vtk_array

        :param arr: 1D numpy array
        """
        vtk_arr = vtk.vtkDoubleArray()
        vtk_arr.SetNumberOfValues(len(arr))
        for i, v in enumerate(arr):
            vtk_arr.SetValue(i, float(v))
        return vtk_arr

    def xarray_to_vtk_array(self, data_arr):
        """
        Convert an xarray data_var to vtk array

        :param ds: xarray dataarray
        :param data_var: str, name of the xarray data_var
        """
        Nx, Ny, Nz = len(data_arr.x), len(data_arr.y), len(data_arr.z)
        if (self.nx, self.ny, self.ny) != (Nx, Ny, Nz):
            UserWarning("" \
            "Xarray dataarray to convert has different coords indices than vtk" \
            f"Dataarray : Nx {Nx}, Ny {Ny}, Nz {Nz}" \
            f"VTK : Nx {self.nx}, Ny {self.ny}, Nz {self.nz}"
            )
        # If it was transposed, need to retranspose for vtk writing
        if self.transpose:
            # Transpose ds xyz coords to vtk zyx
            data_xyz = data_arr.values
            data_zyx = data_xyz.transpose(2, 1, 0)
        else:
            data_zyx = data_arr.values
        # Convert in flat array
        flat = data_zyx.ravel()
        # Create vtk data
        vtk_data = vtk.vtkDoubleArray()
        vtk_data.SetName(data_arr.name)
        vtk_data.SetNumberOfComponents(1)
        vtk_data.SetNumberOfTuples(Nx * Ny * Nz)
        # Fill in values
        for i, v in enumerate(flat):
            vtk_data.SetValue(i, float(v))

        return vtk_data

    def add_array_to_vtk(self, arr, fieldname, type="xarray"):
        """
        Add an array to a vtk dataset

        :param arr: 1D numpy array or Xarray dataarray
        :param arrname: str, name of the field
        :param vtk_in: vtk dataset to update
        :param type: str, "xarray" or "numpy", depending on arr encoding
        """
        for timestep, vtk in enumerate(self.vtr_pdata):
            vtk.add_arr_to_vtk(arr.isel(t=timestep), fieldname, type=type)

    def write_pvd_from_vtk_arrs(self, odir="output", ofilename="output"):
        odir_path = os.getcwd()+"/"+odir
        if os.path.isdir(odir_path) is False:
            print("Creating output dir at ", odir_path)
            os.mkdir(odir_path)
        else:
            print("Writing output in ", odir_path)

        vtr_out = []
        for ts, vtk in enumerate(self.vtr_pdata):
            vtr_filename = f"step_{ts:05d}_"+ofilename+".vtr"
            vtr_out.append(vtr_filename)
            vtk.write_vtk_to_pvtr(odir_path+"/"+vtr_filename)

        pvd_filename = ofilename+".pvd"
        with open(pvd_filename, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            for t, fn in zip(self.timesteps, vtr_out):
                f.write(f'  <DataSet timestep="{t}" file="{odir}/{fn}"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')


class COMPUTE:

    def __init__(self):
        pass

    def thickness(self, surf_top, surf_bot):
        return surf_top - surf_bot

    def mean_between_surf(self, arr, surf_top, surf_bot):
        z = arr['z']
        top3d = surf_top.broadcast_like(arr)
        bot3d = surf_bot.broadcast_like(arr)
        arr_mean = arr.where((z < top3d) & (z > bot3d)).mean(dim='z')
        return arr_mean
    
    def topo_iso(self, rho_p, rho_m, rho_s, dLAB, thick):
        topo_iso_no_norm = xr.where(
            rho_p <= rho_m, (rho_m-rho_p)*abs(dLAB)/(rho_p-rho_s),
            xr.where(
                rho_p > rho_m, (rho_m-rho_p)*thick/(rho_m-rho_s),
                0,
                    )
                )
        topo_iso = topo_iso_no_norm - topo_iso_no_norm.mean(dim=["x","y"])
        return topo_iso
    
    def compensation_depth(self, dLAB, pvd_ds, max_depth=-115):
        dLAB = dLAB.where(dLAB > max_depth, max_depth)
        z_comp_depth = dLAB.min(dim=['x','y'])
        id_comp_depth = pvd_ds.ds.z.where(pvd_ds.ds.z <= z_comp_depth).argmax(dim='z')
        return id_comp_depth

    def Pdyn_topo(self, rho_m, rho_s, id_comp_depth, pvd_ds):
        # z_comp_depth = dLAB.min(dim=['x','y'])
        # id_comp_depth = pvd_ds.z.where(pvd_ds.z <= z_comp_depth).argmax(dim='z')
        P_dyn_at_comp = pvd_ds.ds["pressure_dyn [MPa]"][:,:,:,id_comp_depth]
        Pdyn_topo = P_dyn_at_comp/((rho_m-rho_s)*9.81)*1e3 
        return Pdyn_topo
    
    def stress_topo(self, rho_m, rho_s, id_comp_depth, pvd_ds):
        stress_at_comp = pvd_ds.ds["dev_stress [MPa][8]"][:,:,:,id_comp_depth]
        stress_topo = stress_at_comp/((rho_m-rho_s)*9.81)*1e3 
        return stress_topo
    
    def topographies(self, rho_m, rho_s, dLAB, surface, pvd_ds, max_depth=-115):
        """
        Computes all topographies from pvd_ds, PVDIO object containing an xarray dataset
        with dynamic pressure and deviatoric stress.

        :rho_m: int, mantle density
        :rho_p: xarray datarray, horizontal mean densities between LAB and surface
        :rho_s: int, density at surface (air, water, sediments...)
        :dLAB: xarray dataarray, surface of the LAB
        :thick: xarray dataarray, thickness between LAB and surface
        :pvd_ds: PVDIO object, containing an xarray dataset with dynamic pressure and deviatoric stress
        :max_depth: int, maximum depth of the compensation depth (used for isostasy)
        """

        dz = pvd_ds.ds.z[-1].values - pvd_ds.ds.z[-2].values
        # Clip at maximum depth
        self.dLAB = dLAB.where(dLAB > max_depth, max_depth)
        # Compute plates thicknesses
        self.thick = self.thickness(surface, self.dLAB)
        # Compute mean densities
        self.rho_p = self.mean_between_surf(pvd_ds.ds["density [kg.m-3]"], surface-dz, self.dLAB+dz)

        topo_iso = self.topo_iso(self.rho_p, rho_m, rho_s, self.dLAB, self.thick)
        id_comp_depth = self.compensation_depth(self.dLAB, pvd_ds, max_depth=max_depth)
        Pdyn_topo = self.Pdyn_topo(rho_m, rho_s, id_comp_depth, pvd_ds)
        stress_topo = self.stress_topo(rho_m, rho_s, id_comp_depth, pvd_ds)
        dynamic_topo = Pdyn_topo + stress_topo
        total_topo = dynamic_topo + topo_iso

        # Compute vertical movements
        iso_uplift = topo_iso.differentiate(coord="t")
        Pdyn_uplift = Pdyn_topo.differentiate(coord="t")
        stress_uplift = stress_topo.differentiate(coord="t")
        dynamic_uplift = dynamic_topo.differentiate(coord="t")
        total_uplift = total_topo.differentiate(coord="t")
        
        topo_iso = topo_iso.transpose("t","x","y")
        ds = topo_iso.to_dataset(name="iso_topo [km]")
        ds["Pdyn_topo [km]"] = Pdyn_topo
        ds["stress_topo [km]"] = stress_topo
        ds["dynamic_topo [km]"] = dynamic_topo
        ds["total_calc_topo [km]"] = total_topo

        ds["iso_uplift [mm.yr-1]"] = iso_uplift
        ds["Pdyn_uplift [mm.yr-1]"] = Pdyn_uplift
        ds["stress_uplift [mm.yr-1]"] = stress_uplift
        ds["dynamic_uplift [mm.yr-1]"] = dynamic_uplift
        ds["total_calc_uplift [mm.yr-1]"] = total_uplift


        return ds

    def cut_ds_at_nearest(self, ds, x=None, y=None, z=None):
        
        # One value
        if isinstance(x, (float,int)):
            # find nearest index per timestep
            idx = np.abs(ds.x - x).argmin(dim="x")
            print("idx", idx)
            # select along x using advanced indexing
            ds = ds.isel(x=idx)

        elif isinstance(y, (float,int)):
            idx = np.abs(ds.y - y).argmin(dim="y")
            ds = ds.isel(y=idx)

        elif isinstance(z, (float,int)):
            idx = np.abs(ds.z - z).argmin(dim="z")
            ds = ds.isel(z=idx)
        
        # Min and max values
        elif isinstance(x, (tuple, list)):
            # find nearest index per timestep
            idx_min = np.abs(ds.x - x[0]).argmin(dim="x")
            print("idx", idx_min)
            idx_max = np.abs(ds.x - x[1]).argmin(dim="x")
            # select along x using advanced indexing
            ds = ds.isel(x=slice(idx_min, idx_max))

        elif isinstance(y, (tuple)):
            idx_min = np.abs(ds.y - y[0]).argmin(dim="y")
            idx_max = np.abs(ds.y - y[1]).argmin(dim="y")
            ds = ds.isel(y=slice(idx_min, idx_max))

        elif isinstance(z, (tuple)):
            idx_min = np.abs(ds.z - z[0]).argmin(dim="z")
            idx_max = np.abs(ds.z - z[0]).argmin(dim="z")
            ds = ds.isel(z=slice(idx_min, idx_max))

        else:
            print("Need to set either x, y or z by an int or float. Can also be : tuple(min,max).")
            print("Nothing has been done.")


        return ds

    # Calculate strain rate from velocity divergence
    def extract_strain_rate(self, ds_surf):
        # Convert cm/yr to km/s
        vel_x = ds_surf["velocity [cm.yr-1][0]"] * 1e-5 / (365*24*60*60)
        vel_y = ds_surf["velocity [cm.yr-1][1]"] * 1e-5 / (365*24*60*60)
        # Surface velocity derivatives
        dvx = vel_x.differentiate(coord="x")
        dvy = vel_y.differentiate(coord="y")
        # Divergence
        strain = dvx + dvy
        # Drop useless z axis
        strain_2d = strain.isel(z=0)
        return strain_2d

    # Detect and label trenches function
    def label_trenches(self, mask_2d: np.ndarray, min_size, CONNECTIVITY,
                        merge_distance) -> np.ndarray:
        struct = ndimage.generate_binary_structure(2, CONNECTIVITY)
        # Merge close trenches if needed
        if merge_distance > 0:
            # Build a circular structuring element of radius = merge_distance
            r = merge_distance
            cy, cx = np.ogrid[-r:r+1, -r:r+1]
            disk = (cx**2 + cy**2 <= r**2)

            # Closing = dilation then erosion:
            #   dilation fills gaps between nearby regions,
            #   erosion restores original blob boundaries (without merging artifacts)
            mask_closed = ndimage.binary_closing(mask_2d, structure=disk)
        else:
            mask_closed = mask_2d

        labeled, n = ndimage.label(mask_closed, structure=struct)

        # Remove small regions
        for region_id in range(1, n + 1):
            if (labeled == region_id).sum() < min_size:
                labeled[labeled == region_id] = 0
        labeled, _ = ndimage.label(labeled > 0, structure=struct)
        return labeled.astype(np.int32)

    # ── HELPER: detect dominant orientation of a labeled blob ────────────────────
    def get_orientation(self, sel):
        """
        Returns 'y' if trench is elongated along the Y axis (x is the thin dim),
                'x' if trench is elongated along the X axis (y is the thin dim).
        sel: 2D boolean array of shape (x, y)
        """
        x_extent = sel.any(axis=1).sum()   # number of distinct x indices occupied
        y_extent = sel.any(axis=0).sum()   # number of distinct y indices occupied
        return "y" if y_extent >= x_extent else "x"

    def extract_trench_lines(self, labels_np, x_vals, y_vals,
                              aggregator, expected_orient, dim2):
        """
        For each labeled trench, return a 1D coordinate array representing
        the trench skeleton line.

        Returns a dict:
        {
            trench_id (int): {
            "orientation": "x" or "y",
            "along":   1D array of the axis the trench runs along,  e.g. y_vals
            "pos":     1D array of median positions on the thin axis, e.g. x position at each y
            }
        }
        """
        n_trenches = labels_np.max()
        result = {}
        if aggregator=="mean":
            aggregator = np.mean
        elif aggregator=="median":
            aggregator = np.median
        else:
            raise ValueError("Wrong value for 'aggregator'. Set to 'mean' or 'median'.")
        
        new_label = 1
        # Select one trench after another
        for tid in range(1, n_trenches + 1):
            sel = labels_np == tid          # (x, y) boolean

            # Get dominant direction of the given trench
            orientation = self.get_orientation(sel)

            # if expected_orient != orientation and dim2==False:
            #     pass
            # else:
            #     continue

            # Might be more understandable and clean to switch to vectorized xarray calcul
            if orientation == "y":
                # trench runs along Y → for each y-slice, find median x
                along_vals = []
                pos_vals   = []
                for j, yv in enumerate(y_vals):
                    row = sel[:, j]         # boolean slice along x at this y
                    if row.any():
                        along_vals.append(yv)
                        pos_vals.append(aggregator(x_vals[row]))
                result[new_label] = {
                    "orientation": "y",
                    "along_name":  "y",
                    "pos_name":    "x",
                    "along":       np.array(along_vals),
                    "pos":         np.array(pos_vals),
                }

            else:
                # trench runs along X → for each x-slice, find median y
                along_vals = []
                pos_vals = []
                print("X_Vals")
                print(x_vals)
                for i, xv in enumerate(x_vals):
                    col = sel[i, :]         # boolean slice along y at this x
                    if col.any():
                        along_vals.append(xv)
                        pos_vals.append(aggregator(y_vals[col]))
                result[new_label] = {
                    "orientation": "x",
                    "along_name":  "x",
                    "pos_name":    "y",
                    "along":       np.array(along_vals),
                    "pos":         np.array(pos_vals),
                }
            new_label += 1

        return result, new_label-1

    def detect_trenches(self, ds_surf, k, min_size, aggregator, merge_distance=0,
                        CONNECTIVITY=2, smooth=True, expected_orient=None, plot=False, dim2=False):

        # Calculate strain rate
        strain_2d = self.extract_strain_rate(ds_surf)

        # Smooth strain field if needed
        if smooth==True:
            sigma = 1.5
            strain_smooth = xr.apply_ufunc(
                lambda arr: ndimage.gaussian_filter(arr, sigma=(0, sigma, sigma)),
                strain_2d,
                dask="parallelized",
            )

        else:
            strain_smooth = strain_2d

        # Configure an adaptative threshold based on strain field standard dev
        thresh = strain_smooth.mean(dim=["x","y"]) - k * strain_2d.std(dim=["x","y"])
        # Extract data only where threshold is met
        mask = strain_smooth < thresh

        # Label all trenches according to the mask
        trench_labels = xr.apply_ufunc(
            self.label_trenches,
            mask, 
            kwargs={
                "min_size": min_size, 
                "merge_distance": merge_distance, 
                "CONNECTIVITY": CONNECTIVITY,
            },
            input_core_dims=[["x", "y"]],   # ← your spatial dims order
            output_core_dims=[["x", "y"]],  # ← same here
            vectorize=True,
            output_dtypes=[np.int32],
        )
        trench_labels.name = "trench_label"
        print("Number of detected trenches along time axis:")
        print(trench_labels.max(dim=["x","y"]).values)

        # Convert trenches to lines (removes thickness due to raw masking)
        x_vals = strain_2d.x.values
        y_vals = strain_2d.y.values

        # Store skeleton lines in a nested dict: trench_lines[t][trench_id]
        trench_lines = {}
        n_saved = []
        times = strain_2d.sizes["t"]
        for ti in range(times):
            t_val  = float(strain_2d.t.isel(t=ti).values)
            lbl_np = trench_labels.isel(t=ti).values    # (x, y)
            lines, n_trench = self.extract_trench_lines(lbl_np, x_vals, y_vals,
                                        aggregator, expected_orient, dim2)
            # trench_lines[t_val] = lines
            trench_lines[ti] = lines
            trench_lines[ti]["model_time"] = t_val
            n_saved.append(n_trench)
        print("Number of saved trenches :")
        print(n_saved)

        if plot==True:
            cmap = cmc.bam
            colors = cmap(np.linspace(0, 1, times))
            for i, t in enumerate(trench_lines.keys()):
                for tid, info in trench_lines[t].items():
                    if isinstance(tid, str):
                        pass
                    elif info["orientation"] == "y":
                        # trench runs along y → plot (pos=x, along=y)
                        plt.plot(info["pos"], info["along"],
                                    color=colors[i], linewidth=1, label=f"trench_{tid}")
                    elif info["orientation"] == "x":
                        # trench runs along x → plot (along=x, pos=y)
                        plt.plot(info["along"], info["pos"],
                                    color=colors[i], linewidth=1, label=f"trench_{tid}")
            plt.xlim(ds_surf.x.min(),ds_surf.x.max())
            plt.ylim(ds_surf.y.min(),ds_surf.y.max())
            plt.show()


        return trench_lines, trench_labels, strain_smooth

        
    def get_trench_y_bounds(self, dict_trenches, trench_id=1):
        """
        Returns the y range that is covered by the trench at ALL timesteps,
        i.e. the intersection of all [min_y, max_y] intervals.
        """
        # Need to take into account if orientation is "y"
        maxes_y = []
        mins_y  = []
        for t, trenches in dict_trenches.items():
            if trench_id in trenches:
                along = trenches[trench_id]["along"]
                mins_y.append(along.min())
                maxes_y.append(along.max())

        trench_min_y = max(mins_y)   # highest of all minimums  → safe lower bound
        trench_max_y = min(maxes_y)  # lowest  of all maximums  → safe upper bound
        return trench_min_y, trench_max_y


    
    def normalize_by_trench(self, ds_surf, dict_trenches, trench_id):
        # Need to take into account if orientation is "y"
        # Extract y_bounds
        dim2=False
        if len(ds_surf.y)==2:
            dim2=True
            ds_surf_slice = ds_surf
        else:    
            trench_min_y, trench_max_y = self.get_trench_y_bounds(dict_trenches, trench_id=trench_id)
            print(f"Stable y range: [{trench_min_y:.4f}, {trench_max_y:.4f}]")
            ds_surf_slice = ds_surf.sel(y=slice(trench_min_y,trench_max_y))

        y_grid = ds_surf_slice.y.values   # 1D, the y coords after cutting
        t_vals = list(dict_trenches.keys())

        x_trench_arr = np.full((len(t_vals)), np.nan) if dim2 else np.full((len(t_vals), len(y_grid)), np.nan)

        valid_t = []
        for ti, t in enumerate(t_vals):
            trenches = dict_trenches[t]
            if trench_id not in trenches:
                continue                        # leave NaN for missing timesteps
            info   = trenches[trench_id]
            along  = info["along"]              # y values of skeleton
            pos    = info["pos"]                # x values of skeleton

            if dim2 is True:
                x_trench_arr[ti] = np.mean(along)
                print("xtrench",x_trench_arr)
            # interpolate skeleton onto the full cut y grid
            else:
                f = interp1d(along, pos, kind="linear",
                            bounds_error=False, fill_value="extrapolate")
                x_trench_arr[ti, :] = f(y_grid)
            valid_t.append(ti)
        print("xtrench",x_trench_arr)

        # Only keep timesteps where trench 1 exists
        # valid_t = list(x_trench_arr.keys())
        print(f"Keeping {len(valid_t)}/{len(t_vals)} timesteps with trench 1")
        try:
            ds_cut = ds_surf_slice.isel(t=valid_t, z=0)
        except:
            ds_cut = ds_surf_slice.isel(t=valid_t)
        if dim2 is True:
            x_trench = xr.DataArray(
                np.array([x_trench_arr[t] for t in valid_t]),  # shape (t,)
                dims=["t"],
                coords={"t": ds_cut.t},
                name="x_trench",
             )
            x_norm = (ds_cut.x - x_trench).transpose("t","x")
            ds_norm = ds_cut.assign_coords(x=x_norm)

        else:
            # Wrap as DataArray with (t, y) dims
            x_trench = xr.DataArray(
                np.array([x_trench_arr[t] for t in valid_t]),
                dims=["t", "y"],
                coords={"t": ds_cut.t, "y": y_grid},
                name="x_trench",
            )
            x_norm = (ds_cut.x - x_trench).transpose("t","x","y")
            ds_norm = ds_cut.assign_coords(x=x_norm)

        return ds_norm


    def normalize_by_trench2(self, ds_surf, dict_trenches, trench_id):
        trench_min_y, trench_max_y = self.get_trench_y_bounds(dict_trenches, trench_id=trench_id)
        print(f"Stable y range: [{trench_min_y:.4f}, {trench_max_y:.4f}]")
        ds_surf_slice = ds_surf.sel(y=slice(trench_min_y, trench_max_y))
        dim2 = len(ds_surf_slice.y) == 2

        y_grid = ds_surf_slice.y.values
        t_vals = list(dict_trenches.keys())
        x_trench_arr = [] if dim2 else np.full((len(t_vals), len(y_grid)), np.nan)

        valid_t = []
        for ti, t in enumerate(t_vals):
            trenches = dict_trenches[t]
            if trench_id not in trenches:
                continue
            info  = trenches[trench_id]
            along = info["along"]
            pos   = info["pos"]

            if dim2:
                x_trench_arr.append(np.mean(along))
            else:
                f = interp1d(along, pos, kind="linear",
                            bounds_error=False, fill_value="extrapolate")
                x_trench_arr[ti, :] = f(y_grid)
            valid_t.append(ti)

        print(f"Keeping {len(valid_t)}/{len(t_vals)} timesteps with trench {trench_id}")
        ds_cut = ds_surf_slice.isel(t=valid_t, z=0)

        x_grid = ds_cut.x.values  # shape (Nx,)
        x_norm_grid = x_grid - x_grid.mean()  # common centered output grid

        ds_list = []
        for i, ti in enumerate(valid_t):
            ds_t = ds_cut.isel(t=i)

            if dim2:
                x_shift = x_trench_arr[i]
            else:
                x_shift = float(np.mean(x_trench_arr[ti]))

            x_shifted = x_grid - x_shift

            # Sort by shifted x to avoid duplicate/unsorted index issues
            sort_idx = np.argsort(x_shifted)
            ds_t = ds_t.isel(x=sort_idx)
            ds_t = ds_t.assign_coords(x=x_shifted[sort_idx])

            # Interpolate onto the common centered grid
            ds_t = ds_t.interp(x=x_norm_grid, kwargs={"fill_value": "extrapolate"})
            ds_list.append(ds_t)

        ds_norm = xr.concat(ds_list, dim="t")
        return ds_norm