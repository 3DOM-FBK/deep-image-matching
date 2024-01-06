"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime
import logging
from pathlib import Path
from typing import List, Union
from xml.etree import ElementTree as ET

import Metashape
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

from icepy4d.core.calibration import read_opencv_calibration
from icepy4d.core.constants import DATETIME_FMT
from icepy4d.utils.logger import deprecated
from icepy4d.utils.timer import AverageTimer

from .metashape_core import (
    add_markers,
    cameras_from_bundler,
    create_new_project,
    read_gcp_file,
    save_project,
)

REGION_RESIZE_FCT = 10.0


def build_metashape_cfg(cfg: edict, timestamp: Union[str, datetime.datetime]) -> edict:
    """
    # build_metashape_cfg Build metashape configuration dictionary, starting from global configuration dictionary.

    Args:
        cfg (edict): configuration dictionary for icepy4d
        timestamp (Union[str, datetime.datetime]): timestamp of the current processing epoch

    Returns:
        edict: Metashape configuration dictionary
    """
    ms_cfg = edict()

    # To be sure that the timestamp is in the correct format
    if isinstance(timestamp, str):
        timestamp = timestamp.replace(" ", "_")
    elif isinstance(timestamp, datetime.datetime):
        timestamp = timestamp.strftime(DATETIME_FMT)

    # Paths
    ms_cfg.dir = cfg.paths.results_dir / f"{timestamp}/metashape"
    ms_cfg.project_path = ms_cfg.dir / f"{timestamp}.psx"
    ms_cfg.bundler_file_path = ms_cfg.dir / f"data/{timestamp}.out"
    ms_cfg.bundler_im_list = ms_cfg.dir / "data/im_list.txt"
    ms_cfg.gcp_filename = ms_cfg.dir / "data/gcps.txt"
    ms_cfg.calib_filenames = cfg.metashape.calib_filenames
    ms_cfg.dense_path = cfg.paths.results_dir / "point_clouds"
    ms_cfg.dense_name = f"dense_{timestamp}.ply"
    ms_cfg.mesh_name = f"mesh_{timestamp}.ply"

    # Processing parameters
    ms_cfg.optimize_cameras = cfg.metashape.optimize_cameras
    ms_cfg.build_dense = cfg.metashape.build_dense
    ms_cfg.build_mesh = cfg.metashape.build_mesh

    # Camera location
    ms_cfg.camera_location = cfg.metashape.camera_location

    # A-priori observation accuracy
    ms_cfg.cam_accuracy = cfg.metashape.camera_accuracy
    ms_cfg.gcp_accuracy = cfg.metashape.gcp_accuracy
    ms_cfg.collimation_accuracy = cfg.metashape.collimation_accuracy

    # Interior orientation parameters
    ms_cfg.prm_to_fix = cfg.metashape.camera_prm_to_fix

    # Dense matching and mesh
    ms_cfg.dense_downscale_image = cfg.metashape.dense_downscale_factor
    ms_cfg.depth_filter = cfg.metashape.depth_filter

    # Other parameters
    ms_cfg.use_opk = cfg.metashape.use_omega_phi_kappa
    ms_cfg.force_overwrite_projects = cfg.metashape.force_overwrite_projects

    return ms_cfg


class MetashapeProject:
    def __init__(
        self,
        image_list: List[Path],
        cfg: edict,
        timer: AverageTimer = None,
    ) -> None:
        self.image_list = image_list
        self.cfg = cfg
        self.timer = timer

    @property
    def project_path(self) -> str:
        return str(self.cfg.project_path)

    def create_project(self) -> None:
        self.doc = create_new_project(self.project_path)
        if self.cfg.force_overwrite_projects:
            self.doc.read_only = False
        if self.cfg.use_opk:
            self.doc.chunk.euler_angles = Metashape.EulerAnglesOPK
        logging.info(f"Created project {self.project_path}.")

    def add_images(
        self, image_list: List[Path], load_camera_reference: bool = True
    ) -> None:
        images = [str(x) for x in image_list if x.is_file()]
        self.doc.chunk.addPhotos(images)

        # Add cameras and tie points from bundler output
        cameras_from_bundler(
            chunk=self.doc.chunk,
            fname=self.cfg.bundler_file_path,
            image_list=self.cfg.bundler_im_list,
        )

        # Fix Camera location to reference
        if len(self.cfg.cam_accuracy) == 1:
            accuracy = Metashape.Vector(
                (self.cfg.cam_accuracy, self.cfg.cam_accuracy, self.cfg.cam_accuracy)
            )
        elif len(self.cfg.cam_accuracy) == 3:
            accuracy = Metashape.Vector(self.cfg.cam_accuracy)
        else:
            raise ValueError(
                "Wrong input type for accuracy parameter. Provide a list of floats (it can be a list of a single element or of three elements)."
            )
        if load_camera_reference:
            for i, camera in enumerate(self.doc.chunk.cameras):
                camera.reference.location = Metashape.Vector(
                    self.cfg.camera_location[i]
                )
                camera.reference.accuracy = accuracy
                camera.reference.enabled = True

    def import_sensor_calibration(self) -> None:
        for i, sensor in enumerate(self.doc.chunk.sensors):
            cal = Metashape.Calibration()
            cal.load(str(self.cfg.calib_filenames[i]))
            sensor.user_calib = cal
            sensor.fixed_calibration = True
            # if self.cfg.prm_to_fix:
            sensor.fixed_params = self.cfg.prm_to_fix

            logging.info(f"sensor {sensor} loaded.")

    def add_gcps(self) -> None:
        gcps = read_gcp_file(self.cfg.gcp_filename)
        gcp_accuracy = self.cfg.gcp_accuracy
        for point in gcps:
            add_markers(
                self.doc.chunk,
                point["world"],
                point["projections"],
                point["label"],
                bool(point["enabled"]),
                gcp_accuracy,
            )

    def set_a_priori_accuracy(self):
        if (
            "collimation_accuracy" in self.cfg.keys()
            and self.cfg.collimation_accuracy is not None
        ):
            self.doc.chunk.marker_projection_accuracy = self.cfg.collimation_accuracy

    def solve_bundle(self) -> None:
        self.doc.chunk.optimizeCameras(fit_f=True, tiepoint_covariance=True)

    def build_dense_cloud(
        self, save_cloud: bool = True, depth_filter: str = "ModerateFiltering"
    ) -> None:
        """
        build_dense_cloud

        Args:
            save_cloud (bool, optional): Save point cloud to disk. Defaults to True.
            depth_filter (str, optional): Depth filtering mode in [NoFiltering, MildFiltering, ModerateFiltering, AggressiveFiltering]. Defaults to "moderate".
        """

        if depth_filter == "NoFiltering":
            filter = Metashape.FilterMode.NoFiltering
        elif depth_filter == "MildFiltering":
            filter = Metashape.FilterMode.MildFiltering
        elif depth_filter == "ModerateFiltering":
            filter = Metashape.FilterMode.ModerateFiltering
        elif depth_filter == "AggressiveFiltering":
            filter = Metashape.FilterMode.AggressiveFiltering
        else:
            raise ValueError(
                "Error: invalid choise of depth filtering. Choose one in [NoFiltering, MildFiltering, ModerateFiltering, AggressiveFiltering]"
            )

        self.doc.chunk.buildDepthMaps(
            downscale=self.cfg.dense_downscale_image,
            filter_mode=filter,
            reuse_depth=False,
            max_neighbors=16,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
        )
        self.doc.chunk.buildDenseCloud(
            point_colors=True,
            point_confidence=True,
            keep_depth=True,
            max_neighbors=2,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
        )
        if save_cloud:
            self.doc.chunk.exportPoints(
                path=str(self.cfg.dense_path / self.cfg.dense_name),
                source_data=Metashape.DataSource.DenseCloudData,
            )

    def build_mesh(
        self,
        save_mesh: bool = True,
    ) -> None:
        self.doc.chunk.buildModel(
            surface_type=Metashape.SurfaceType.Arbitrary,
            interpolation=Metashape.Interpolation.EnabledInterpolation,
            face_count=Metashape.FaceCount.HighFaceCount,
            source_data=Metashape.DataSource.DepthMapsData,
            vertex_colors=True,
            vertex_confidence=True,
        )
        if save_mesh:
            self.doc.chunk.exportModel(
                path=str(self.cfg.dense_path / self.cfg.mesh_name),
                save_confidence=True,
            )

    def save_project(self) -> None:
        save_project(self.doc, self.project_path)
        self.doc.read_only = False

    def export_camera_estimated(self, path: str = None):
        if path is None:
            path = self.cfg.project_path.parent / (
                self.cfg.project_path.stem + "_camera_estimated.txt"
            )
        self.doc.chunk.exportReference(
            path=str(path),
            format=Metashape.ReferenceFormatCSV,
            items=Metashape.ReferenceItemsCameras,
            delimiter=",",
        )

    def export_camera_extrinsics(self, dir: Union[str, Path] = None):
        if dir is None:
            dir = self.cfg.project_path.parent / "camera_estimated_extrinsics"
        else:
            dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)

        T = self.doc.chunk.transform.matrix

        for camera in self.doc.chunk.cameras:
            # Camera Pose in chunk reference system
            pose = camera.transform

            # Camera Pose in world reference system
            pose_w = T * pose
            Cw = pose_w.translation()

            # Camera extrinsics (R,t) in world reference system
            Rw = pose_w.rotation().inv()
            A = Metashape.Matrix.Rotation(Rw) * Metashape.Matrix.Translation(Cw)
            tw = -A.translation()

            fname = dir / f"{camera.label}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                for x in range(3):
                    f.write(f"{Rw.row(x)[0]} {Rw.row(x)[1]} {Rw.row(x)[2]} {tw[x]}\n")
                f.write(f"{0.0} {0.0} {0.0} {1.0}\n")
                f.close()

    def export_sensor_parameters(
        self, export_dir: str = None, format: str = "opencv", save_as_txt: bool = True
    ):
        if export_dir is None:
            export_dir = self.cfg.project_path.parent
        if format == "opencv":
            cal_fmt = Metashape.CalibrationFormat.CalibrationFormatOpenCV

        # Export calibrated sensor parameters as xml file
        sensors = self.doc.chunk.sensors
        for id, sensor in enumerate(sensors):
            sensor.calibration.save(
                str(export_dir / f"sensor_{id}_calib.xml"),
                format=cal_fmt,
            )

        # Write legend file
        with open(str(export_dir / "sensors_legend.txt"), "w") as f:
            f.write("id, label\n")
            for id, sensor in enumerate(sensors):
                f.write(f"{id}, {sensor.label}\n")
            f.close()

        # Write camera parameters as normal text file
        if save_as_txt:
            for id, sensor in enumerate(sensors):
                in_path = str(export_dir / f"sensor_{id}_calib.xml")
                out_path = str(export_dir / f"sensor_{id}_calib.txt")
                tree = ET.parse(in_path)
                root = tree.getroot()
                w = root[1].text
                h = root[2].text
                K = root[3][3].text.split()
                d = root[4][3].text.split()
                with open(out_path, "w") as f:
                    f.write(f"{w} {h} ")
                    for prm in K:
                        f.write(f"{prm} ")
                    for prm in d:
                        f.write(f"{prm} ")
                    f.close()

    def expand_region(self, resize_fct: float) -> None:
        self.doc.chunk.resetRegion()
        self.doc.chunk.region.size = resize_fct * self.doc.chunk.region.size

    def run_full_workflow(self) -> bool:
        self.create_project()
        self.add_images(self.image_list)
        self.add_gcps()
        self.import_sensor_calibration()
        self.set_a_priori_accuracy()
        self.solve_bundle()
        if self.timer:
            self.timer.update("bundle")
        if self.cfg.build_dense:
            self.expand_region(resize_fct=REGION_RESIZE_FCT)
            if self.cfg.depth_filter:
                self.build_dense_cloud(depth_filter=self.cfg.depth_filter)
            else:
                self.build_dense_cloud()
            if self.timer:
                self.timer.update("dense")
        if self.cfg.build_mesh:
            self.build_mesh()
            if self.timer:
                self.timer.update("mesh")
        self.export_camera_extrinsics()
        self.export_sensor_parameters()
        self.save_project()

        return True


class MetashapeWriter:
    def __init__(self, chunk: Metashape.Chunk):
        pass


class MetashapeReader:
    def __init__(
        self,
        metashape_dir: Union[str, Path],
        num_cams: int,
    ) -> None:
        self.folder_path = metashape_dir
        self.num_cams = num_cams
        self.K = {}  # dict with 3x3 intrinsics matrix for each camera
        self.dist = {}  # dict with distortion parameters vector for each camera
        self.extrinsics = {}  # dict with 4x4 homogeneous matrix for each image

    @deprecated
    # NOTE: this function is not working properly. Need to be fixed
    def get_K(self, camera_id: int = None) -> Union[np.ndarray, dict]:
        if camera_id:
            return self.K[camera_id]
        else:
            return self.K

    def get_dist(self, camera_id: int = None) -> Union[np.ndarray, dict]:
        if camera_id:
            return self.dist[camera_id]
        else:
            return self.dist

    def get_extrinsics(self, camera_id: int = None) -> Union[np.ndarray, dict]:
        if camera_id:
            return self.extrinsics[camera_id]
        else:
            return self.extrinsics

    def get_focal_lengths(self) -> dict:
        return {cam_id: K[1, 1] for cam_id, K in self.K.items()}

    def read_calibration_from_file(
        self,
        filename_list: List[Union[str, Path]] = None,
    ) -> None:
        """
        read_calibration_from_file Read Calibration txt file formatted as described in read_opencv_calibration function (lib.import_export.importing)

        Args:
            filename_list (List[Union[str, Path]], optional): path to the file to read. Defaults to None.
        """
        for i in range(self.num_cams):
            if filename_list is None:
                path = Path(self.folder_path / f"sensor_{i}_calib.txt")
            else:
                path = filename_list[i]
            h, w, K, dist = read_opencv_calibration(path)
            self.K[i] = K
            self.dist[i] = dist

    def read_cameras_from_file(
        self,
        path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        read_cameras_from_file read Metashape default export for Camera Reference

        Args:
            path (Union[str, Path]): path of the file to read

        Returns:
            pd.DataFrame: Dataframe with the camera information
        """
        df = pd.read_csv(path, sep=",", header=1)
        df.drop(df.tail(1).index, inplace=True)

        return df

        # for i in range(self.num_cams):
        #     C = np.asarray([df.X_est[i], df.Y_est[i], df.Z_est[i]]).reshape(3, 1)
        #     rot = np.radians(
        #         np.asarray([df.Omega_est[i], df.Phi_est[i], df.Kappa_est[i]])
        #     )
        #     R = euler_matrix(rot[0], rot[1], rot[2])
        #     R = R[:3, :3]
        #     t = -R @ C

        #     extrinsics = np.identity(4)
        #     extrinsics[:3, :3] = R
        #     extrinsics[0:3, 3:4] = t

        #     self.extrinsics[i] = extrinsics

    def read_cameras_extrinsics(
        self,
        dir: Union[str, Path] = None,
        file_ext: str = "txt",
    ) -> dict:
        """
        read_cameras_extrinsics

        Args:
            path (Union[str, Path]): Path to the folder containing the files with the extrinsics matrix
            Each file must be a .txt file containing with float numbers delimited by spaces. The extrinsics matrix must be written as a 4x4 matrix as follows:
                -0.174 0.984 0.003 -242.747
                0.0175 0.006 -0.999 127.660
                -0.984 -0.174 -0.0183 359.552
                0.0 0.0 0.0 1.0

        Returns:
            dict: dictionary with image names (labels without extension) as keys and extrinsics matrix as value.
        """
        if dir is None:
            dir = self.folder_path / "camera_estimated_extrinsics"
        else:
            dir = Path(dir)
        assert dir.is_dir, f"Directory {dir} does not exist. Provide a valid diretory"

        self.extrinsics = {}
        for file in dir.glob(f"*.{file_ext}"):
            extrinsics = np.loadtxt(file)
            self.extrinsics[file.stem] = extrinsics

    def read_icepy4d_outputs(self) -> None:
        self.read_calibration_from_file()
        self.read_cameras_extrinsics()


if __name__ == "__main__":
    from src.icepy4d.visualization.visualization import make_focal_length_variation_plot

    root_path = Path().absolute()

    # Process data
    # epoches_2_process = range(1)
    # for epoch in epoches_2_process:
    #     cfg = build_ms_cfg_base(root_path, epoch)
    #     cfg.build_dense = False
    #     timer = AverageTimer()

    #     print("Processing started:")
    #     print("-----------------------")

    #     ms = MetashapeProject(cfg, timer)
    #     ms.process_full_workflow()

    #     timer.print(f"Epoch {epoch} completed")

    # Make plot of estimated focal length
    # @TODO: move to visualization.py
    focals = {0: [], 1: []}
    num_cams = 2
    epoches_2_process = range(27)
    for epoch in epoches_2_process:
        epoch_path = root_path / f"res/epoch_{epoch}/metashape"
        ms_reader = MetashapeReader()
        ms_reader.read_calibration_from_file(epoch_path, num_cams=num_cams)
        for i in range(num_cams):
            focals[i].append(ms_reader.get_focal_lengths()[i])

    make_focal_length_variation_plot(focals)

    epoch = 0
    epoch_path = root_path / f"res/epoch_{epoch}/metashape"
    path = epoch_path / "icepy_epoch_0_camera_estimated.txt"
    ms_reader.read_cameras_from_file(path, num_cams)

    # Old code for focals plot
    #     for id in range(2):
    #         path = str(cfg.project_path.parent / f"sensor_{id}_calib.txt")
    #         with open(path, "r") as f:
    #             line = f.readline(-1)
    #             f = float(line.split()[2])
    #             focals[id].append(f)

    # fig, ax = plt.subplots(1, 2)
    # for s_id in range(2):
    #     ax[s_id].plot(epoches_2_process, focals[s_id], "o")
    #     ax[s_id].grid(visible=True, which="both")
    #     ax[s_id].set_xlabel("Epoch")
    #     ax[s_id].set_ylabel("Focal lenght [px]")
    # plt.show()

    print("Done.")
