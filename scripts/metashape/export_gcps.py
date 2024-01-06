from collections import namedtuple

import Metashape

marker = namedtuple("gcp", ["name", "camera", "x", "y"])


def export_markers(
    path: str,
    chunk: Metashape.Chunk = None,
    sort_by_camera=True,
):
    if chunk is None:
        chunk = Metashape.app.document.chunk

    # writing header
    file = open(path, "wt")

    if sort_by_camera:
        file.write("image_name,gcp_name,image_u,image_v\n")
    else:
        file.write("gcp_name,image_name,image_u,image_v\n")

    markers_list = []
    for m in chunk.markers:  # processing every marker in chunk
        projections = m.projections  # list of marker projections
        marker_name = m.label

        for camera in m.projections.keys():  #
            # Get x and y coordinates of the current projection in pixels
            x, y = projections[camera].coord
            cam_label = camera.label

            # Create a marker object and append it to the list
            markers_list.append(marker(marker_name, cam_label, x, y))

    # Sort the list of markers by camera label
    if sort_by_camera:
        markers_list.sort(key=lambda x: x.camera)

    # writing output
    if sort_by_camera:
        for m in markers_list:
            file.write(f"{m.camera},{m.name},{m.x:.4f},{m.y:.4f}\n")

    else:
        for m in markers_list:
            file.write(f"{m.name},{m.camera},{m.x:.4f},{m.y:.4f}\n")

    file.close()


if __name__ == "__main__":
    path = Metashape.app.getSaveFileName("Specify export file name:", filter=" *.txt")
    chunk = Metashape.app.document.chunks
    export_markers(path, sort_by_camera=True)

    # doc_path = "/home/francesco/casalbagliano/subset_A/metashape/subset_full.psx"
    # path = "/home/francesco/casalbagliano/subset_A/metashape/subset_full_markers_by_marker.txt"

    # doc = Metashape.Document()
    # doc.open(doc_path)
    # chunk = doc.chunks[0]

    # export_markers(path, chunk, sort_by_camera=True)
