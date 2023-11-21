from database import COLMAPDatabase
from database import blob_to_array
from database import array_to_blob
from database import pair_id_to_image_ids
import numpy as np

db_path2 = r"/media/luca/T7/2022-06-30/alike+KN+sift+glue/joined.db"
db2 = COLMAPDatabase.connect(db_path2)

db_path3 = r"/media/luca/T7/2022-06-30/alike+KN+sift+glue/joined2raw.db"
db3 = COLMAPDatabase.connect(db_path3)
db3.create_tables()


### MAIN ###
# Load cameras in merged database
rows = db2.execute("SELECT * FROM cameras")
for row in rows:
    camera_id, model, width, height, params, prior = row
    params = blob_to_array(params, np.float64)
    camera_id1 = db3.add_camera(model, width, height, params, prior)

# Read existing kpts from database 2
keypoints2 = dict(
    (image_id, blob_to_array(data, np.float32, (-1, 2)))
    for image_id, data in db2.execute(
        "SELECT image_id, data FROM keypoints"))

# Read existing matches from database 2
matches2 = {}
for pair_id, r, c, data in db2.execute("SELECT pair_id, rows, cols, data FROM matches"):
    if data != None:
        pair_id = pair_id_to_image_ids(pair_id)
        matches2[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(data, np.uint32, (-1, 2))

two_views_matches2 = {}
for pair_id, r, c, data in db2.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
    if data != None:
        pair_id = pair_id_to_image_ids(pair_id)
        two_views_matches2[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(data, np.uint32, (-1, 2))

# Store all imgs
img_list = list(keypoints2.keys())

# Add images in merged database
imgs2 = dict(
    (image_id, (name, camera_id))
    for image_id, name, camera_id in db2.execute(
        "SELECT image_id, name, camera_id FROM images"))

for image_id in list(imgs2.keys()):
    db3.add_image(imgs2[image_id][0], imgs2[image_id][1])

# Add kpts
for im_id in img_list:
    keypoints = keypoints2[im_id][:, :2]
    db3.add_keypoints(im_id, keypoints)

# Add raw matches
all_matches = {}
for pair in two_views_matches2:
    all_matches[pair] = two_views_matches2[pair]

for pair in all_matches:
    im1 = int(pair[0])
    im2 = int(pair[1])
    db3.add_matches(im1, im2, all_matches[pair])

db3.commit()
db2.close()
db3.close()