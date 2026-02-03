"""export h5 features and matches to Bundler format"""

import argparse
import logging
import sys
from io import StringIO
from pathlib import Path
from itertools import permutations
from typing import Any, Optional

import h5py
import numpy as np
import yaml
from tqdm import tqdm

# dependencies check
try:
    import pandas as pd
    from scipy.spatial import cKDTree
except ImportError:
    print("Error: This exporter requires 'pandas' and 'scipy'. "
          "Please install them via pip: pip install pandas scipy")
    sys.exit(1)

from ..utils.image import read_image, IMAGE_EXT

logger = logging.getLogger("dim")

def export_to_bundler(image_dir: Path, feat_h5: Path,
                      match_h5: Path, output_pth: Path,
                      px_tolerance: int = 1,
                      use_descriptors: bool = False,
                      descriptor_threshold: float = 0.8,
                      max_merge_iters: int = 10,
                      s_min: int = 1, s_max: int = 50,
                      camera_config_path: Optional[Path] = None) -> None:
    """
    Exports H5 features and matches to a Bundler .out file
    """

    image_dir = Path(image_dir)
    feature_path = Path(feat_h5)
    match_path = Path(match_h5)
    output_pth = Path(output_pth)

    # check inputs
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file {feature_path} does not exist")
    if not match_path.exists():
        raise FileNotFoundError(f"Matches file {match_path} does not exist")

    # create output directory if needed
    out_dir = output_pth.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # Extract tie points, confidence, AND descriptors from h5 files
    image_ids, tp_dict, conf_dict, desc_dict = _extract_dicts_from_h5(image_dir,
                                                                      feature_path,
                                                                      match_path,
                                                                      use_descriptors)

    # Find all images on disk and get their paths and shapes
    logger.info(f"Reading image dimensions from {image_dir}...")
    all_images = [p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXT]
    path_map = {p.name: p for p in all_images}

    # store the images and their shapes
    dict_images = {}
    dict_image_shapes = {}

    # Get image IDs from H5 file keys
    h5_image_ids = set(image_ids)

    for img_name in tqdm(h5_image_ids, desc="Loading image shapes"):
        if img_name in path_map:
            img_path = path_map[img_name]
            try:
                img = read_image(img_path)
                # Get (Width, Height)
                dict_images[img_name] = img
                dict_image_shapes[img_name] = (img.shape[1], img.shape[0])
            except Exception as e:
                logger.warning(f"Could not read image {img_name}: {e}")
        else:
            logger.warning(f"Image {img_name} from H5 not found in {image_dir}")

    # Parse Camera Config
    camera_data = None
    if camera_config_path and camera_config_path.exists():
        logger.info(f"Loading camera configuration from {camera_config_path}")
        with open(camera_config_path) as f:
            cam_config = yaml.safe_load(f)

        # Generate the list of camera parameters aligned with h5_image_ids
        camera_data = _parse_camera_config(h5_image_ids, dict_image_shapes, cam_config)
    elif camera_config_path:
        logger.warning(f"Camera config path provided but does not exist: {camera_config_path}")

    # Filter all dicts to only include images found on disk
    valid_tp_dict = {}
    valid_conf_dict = {}
    valid_desc_dict = {}
    for (i0, i1), tps in tp_dict.items():
        if i0 in dict_images and i1 in dict_images:
            valid_tp_dict[(i0, i1)] = tps
            valid_conf_dict[(i0, i1)] = conf_dict[(i0, i1)]
            if desc_dict:
                valid_desc_dict[(i0, i1)] = desc_dict[(i0, i1)]

    logger.info(f"Filtered {len(tp_dict)} pairs down to {len(valid_tp_dict)} valid pairs "
                f"with images on disk.")

    if not valid_tp_dict:
        logger.error("No valid tie points found. Aborting.")
        return

    # Handle empty descriptor dict if none were found
    if not valid_desc_dict:
        valid_desc_dict = None

    # Create bundler DataFrame
    bundler_df = _build_bundler(
        tp_dict=valid_tp_dict,
        conf_dict=valid_conf_dict,
        dict_images=dict_images,
        dict_image_shapes=dict_image_shapes,
        desc_dict=valid_desc_dict,
        px_tolerance=px_tolerance,
        use_descriptors=use_descriptors,
        descriptor_threshold=descriptor_threshold,
        s_min=s_min,
        s_max=s_max,
        max_merge_iters=max_merge_iters,
        drop_orphans=True
    )

    # Export bundler file using the new, optimized exporter
    total_cameras = len(dict_image_shapes)
    _export_bundler(bundler_df, camera_data=camera_data,
                    bundler_pth=output_pth,
                    total_cameras=total_cameras)


def _parse_camera_config(image_ids: list[str],
                         image_shapes: dict[str, tuple[int, int]],
                         config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parses the YAML camera config and maps parameters to the sorted list of images.
    Returns a list of camera dicts in the exact order of image_ids.
    """
    camera_data_list = []

    # Map image_name -> config_entry
    img_to_cam_opts = {}

    general_opts = config.get("general", {})

    # Iterate over 'cam0', 'cam1', etc.
    for key, opts in config.items():
        if key == "general":
            continue

        # Check if 'images' key exists and parse patterns
        if "images" in opts:
            patterns = opts["images"].split(",")
            for pattern in patterns:
                pattern = pattern.strip()
                if not pattern:
                    continue
                if pattern in image_ids:
                    img_to_cam_opts[pattern] = opts

    # Build the final list
    for img_name in image_ids:
        width, height = image_shapes.get(img_name, (0, 0))
        max_size = max(width, height)

        # Defaults
        focal = 1.2 * max_size if max_size > 0 else 1.0
        k1 = 0.0
        k2 = 0.0

        # Try specific camera assignment
        opts = img_to_cam_opts.get(img_name)

        # If no specific assignment, check if "single_camera" is enforced in general
        if opts is None and general_opts.get("single_camera") is True:
            pass

        # Extract intrinsics if available: [f, cx, cy, k, ...]
        if opts and "intrinsics" in opts and opts["intrinsics"]:
            intr = opts["intrinsics"]
            # Handle different lengths based on model
            if len(intr) >= 1:
                focal = float(intr[0])

            if len(intr) > 3:
                k1 = float(intr[3])
            if len(intr) > 4:
                k2 = float(intr[4])

        camera_data_list.append({
            "focal_length": focal,
            "k1": k1,
            "k2": k2,
            "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation_vector": [0.0, 0.0, 0.0]
        })

    return camera_data_list


def _extract_dicts_from_h5(image_dir: Path,
                           feature_path: Path,
                           match_path: Path,
                           use_descriptors: bool) -> tuple:
    """
    Extracts tie points (tp_dict), confidences (conf_dict),
    descriptors (desc_dict), and image IDs from H5 files.
    """
    tp_dict, conf_dict = {}, {}
    desc_dict = {} if use_descriptors else None
    descriptors_found = False

    # open with h5py to read features and matches
    with h5py.File(feature_path, "r") as f, h5py.File(match_path, "r") as m:
        image_ids = list(f.keys())

        # order images by filename
        image_ids.sort(key=lambda x: str(image_dir / x).lower())

        for i0, i1 in permutations(image_ids, 2):
            if i0 not in m or i1 not in m[i0]:
                # no matches between these images
                continue

            # Keypoints
            kpts0 = np.array(f[i0]["keypoints"])  # (N0, 2)
            kpts1 = np.array(f[i1]["keypoints"])  # (N1, 2)

            # Descriptors
            if use_descriptors:
                try:
                    descr0 = np.array(f[i0]["descriptors"]) # (N0, D)
                    descr1 = np.array(f[i1]["descriptors"]) # (N1, D)
                    descriptors_found = True
                except KeyError:
                    if descriptors_found: # Warn if descriptors are inconsistent
                        logger.warning(f"Missing descriptors for {i0} or {i1}, but not for others.")
                    descriptors_found = False
                    descr0, descr1 = None, None

            # Matches
            group = m[i0][i1]
            if "matches0" not in group:
                continue

            matches0 = np.array(group["matches0"])  # length N0, -1 if no match
            mask = matches0 >= 0
            if not np.any(mask):
                continue

            idx0 = np.where(mask)[0]
            idx1 = matches0[mask]

            # Tie Points
            x0y0 = kpts0[idx0, :2]
            x1y1 = kpts1[idx1, :2]
            tp_dict[(i0, i1)] = np.hstack([x0y0, x1y1])  # (N, 4)

            # Confidence
            if "scores0" in group:
                conf = np.array(group["scores0"])[mask]
                # normalize to [0,1]
                if conf.max() > 1.0:
                    conf = conf / conf.max()
            else:
                conf = np.ones(len(idx0), dtype=np.float32)
            conf_dict[(i0, i1)] = conf  # (N,)

            # Matched Descriptors
            if use_descriptors:
                if descriptors_found and descr0 is not None and descr1 is not None:
                    d0 = descr0[idx0] # (N, D)
                    d1 = descr1[idx1] # (N, D)
                    desc_dict[(i0, i1)] = np.hstack([d0, d1]) # (N, D*2)

    if use_descriptors and not descriptors_found:
        logger.warning("No descriptors found in H5 file. Merging by proximity only.")
        desc_dict = None

    return image_ids, tp_dict, conf_dict, desc_dict

def _build_bundler(tp_dict: dict,
                  conf_dict: dict,
                  dict_images: dict,
                  dict_image_shapes: dict,
                  desc_dict: dict | None = None,
                  px_tolerance: float = 10.0,
                  use_descriptors: bool = False,
                  descriptor_threshold: float = 0.80,
                  s_min: int = 1,
                  s_max: int = 50,
                  max_merge_iters: int = 10,
                  drop_orphans: bool = True) -> pd.DataFrame:
    """
    Build Bundler structure using descriptor similarity for track merging.
    """

    if use_descriptors and desc_dict is None:
        raise ValueError("Descriptor dictionary is None, but use_descriptors is True.")

    if use_descriptors:
        logger.info("Building Bundler structure with descriptor-based merging...")
        logger.info(f"Parameters: px_tolerance={px_tolerance}, "
                    f"descriptor_threshold={descriptor_threshold}")
    else:
        logger.info("Building Bundler structure (proximity-only merging)...")
        logger.info(f"Parameters: px_tolerance={px_tolerance}")

    # Create initial arrays from pairwise matches
    x_arr, y_arr, img_idx_arr, conf_arr, track_arr, color_arr, desc_arr, image_ids = \
        _create_arrays(tp_dict, conf_dict, desc_dict, dict_images,
                       s_min, s_max)

    logger.info(f"Created {len(x_arr)} initial observations from "
                f"{len(tp_dict)} image pairs")

    # Build DataFrame
    df_data = {
        'x': x_arr,
        'y': y_arr,
        'image_idx': img_idx_arr.astype(np.int32),
        'confidence': conf_arr,
        'track_idx': track_arr.astype(np.int64),
        'color': color_arr,
    }

    if desc_arr is not None:
        df_data['descriptor'] = list(desc_arr)  # Store as list of arrays

    # convert to dataframe
    df = pd.DataFrame(df_data)

    # Validate: each raw track must appear exactly twice (pairwise matches)
    raw_counts = df['track_idx'].value_counts()
    if not (raw_counts == 2).all():
        bad = raw_counts[raw_counts != 2]
        raise ValueError(f"Pre-merge invariant failed: some track_ids don't "
                         f"appear exactly twice: {bad.to_dict()}")

    # Iterative merging
    df = _merge_tracks(df, dict_image_shapes, image_ids,
                                        px_tolerance,
                                        use_descriptors,
                                        descriptor_threshold,
                                        max_merge_iters)

    # Final validation and cleanup
    df = _finalize_bundler(df, image_ids, dict_image_shapes,
                           drop_orphans)


    num_tracks = df['track_idx'].nunique()
    num_images = len(dict_image_shapes.keys())
    num_images_with_tracks = df['image_idx'].nunique()
    logger.info(f"Final bundler: {len(df)} observations in "
                f"{num_tracks} tracks across "
                f"{num_images_with_tracks} images")
    if num_images != num_images_with_tracks:
        logger.warning(f"{num_images - num_images_with_tracks} images have no tracks")

    return df


def _create_arrays(tp_dict: dict,
                   conf_dict: dict,
                   desc_dict: dict | None,
                   dict_images: dict,
                   s_min: int,
                   s_max: int) -> tuple:
    """
    Create arrays from dictionaries including descriptors.

    Returns:
        x_arr, y_arr, img_idx_arr, conf_arr, track_arr, color_arr, desc_arr, image_ids
    """

    logger.debug("Creating arrays from dictionaries")
    use_descriptors = desc_dict is not None

    # get image ids and mapping to index
    image_ids = list(dict_images.keys())
    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    # Pre-calculate total size to avoid list extensions
    total_observations = sum(tps.shape[0] * 2 for tps in tp_dict.values())

    # Pre-allocate arrays
    x_arr = np.empty(total_observations, dtype=np.float32)
    y_arr = np.empty(total_observations, dtype=np.float32)
    img_idx_arr = np.empty(total_observations, dtype=np.int32)
    conf_arr = np.empty(total_observations, dtype=np.float32)
    track_arr = np.empty(total_observations, dtype=np.int64)
    color_arr = np.empty(total_observations, dtype=np.uint8)

    # Get descriptor dimension from first entry
    desc_arr = None
    desc_dim = 0
    if use_descriptors:
        # Get descriptor dimension from first entry
        first_desc = next(iter(desc_dict.values()))
        desc_dim = first_desc.shape[1] // 2  # Half for each image
        desc_arr = np.empty((total_observations, desc_dim), dtype=np.float32)

    # keep track of counters
    current_idx = 0
    track_counter = 0

    # iterate over dicts
    for key in tqdm(tp_dict.keys(), desc="Creating arrays", unit="pair"):
        # get id of img1 and img2
        img1_id, img2_id = key

        # get tps, conf, and descriptors
        tps = tp_dict[key]
        conf = conf_dict[key]
        desc1, desc2 = None, None
        if use_descriptors:
            desc = desc_dict[key]
            # Split descriptors
            desc1 = desc[:, :desc_dim]
            desc2 = desc[:, desc_dim:]
        # get points from both images
        pts1 = tps[:, :2]
        pts2 = tps[:, 2:]

        # Load pixel colors
        colors1 = np.array(_load_pixel_values(dict_images[img1_id],
                                            pts1[:, 0].astype(int).tolist(),
                                            pts1[:, 1].astype(int).tolist()), dtype=np.uint8)
        colors2 = np.array(_load_pixel_values(dict_images[img2_id],
                                            pts2[:, 0].astype(int).tolist(),
                                            pts2[:, 1].astype(int).tolist()), dtype=np.uint8)

        # Compute confidence sizes
        conf_clipped = np.clip(conf, 0, 1)
        sizes = s_min + (s_max - s_min) * (1 - conf_clipped)

        # get number of points
        n = pts1.shape[0]

        # Fill arrays for image 1
        end_idx = current_idx + n
        x_arr[current_idx:end_idx] = pts1[:, 0]
        y_arr[current_idx:end_idx] = pts1[:, 1]
        img_idx_arr[current_idx:end_idx] = id_to_idx[img1_id]
        conf_arr[current_idx:end_idx] = sizes
        track_arr[current_idx:end_idx] = np.arange(track_counter, track_counter + n)
        color_arr[current_idx:end_idx] = colors1
        if use_descriptors:
            desc_arr[current_idx:end_idx] = desc1
        current_idx = end_idx

        # Fill arrays for image 2
        end_idx = current_idx + n
        x_arr[current_idx:end_idx] = pts2[:, 0]
        y_arr[current_idx:end_idx] = pts2[:, 1]
        img_idx_arr[current_idx:end_idx] = id_to_idx[img2_id]
        conf_arr[current_idx:end_idx] = sizes
        track_arr[current_idx:end_idx] = np.arange(track_counter, track_counter + n)
        color_arr[current_idx:end_idx] = colors2
        if use_descriptors:
            desc_arr[current_idx:end_idx] = desc2
        current_idx = end_idx

        # Increment track counter
        track_counter += n

    return (x_arr, y_arr, img_idx_arr, conf_arr,
            track_arr, color_arr, desc_arr, image_ids)


def _load_pixel_values(img: np.ndarray,
                       x: int | list[int],
                       y: int | list[int]
                       ) -> float | list[float] | list[list[float]]:
    """
    Load pixel value(s) from a NumPy image array.
    Uses vectorized NumPy indexing for speed.
    """

    # Ensure x and y are arrays so we can use "fancy indexing"
    # np.atleast_1d handles both single integers and lists automatically
    x_idx = np.atleast_1d(x).astype(int)
    y_idx = np.atleast_1d(y).astype(int)

    # Access the data directly (Vectorized)
    try:
        samples = img[y_idx, x_idx]
    except IndexError:
        raise ValueError(f"Coordinates out of image bounds. Image shape: {img.shape}")

    # Return as standard Python lists/floats
    result = samples.astype(float).tolist()

    # If the original input was just a single scalar (not a list),
    # you might want to return the raw value instead of a list wrapping it.
    if isinstance(x, int) and isinstance(y, int):
        return result[0] if isinstance(result, list) else result

    return result


def _merge_tracks(df: pd.DataFrame,
                 dict_image_shapes: dict,
                 image_ids: list,
                 px_tolerance: float,
                 use_descriptors: bool,
                 descriptor_threshold: float,
                 max_merge_iters: int) -> pd.DataFrame:
    """
    Iteratively merge tracks within each image using descriptor similarity.
    """

    def _norm_tol(width: int, height: int, px_tol: float) -> float:
        """Normalize pixel tolerance by image dimensions"""
        return px_tol / float(max(width, height))

    # Check if descriptors are available to use
    if use_descriptors:
        logger.info(f"Starting iterative track merging (max {max_merge_iters} iterations) with descriptors")
    else:
        logger.info(f"Starting iterative track merging (max {max_merge_iters} iterations) by proximity only")

    desc_full = None
    if use_descriptors:
        # Convert descriptor column to numpy array for easier manipulation
        desc_full = np.vstack(df['descriptor'].values)

    for it in range(max_merge_iters):
        mappings_accum = []
        merge_count = 0

        # Process each image separately
        for image_idx in tqdm(df['image_idx'].unique(),
                              desc=f"Merge iteration {it + 1}",
                              unit="image"):

            mask = df['image_idx'] == image_idx
            idx_in_df = df.index[mask].to_numpy()

            if len(idx_in_df) < 2:
                continue

            # Get image dimensions
            w, h = dict_image_shapes[image_ids[image_idx]]

            # Get subset data
            coords = df.loc[idx_in_df, ['x', 'y']].to_numpy()
            coords_n = coords / float(max(w, h))
            tol_n = _norm_tol(w, h, px_tolerance)

            # Find spatially close candidates
            tree = cKDTree(coords_n)
            pairs = tree.query_pairs(r=tol_n, output_type="ndarray")

            if pairs.size == 0:
                continue

            valid_pairs = []
            desc_subset = None
            if use_descriptors:
                # Get descriptors for this subset
                desc_subset = desc_full[idx_in_df]

                # Filter pairs by descriptor similarity
                for i, j in pairs:
                    similarity = _cosine_similarity(desc_subset[i], desc_subset[j])
                    if similarity >= descriptor_threshold:
                        valid_pairs.append([i, j])
            else:
                # No descriptors: all spatially close pairs are valid
                valid_pairs = pairs.tolist()

            if len(valid_pairs) == 0:
                continue

            valid_pairs = np.array(valid_pairs)

            # Get current values
            x_vals = df.loc[idx_in_df, 'x'].to_numpy()
            y_vals = df.loc[idx_in_df, 'y'].to_numpy()
            color_vals = df.loc[idx_in_df, 'color'].to_numpy()
            track_vals = df.loc[idx_in_df, 'track_idx'].to_numpy()

            # Merge points and descriptors
            x_new, y_new, color_new, track_new, desc_new, mapping = \
                _merge_points_with_descriptors(
                    valid_pairs, x_vals, y_vals, color_vals,
                    track_vals, desc_subset
                )

            # Update DataFrame
            df.loc[idx_in_df, 'x'] = x_new
            df.loc[idx_in_df, 'y'] = y_new
            df.loc[idx_in_df, 'color'] = color_new
            df.loc[idx_in_df, 'track_idx'] = track_new

            if use_descriptors:
                # Update descriptors in full array
                desc_full[idx_in_df] = desc_new

            if mapping.shape[0] > 0:
                mappings_accum.extend(mapping.tolist())
                merge_count += mapping.shape[0]

        if not mappings_accum:
            logger.info(f"No new merges in iteration {it + 1}; stopping.")
            break

        logger.info(f"Iteration {it + 1}: merged {merge_count} track pairs")

        # Resolve transitive mappings globally
        mapping_dict = _resolve_mapping_chains(mappings_accum)
        mapped = df['track_idx'].map(mapping_dict)
        df['track_idx'] = np.where(mapped.notna(), mapped,
                                   df['track_idx']).astype(np.int64)

        # Drop duplicate observations of same (image, track)
        before_len = len(df)
        df = df.drop_duplicates(subset=['image_idx', 'track_idx']).copy()

        # Also filter descriptor array to match
        if len(df) < before_len and use_descriptors:
            # We need to re-align desc_full with the new df
            desc_full = np.vstack(df['descriptor'].values)

    # Reconstruct descriptor column at the end
    if use_descriptors:
        df['descriptor'] = list(desc_full)

    return df


def _merge_points_with_descriptors(pairs_array: np.ndarray,
                                  x_arr: np.ndarray,
                                  y_arr: np.ndarray,
                                  color_arr: np.ndarray,
                                  track_idx_arr: np.ndarray,
                                  desc_arr: np.ndarray | None) -> tuple:
    """
    Merge points and their descriptors.
    Returns updated arrays and mapping information.
    """
    use_descriptors = desc_arr is not None

    unique_tracks = np.unique(track_idx_arr)
    max_track_id = int(np.max(unique_tracks)) + 1
    processed_tracks = np.zeros(max_track_id, dtype=bool)

    max_mappings_list = []
    min_mappings_list = []

    for pair_idx in range(pairs_array.shape[0]):
        i = pairs_array[pair_idx, 0]
        j = pairs_array[pair_idx, 1]

        track_idx_i = int(track_idx_arr[i])
        track_idx_j = int(track_idx_arr[j])

        # Skip if already processed
        if processed_tracks[track_idx_i] or processed_tracks[track_idx_j]:
            continue

        min_track_idx = min(track_idx_i, track_idx_j)
        max_track_idx = max(track_idx_i, track_idx_j)

        # Average coordinates and color
        avg_x = (x_arr[i] + x_arr[j]) / 2.0
        avg_y = (y_arr[i] + y_arr[j]) / 2.0
        c_i = color_arr[i].astype(np.uint16)
        c_j = color_arr[j].astype(np.uint16)
        avg_color = ((c_i + c_j) // 2).astype(color_arr.dtype)

        # Update both points
        x_arr[i] = x_arr[j] = avg_x
        y_arr[i] = y_arr[j] = avg_y
        color_arr[i] = color_arr[j] = avg_color
        track_idx_arr[i] = track_idx_arr[j] = min_track_idx

        if use_descriptors:
            # Average and normalize descriptors
            avg_desc = (desc_arr[i] + desc_arr[j]) / 2.0
            norm = np.linalg.norm(avg_desc)
            if norm > 1e-8:
                avg_desc = avg_desc / norm
            else:
                avg_desc = np.zeros_like(avg_desc) # Handle zero-norm case

            # Update descriptors
            desc_arr[i] = desc_arr[j] = avg_desc

        # Record mapping
        max_mappings_list.append(max_track_idx)
        min_mappings_list.append(min_track_idx)

        # Mark as processed
        processed_tracks[track_idx_i] = True
        processed_tracks[track_idx_j] = True

    # Create mapping array
    if len(max_mappings_list) > 0:
        mapping_data = np.column_stack([
            np.array(max_mappings_list, dtype=np.int64),
            np.array(min_mappings_list, dtype=np.int64)
        ])
    else:
        mapping_data = np.empty((0, 2), dtype=np.int64)

    return x_arr, y_arr, color_arr, track_idx_arr, desc_arr, mapping_data


def _finalize_bundler(df: pd.DataFrame,
                      image_ids: list,
                      dict_image_shapes: dict,
                      drop_orphans: bool) -> pd.DataFrame:
    """
    Final validation and formatting of bundler structure.
    """

    # Check each track appears in >= 2 images
    track_img_counts = df.groupby('track_idx')['image_idx'].nunique()

    if drop_orphans:
        orphans = track_img_counts[track_img_counts < 2].index
        if len(orphans) > 0:
            logger.warning(f"Dropping {len(orphans)} orphan tracks (<2 images)")
            df = df[~df['track_idx'].isin(orphans)].copy()
    else:
        if (track_img_counts < 2).any():
            raise ValueError("Post-merge invariant failed: some tracks "
                             "appear in < 2 images")

    # Reindex tracks sequentially
    unique_tracks = sorted(df['track_idx'].unique())
    trk_remap = {old: new for new, old in enumerate(unique_tracks)}
    df['track_idx'] = df['track_idx'].map(trk_remap).astype(np.int64)

    # Per-image feature indices (required by Bundler)
    df = df.sort_values(['image_idx', 'track_idx']).reset_index(drop=True)
    df['feature_idx'] = (
        df.groupby('image_idx')
        .cumcount()
        .astype(np.int64)
    )

    # Add image dimensions and Bundler-centered coordinates
    df['image_dims_x'] = df['image_idx'].map(
        lambda i: dict_image_shapes[image_ids[i]][0])
    df['image_dims_y'] = df['image_idx'].map(
        lambda i: dict_image_shapes[image_ids[i]][1])
    df['bundler_x'] = df['x'] - df['image_dims_x'] / 2.0
    df['bundler_y'] = df['image_dims_y'] / 2.0 - df['y']

    # Drop descriptor column (if it exists)
    if 'descriptor' in df.columns:
        df = df.drop(columns=['descriptor'])

    return df

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two descriptor vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return dot_product / (norm1 * norm2)


def _resolve_mapping_chains(mappings: list) -> dict:
    """
    Resolve transitive mappings using union-find with path compression.
    """
    if not mappings:
        return {}

    # Build parent dict
    parent = {}
    for old, new in mappings:
        parent[int(old)] = int(new)

    # Find root with path compression
    def find_root(x):
        if x not in parent:
            return x
        root = x
        while root in parent:
            root = parent[root]
        # Path compression
        current = x
        while current != root:
            next_node = parent[current]
            parent[current] = root
            current = next_node
        return root

    # Resolve all mappings to their roots
    resolved = {}
    for old in parent.keys():
        resolved[old] = find_root(old)

    return resolved

def _export_bundler(df: pd.DataFrame,
                    camera_data,
                    bundler_pth: Path,
                    total_cameras: int) -> None:
    """
    Exports the track DataFrame to a 'bundler.out' file.

    This function is highly optimized to write the bundler file quickly
    by vectorizing string operations with pandas and writing to an
    in-memory buffer.

    Args:
        df: The DataFrame from build_bundler.
        camera_data: A list of dictionaries, one per camera, containing
            `focal_length`, `k1`, `k2`, `rotation_matrix`, and
            `translation_vector`. If None, dummy data is used.
        bundler_pth: Path to save 'bundler.out'.
        total_cameras: Total number of cameras in the dataset.
    """

    if df.empty:
        logger.warning(f"Bundler DataFrame is empty. Skipping export to {bundler_pth}")
        return

    logger.info(f"Exporting {len(df)} observations to {bundler_pth}")

    # Ensure numeric types ONCE before grouping
    obs_cols = ["image_idx", "feature_idx", "bundler_x", "bundler_y"]
    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, obs_cols] = df[obs_cols].apply(pd.to_numeric, errors="raise")

    # Group observations by track
    df_grouped = df.groupby("track_idx")
    num_tracks = len(df_grouped)

    # Fill in dummy camera data if missing
    num_cameras = total_cameras

    if camera_data is None or len(camera_data) != num_cameras:
        if camera_data is not None:
            logger.warning(f"Camera data length mismatch ({len(camera_data)} != {num_cameras})."
                           " Reverting to dummy data.")

        camera_data_list = [{
            "focal_length": 1.0,
            "k1": 0.0, "k2": 0.0,
            "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation_vector": [0.0, 0.0, 0.0]
        } for _ in range(num_cameras)]
    else:
        logger.info("Using provided a priori camera data for bundler export.")
        camera_data_list = camera_data

    logger.info(f"Writing {num_cameras} cameras and {num_tracks} tracks to {bundler_pth}...")

    # Writing Bundler File
    with open(bundler_pth, "w") as f:
        # Header block
        f.write("# Bundle file v0.3\n")
        f.write(f"{num_cameras} {num_tracks}\n")

        # Camera block
        for cam in camera_data_list:
            f.write(f'{cam["focal_length"]} {cam["k1"]} {cam["k2"]}\n')
            f.write(" ".join(map(str, cam["rotation_matrix"][0:3])) + "\n")
            f.write(" ".join(map(str, cam["rotation_matrix"][3:6])) + "\n")
            f.write(" ".join(map(str, cam["rotation_matrix"][6:9])) + "\n")
            f.write(" ".join(map(str, cam["translation_vector"])) + "\n")

        # Optimized Points Block
        logger.info("Optimizing track export...")

        # Pre-calculate track stats (color and count)
        pbar = tqdm(total=4, desc="Vectorizing bundler export")
        pbar.set_postfix_str("Aggregating stats...")
        track_stats = df_grouped.agg(
            avg_color=("color", "mean"),
            count=("image_idx", "size")  # 'size' is fast
        )
        track_stats["avg_color"] = track_stats["avg_color"].round().astype(int)
        pbar.update(1)

        # Vectorize observation string formatting
        pbar.set_postfix_str("Formatting obs strings...")
        df_str = pd.DataFrame({
            "track_idx": df["track_idx"],
            "obs_part": (
                    df["image_idx"].astype(int).astype(str) + " " +
                    df["feature_idx"].astype(int).astype(str) + " " +
                    df["bundler_x"].astype(str) + " " +
                    df["bundler_y"].astype(str)
            )
        })
        pbar.update(1)

        # Aggregate strings by track
        pbar.set_postfix_str("Grouping obs strings...")
        track_obs_lines = df_str.groupby("track_idx")["obs_part"].apply(" ".join)
        pbar.update(1)

        # Combine stats and observation strings
        pbar.set_postfix_str("Building final summary...")
        summary_df = track_stats.join(track_obs_lines.rename("obs_line"))
        pbar.update(1)
        pbar.close()
        logger.info("Vectorization complete. Writing to file...")

        # Write to buffer
        buffer = StringIO()
        lines = [
            f"0.0 0.0 0.0\n{row.avg_color} {row.avg_color} {row.avg_color}\n{row.count} {row.obs_line}\n"  # noqa
            for row in tqdm(summary_df.itertuples(), total=len(summary_df), desc="Write bundler file")
        ]
        buffer.write("".join(lines))

        # Write buffer to file
        f.write(buffer.getvalue())

    logger.info("Bundler file written successfully.")


def main():
    parser = argparse.ArgumentParser(description="Export to Bundler from H5")
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        required=True,
        help="Path to the image directory.",
    )
    parser.add_argument(
        "-f",
        "--features_h5",
        type=str,
        required=True,
        help="Path to the features.h5 file.",
    )
    parser.add_argument(
        "-m",
        "--matches_h5",
        type=str,
        required=True,
        help="Path to the matches.h5 file.",
    )
    parser.add_argument(
        "-o",
        "--out_pth",
        type=str,
        required=True,
        help="Output path for the Bundler file.",
    )
    parser.add_argument(
        "-x",
        "--img_ext",
        type=str,
        default=IMAGE_EXT,
        help="Image extension."
    )
    parser.add_argument(
        "-px",
        "--pixel_tolerance",
        type=int,
        required=False,
        default=1,
        help="Pixel tolerance for merging points.",
    )
    parser.add_argument(
        "-d",
        '--use_descriptors',
        type=bool,
        required=False,
        default=False,
        help="Whether to use descriptors for merging tracks.",
    )
    parser.add_argument(
        "-dt",
        '--descriptor_threshold',
        type=float,
        required=False,
        default=0.80,
        help="Cosine similarity threshold for merging tracks based on descriptors.",
    )
    parser.add_argument(
        "-mmi",
        '--max_merge_iters',
        type=int,
        required=False,
        default=10,
        help="Maximum number of iterations for merging tracks.",
    )
    parser.add_argument(
        "-smin",
        "--size_min",
        type=int,
        required=False,
        default=1,
        help="Minimum marker size.",
    )
    parser.add_argument(
        "-smax",
        "--size_max",
        type=int,
        required=False,
        default=50,
        help="Maximum marker size.",
    )
    parser.add_argument(
        "-cc",
        "--camera_config_path",
        type=str,
        required=False,
        default=None,
        help="Path to the camera configuration YAML file.",
    )

    args = parser.parse_args()

    # Convert to Path objects
    image_dir = Path(args.image_dir)
    features_h5 = Path(args.features_h5)
    matches_h5 = Path(args.matches_h5)
    out_pth = Path(args.out_pth)

    # get the other args
    px_tolerance = args.pixel_tolerance
    use_descriptors = args.use_descriptors
    descriptor_threshold = args.descriptor_threshold
    max_merge_iters = args.max_merge_iters
    s_min = args.size_min
    s_max = args.size_max
    camera_config_path = args.camera_config_path

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    images = sorted([e for e in image_dir.glob("*") if e.suffix in args.img_ext])
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if not features_h5.exists():
        raise FileNotFoundError(f"Feature file {features_h5} does not exist")
    if not matches_h5.exists():
        raise FileNotFoundError(f"Matches file {matches_h5} does not exist")

    # Configure basic logging for command-line use
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Call the export
    export_to_bundler(image_dir, features_h5, matches_h5, out_pth,
                      px_tolerance,
                      use_descriptors, descriptor_threshold,
                      max_merge_iters,
                      s_min, s_max,
                      camera_config_path)


if __name__ == "__main__":
    # This block is for testing, updated to match the new structure

    # Configure basic logging for testing
    logging.basicConfig(level=logging.INFO)

    project_path = Path("your/test/project/path") # CHANGE THIS
    img_dir = project_path / "images"
    features_h5 = project_path / "features.h5"
    matches_h5 = project_path / "matches.h5"
    out_pth = project_path / "bundler.out"

    # Example call
    try:
        export_to_bundler(img_dir, features_h5, matches_h5, out_pth,
                          px_tolerance=1, s_min=1, s_max=50)
        print("Done.")
    except FileNotFoundError as e:
        print(f"Test run failed. {e}. Please update the test paths in __main__.")
    except Exception as e:
        print(f"An error occurred: {e}")