import logging
import os
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Union

import matplotlib
import networkx as nx
import numpy as np
from pyvis.network import Network

from deep_image_matching.utils.database import pair_id_to_image_ids

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

logger = logging.getLogger("dim")

TEMPLATE_DIR = Path(__file__).parent / "utils" / "templates"


def save_output_graph(G: nx.Graph, name: Union[str, Path]) -> None:
    """Save a NetworkX graph as an HTML visualization using pyvis.

    Args:
        G: The NetworkX graph to visualize.
        name: The output filename/path for the HTML file.
    """
    nt = Network()

    # HTML template for view graph details panel
    template_path = TEMPLATE_DIR / "template.html"
    nt.set_template(str(template_path))
    print(f"Using template: {template_path}")
    nt.from_nx(G)
    nt.toggle_physics(False)

    # Get graph attributes with defaults
    edges = G.number_of_edges()
    aligned = G.graph.get("aligned_nodes", 0)
    not_aligned = G.graph.get("na_aligned_nodes", 0)
    communities = G.graph.get("communities", [])

    nt.set_options(
        f"""
    var options = {{
     "properties": {{
      "edges": {edges},
      "aligned": {aligned},
      "not_aligned": {not_aligned},
      "communities": {communities}
      }}
    }}
    """
    )
    html = nt.generate_html(str(name), notebook=False)
    with open(name, mode="w", encoding="utf-8") as fp:
        fp.write(html)


def view_graph(
    db: Union[str, Path], output_dir: Union[str, Path], imgs_dir: Union[str, Path]
) -> None:
    """Create graph visualizations from image matching database.

    This function creates interactive HTML visualizations of the image matching graph,
    including community detection, maximum spanning trees, and outlier identification.

    Args:
        db: Path to the SQLite database containing image matching results.
        output_dir: Directory where output files will be saved.
        imgs_dir: Directory containing the input images.
    """
    logger.info("Creating view graph visualization...")

    # Convert to Path objects
    db_path = Path(db)
    output_path = Path(output_dir)
    imgs_path = Path(imgs_dir).resolve()

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Add nodes
    G = nx.Graph()
    res = cur.execute("SELECT name, image_id from images")
    for name, img_id in res.fetchall():
        G.add_node(int(img_id), label=str(img_id), shape="circle", title=name)

    # Add edges
    res = cur.execute("SELECT pair_id, rows FROM two_view_geometries")
    for pair_id, rows in res.fetchall():
        img1, img2 = pair_id_to_image_ids(pair_id)
        img1 = int(img1)
        img2 = int(img2)
        G.add_edge(img1, img2, matches=rows)

    # Handle case where there are no edges
    if G.number_of_edges() == 0:
        logger.warning(
            "No edges found in the graph. Cannot proceed with visualization."
        )
        return

    # Create list of aligned images and add NA prefix for not aligned images
    aligned_nodes = []
    na_nodes = []
    for n in G.nodes():
        if G.degree[n] == 0:
            G.nodes[n]["aligned"] = 0
            na_nodes.append(n)
        else:
            aligned_nodes.append(n)
            G.nodes[n]["aligned"] = 1

    G.graph["aligned_nodes"] = len(aligned_nodes)
    G.graph["na_aligned_nodes"] = len(na_nodes)

    # Find min and max edge values for normalization
    edge_data = [data["matches"] for _, _, data in G.edges(data=True)]
    max_edge_value = max(edge_data)
    min_edge_value = min(edge_data)

    # Scale edge width and assign label to edge popup
    for edge in G.edges():
        matches = G.edges[edge]["matches"]
        G.edges[edge]["weight"] = (
            np.power(
                ((matches - min_edge_value) + 1) / (max_edge_value - min_edge_value),
                2,
            )
            * 10
        )
        G.edges[edge]["title"] = matches

    # Compute node positions using the spring layout
    if aligned_nodes:
        AG = G.subgraph(aligned_nodes)
        pos_aligned = nx.spring_layout(
            AG, seed=0, weight="matches", iterations=50, scale=800
        )

        for n, pos in pos_aligned.items():
            G.nodes[n]["x"] = pos[0]
            G.nodes[n]["y"] = -pos[1]

        # Compute communities using modularity
        C = nx.community.greedy_modularity_communities(AG, "matches", resolution=1)

        # Compute clustering coefficient for each node
        clustering = nx.clustering(AG, weight="matches")

        # Compute maximum spanning tree
        MST = nx.maximum_spanning_tree(AG, "matches")
        MST_raw = nx.maximum_spanning_tree(AG, "matches")
    else:
        C = []
        clustering = {}
        MST = nx.Graph()
        MST_raw = nx.Graph()

    if na_nodes:
        NAG = G.subgraph(na_nodes)
        pos_na = nx.planar_layout(NAG, scale=100, center=[-800, -800], dim=2)
        for n, pos in pos_na.items():
            G.nodes[n]["x"] = pos[0]
            G.nodes[n]["y"] = -pos[1]

    # Remove output files if they exist
    files_to_remove = [
        output_path / "communities.txt",
        output_path / "raw_mst_pairs.txt",
        output_path / "exp_mst_pairs.txt",
    ]
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()

    # Write communities output file
    communities_file = output_path / "communities.csv"
    with open(communities_file, "w", encoding="utf-8") as comm_file:
        print(
            "IMG_ID,IMG_NAME,Community_ID,Clustering_coefficient(0,1),IS_OUTLIER?[0,1]",
            file=comm_file,
        )

        i = 0
        Cs = []
        for c in C:
            Cg = G.subgraph(c)
            comm_clustering = [clustering[n] for n in Cg.nodes if n in clustering]

            if comm_clustering:
                avg_comm_clustering = mean(comm_clustering)
                threshold = 0.3
                MST.add_edges_from(Cg.edges(data=True))

                for n in Cg.nodes():
                    # Draw communities with different colors
                    G.nodes[n]["group"] = i
                    MST.nodes[n]["group"] = i
                    MST_raw.nodes[n]["group"] = i

                    # Draw probable outliers with larger shape
                    node_clustering = clustering.get(n, 0)
                    if node_clustering < threshold * avg_comm_clustering:
                        G.nodes[n]["font"] = {"size": 12}
                        G.nodes[n]["opacity"] = 1
                        is_outlier = 1
                    else:
                        is_outlier = 0

                    out = f"{n},{G.nodes[n]['title']},{i},{node_clustering:.4f},{is_outlier}"
                    print(out, file=comm_file)

            i += 1
            Cs.append(Cg.number_of_nodes())

        # MST expansion - Find edges between communities (after all communities are processed)
        for e in MST.edges():
            if (
                "group" in G.nodes[e[0]]
                and "group" in G.nodes[e[1]]
                and G.nodes[e[0]]["group"] != G.nodes[e[1]]["group"]
            ):
                inter_community_edges = [
                    (u, v, d)
                    for u, v, d in G.edges(data=True)
                    if (
                        "group" in G.nodes[u]
                        and "group" in G.nodes[v]
                        and G.nodes[u]["group"] == G.nodes[e[0]]["group"]
                        and G.nodes[v]["group"] == G.nodes[e[1]]["group"]
                    )
                ]
                MST.add_edges_from(inter_community_edges)

    G.graph["communities"] = Cs

    # Copy essential graph attributes to MST graphs
    MST.graph["aligned_nodes"] = G.graph["aligned_nodes"]
    MST.graph["na_aligned_nodes"] = G.graph["na_aligned_nodes"]
    MST.graph["communities"] = G.graph["communities"]

    MST_raw.graph["aligned_nodes"] = G.graph["aligned_nodes"]
    MST_raw.graph["na_aligned_nodes"] = G.graph["na_aligned_nodes"]
    MST_raw.graph["communities"] = G.graph["communities"]

    # Write MST and MST_expanded pairs output files
    raw_mst_file = output_path / "raw_mst_pairs.txt"
    with open(raw_mst_file, "w", encoding="utf-8") as f:
        for e in MST_raw.edges():
            node1_title = G.nodes[e[0]]["title"]
            node2_title = G.nodes[e[1]]["title"]
            print(f"{node1_title} {node2_title}", file=f)

    exp_mst_file = output_path / "exp_mst_pairs.txt"
    with open(exp_mst_file, "w", encoding="utf-8") as f:
        for e in MST.edges():
            node1_title = G.nodes[e[0]]["title"]
            node2_title = G.nodes[e[1]]["title"]
            print(f"{node1_title} {node2_title}", file=f)

    # Save graph visualizations
    original_cwd = os.getcwd()
    try:
        os.chdir(output_path)
        save_output_graph(G, "graph.html")
        save_output_graph(MST, "exp_mst.html")
        save_output_graph(MST_raw, "raw_mst.html")
    finally:
        os.chdir(original_cwd)

    logger.info(f"View graphs written at {output_path}")
    con.close()
