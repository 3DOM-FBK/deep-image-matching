import logging
import os
import sqlite3

import networkx as nx
import numpy as np
from pyvis.network import Network

from deep_image_matching.utils.database import pair_id_to_image_ids

logger = logging.getLogger("dim")

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "utils/templates")


def view_graph(db, output_dir, imgs_dir):
    logger.info("Creating view graph visualization...")

    imgs_dir = os.path.abspath(imgs_dir)

    con = sqlite3.connect(db)
    cur = con.cursor()

    # Create network
    nt = Network()

    # HTML template for view graph details panel
    nt.set_template(os.path.join(TEMPLATE_DIR, "template.html"))

    # Add nodes
    G = nx.Graph()
    res = cur.execute("SELECT name, image_id from images")
    for name, id in res.fetchall():
        G.add_node(int(id), label=str(id), shape="circle", title=name)

    # Add edges
    res = cur.execute("SELECT pair_id, rows FROM two_view_geometries")
    weight_sum = 0
    for pair_id, rows in res.fetchall():
        img1, img2 = pair_id_to_image_ids(pair_id)
        img1 = int(img1)
        img2 = int(img2)
        G.add_edge(img1, img2, matches=rows)
        weight_sum += rows

    # avg_weight = weight_sum / len(G.edges())

    # # Load images for small networks
    # if G.number_of_nodes() <= 30:
    #     for n in G.nodes():
    #         G.nodes[n]["shape"] = "image"
    #         G.nodes[n]["label"] = G.nodes[n]["title"]
    #         G.nodes[n]["image"] = os.path.join(imgs_dir, G.nodes[n]["label"])

    # Create list of aligned images and
    # add NA prefix for not aligned images
    aligned_nodes = []
    na_nodes = []
    for n in G.nodes():
        if G.degree[n] == 0:
            G.nodes[n]["aligned"] = 0
            na_nodes.append(n)
        else:
            aligned_nodes.append(n)
            G.nodes[n]["aligned"] = 1

    maxnodes, _, attributes = max(G.edges(data=True), key=lambda edge: edge[2]["matches"])
    max_edge_value = attributes["matches"]

    minnodes, _, attributes = min(G.edges(data=True), key=lambda edge: edge[2]["matches"])
    min_edge_value = attributes["matches"]

    # Scale edge width and assign label to edge popup
    for e in G.edges():
        G.edges[e]["weight"] = (
            np.power(
                ((G.edges[e]["matches"] - min_edge_value) + 1) / (max_edge_value - min_edge_value),
                2,
            )
            * 10
        )
        G.edges[e]["title"] = G.edges[e]["matches"]

    # Compute node positions using the spring layout

    AG = nx.subgraph(G, aligned_nodes)
    pos_aligned = nx.spring_layout(AG, seed=0, weight="matches", iterations=100, scale=800)

    for n, pos in pos_aligned.items():
        G.nodes[n]["x"] = pos[0]
        G.nodes[n]["y"] = -pos[1]

    if len(na_nodes) > 0:
        NAG = nx.subgraph(G, na_nodes)
        pos_na = nx.planar_layout(NAG, scale=100, center=[-800, -800], dim=2)
        for n, pos in pos_na.items():
            G.nodes[n]["x"] = pos[0]
            G.nodes[n]["y"] = -pos[1]

    # Compute communities using modularity
    C = nx.community.greedy_modularity_communities(AG, "matches")
    i = 0
    Cs = []
    for c in C:
        Cg = G.subgraph(c)  # Draw communities with different colors
        for n in Cg.nodes():
            G.nodes[n]["group"] = i
        i += 1
        Cs.append(Cg.number_of_nodes())

    nt.from_nx(G)
    nt.toggle_physics(False)

    # Send additional options to the web page
    nt.set_options(
        """
    var options = {{
     "properties": {{
      "edges": {},
      "aligned": {},
      "not_aligned": {},
      "communities": {}
      }}
    }}
    """.format(G.number_of_edges(), len(aligned_nodes), len(na_nodes), Cs)
    )

    # Write graph.html
    cwd = os.getcwd()
    os.chdir(output_dir)
    out = os.path.join(output_dir, "graph.html")
    nt.write_html("graph.html", notebook=False, open_browser=False)
    logger.info("View graph written at {}".format(out))
    os.chdir(cwd)

    return
