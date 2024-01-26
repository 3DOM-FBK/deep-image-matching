import sqlite3
import os
import networkx as nx
from pyvis.network import Network
from deep_image_matching.utils.database import pair_id_to_image_ids
from . import logger


def view_graph(db, output_dir, imgs_dir):
    logger.info("Creating view graph visualization...")

    imgs_dir = os.path.abspath(imgs_dir)

    con = sqlite3.connect(db)
    cur = con.cursor()

    # Create network
    nt = Network(height="50vw")

    # Add nodes
    G = nx.Graph()
    res = cur.execute("SELECT name, image_id from images")
    for name, id in res.fetchall():
        G.add_node(int(id), label=name, shape="ellipse")

    # Add edges
    res = cur.execute("SELECT pair_id, rows FROM matches")
    weight_sum = 0
    for pair_id, rows in res.fetchall():
        if rows != 0:
            img1, img2 = pair_id_to_image_ids(pair_id)
            img1 = int(img1)
            img2 = int(img2)
            # if img1 not in G:
            #     res_images = cur.execute(
            #         "SELECT name from images WHERE image_id = ?", [img1]
            #     )
            #     name = res_images.fetchone()[0]
            #     G.add_node(img1, label=name, shape="ellipse")
            # if img2 not in G:
            #     res_images = cur.execute(
            #         "SELECT name from images WHERE image_id = ?", [img2]
            #     )
            #     name = res_images.fetchone()[0]
            #     G.add_node(img2, label=name, shape="ellipse")
            G.add_edge(img1, img2, matches=rows)
            weight_sum += rows

    avg_weight = weight_sum / len(G.edges())

    # Load images for small networks
    if G.number_of_nodes() <= 30:
        for n in G.nodes():
            G.nodes[n]["shape"] = "image"
            G.nodes[n]["image"] = os.path.join(imgs_dir, G.nodes[n]["label"])

    # Create list of aligned images and
    # add NA prefix for not aligned images
    aligned_nodes = []
    na_nodes = []
    for n in G.nodes():
        if G.degree[n] == 0:
            G.nodes[n]["label"] = "[NA]_" + G.nodes[n]["label"]
            na_nodes.append(n)
        else:
            aligned_nodes.append(n)

    for e in G.edges():
        G.edges[e]["weight"] = G.edges[e]["matches"] / avg_weight
        G.edges[e]["title"] = G.edges[e]["matches"]

    # Compute node positions using the spring layout

    pos_aligned = nx.spring_layout(
        aligned_nodes, seed=0, weight="matches", iterations=100, scale=800
    )
    for n, pos in pos_aligned.items():
        G.nodes[n]["x"] = pos[0]
        G.nodes[n]["y"] = -pos[1]

    if len(na_nodes) > 0:
        pos_na = nx.planar_layout(na_nodes, scale=100, center=[-800, -800], dim=2)
        for n, pos in pos_na.items():
            G.nodes[n]["x"] = pos[0]
            G.nodes[n]["y"] = -pos[1]

    # Compute communities using modularity
    C = nx.community.greedy_modularity_communities(G, "matches")
    i = 0
    for c in C:
        Cg = G.subgraph(c)  # Draw communities with different colors
        for n in Cg.nodes():
            G.nodes[n]["group"] = i
        i += 1

    nt.from_nx(G)
    nt.toggle_physics(False)

    # Write graph.html
    cwd = os.getcwd()
    os.chdir(output_dir)
    out = os.path.join(output_dir, "graph.html")
    nt.write_html("graph.html", notebook=False, open_browser=False)
    logger.info("View graph written at {}".format(out))
    os.chdir(cwd)

    return
