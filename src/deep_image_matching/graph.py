import logging
import os
import sqlite3
from statistics import mean

import networkx as nx
import numpy as np
from pyvis.network import Network

from deep_image_matching.utils.database import pair_id_to_image_ids

logger = logging.getLogger("dim")

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "utils","templates"
)

def save_output_graph(G, name):
    nt = Network()

    # HTML template for view graph details panel
    nt.set_template(os.path.join(TEMPLATE_DIR, "template.html").replace('\\', '/'))
    nt.from_nx(G)
    nt.toggle_physics(False)

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
    """.format(
            G.number_of_edges(),
            G.graph["aligned_nodes"],
            G.graph["na_aligned_nodes"],
            G.graph["communities"],
        )
    )
    html = nt.generate_html(name, notebook=False)
    with open(name, mode='w', encoding='utf-8') as fp:
        fp.write(html)
    #nt.write_html(name, notebook=False, open_browser=False)

    return


def view_graph(db, output_dir, imgs_dir):
    logger.info("Creating view graph visualization...")

    imgs_dir = os.path.abspath(imgs_dir)

    con = sqlite3.connect(db)
    cur = con.cursor()

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

    """
        Remove output files if they exist
    """
    try:
        os.remove(os.path.join(output_dir, "communities.txt"))
    except OSError:
        pass

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

    G.graph["aligned_nodes"] = len(aligned_nodes)
    G.graph["na_aligned_nodes"] = len(na_nodes)

    _, _, attributes = max(G.edges(data=True), key=lambda edge: edge[2]["matches"])
    max_edge_value = attributes["matches"]

    _, _, attributes = min(G.edges(data=True), key=lambda edge: edge[2]["matches"])
    min_edge_value = attributes["matches"]

    # Scale edge width and assign label to edge popup
    for e in G.edges():
        G.edges[e]["weight"] = (
            np.power(
                ((G.edges[e]["matches"] - min_edge_value) + 1)
                / (max_edge_value - min_edge_value),
                2,
            )
            * 10
        )
        G.edges[e]["title"] = G.edges[e]["matches"]

    # Compute node positions using the spring layout

    AG = nx.subgraph(G, aligned_nodes)
    pos_aligned = nx.spring_layout(
        AG, seed=0, weight="matches", iterations=50, scale=800
    )

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
    # Compute clustering coefficient for each node
    clustering = nx.clustering(AG, weight="matches")

    # Write communities output file
    with open(os.path.join(output_dir, "communities.csv"), "a") as comm_file:
        print(
            "IMG_ID,IMG_NAME,Community_ID,Clustering_coefficient(0,1),IS_OUTLIER?[0,1]",
            file=comm_file,
        )
    for c in C:
        Cg = G.subgraph(c)  # Draw communities with different colors
        comm_clustering = [clustering[n] for n in Cg.nodes]
        avg_comm_clustering = mean(comm_clustering)
        threshold = 0.3
        for n in Cg.nodes():
            # Draw communities with different colors
            G.nodes[n]["group"] = i
            # Draw probable outliers with larger shape
            if clustering[n] < threshold * avg_comm_clustering:
                G.nodes[n]["font"] = {"size": 12}
                G.nodes[n]["opacity"] = 1
                out = "{},{},{},{:.4f},{}".format(
                    n, G.nodes[n]["title"], i, clustering[n], 1
                )
            else:
                out = "{},{},{},{:.4f},{}".format(
                    n, G.nodes[n]["title"], i, clustering[n], 0
                )
            with open(comm_file.name, "a") as comm_file:
                print(out, file=comm_file)
        i += 1
        Cs.append(Cg.number_of_nodes())
    G.graph["communities"] = Cs

    cwd = os.getcwd()
    os.chdir(output_dir)
    save_output_graph(G, "graph.html")
    logger.info(
        "View graph written at {}".format(os.path.join(output_dir, "graph.html"))
    )
    os.chdir(cwd)

    return
