import sqlite3
import os
import networkx as nx
from pyvis.network import Network
from deep_image_matching.utils.database import pair_id_to_image_ids


def view_graph(db, output_dir, imgs_dir):
    print("Creating view graph visualization")
    con = sqlite3.connect(db)
    cur = con.cursor()

    nt = Network(height="50vw")
    # Load matches table as graph
    res = cur.execute("SELECT pair_id, rows FROM matches")
    G = nx.Graph()
    weight_sum = 0
    for pair_id, rows in res.fetchall():
        if rows != 0:
            img1, img2 = pair_id_to_image_ids(pair_id)
            img1 = int(img1)
            img2 = int(img2)
            if img1 not in G:
                res_images = cur.execute(
                    "SELECT name from images WHERE image_id = ?", [img1]
                )
                name = res_images.fetchone()[0]
                G.add_node(img1, label=name, shape="ellipse")
            if img2 not in G:
                res_images = cur.execute(
                    "SELECT name from images WHERE image_id = ?", [img2]
                )
                name = res_images.fetchone()[0]
                G.add_node(img2, label=name, shape="ellipse")
            G.add_edge(img1, img2, matches=rows)
            weight_sum += rows

    avg_weight = weight_sum / len(G.edges())

    # Load images for small networks
    if G.number_of_nodes() <= 30:
        for n in G.nodes():
            G.nodes[n]["shape"] = "image"
            G.nodes[n]["image"] = os.path.join(imgs_dir, G.nodes[n]["label"])

    for e in G.edges():
        G.edges[e]["weight"] = G.edges[e]["matches"] / avg_weight
        G.edges[e]["title"] = G.edges[e]["matches"]

    # Compute node positions using the spring layout
    pos = nx.spring_layout(G, seed=0, weight="matches", iterations=100, scale=800)
    for key, array in pos.items():
        G.nodes[key]["x"] = array[0]
        G.nodes[key]["y"] = -array[1]

    # Compute communities using modularity
    C = nx.community.greedy_modularity_communities(G, "matches")
    i = 0

    for c in C:
        Cg = G.subgraph(c)  # community graph
        for n in Cg.nodes():
            G.nodes[n]["group"] = i
        i += 1

    nt.from_nx(G)
    nt.toggle_physics(False)

    cwd = os.getcwd()
    os.chdir(output_dir)
    out = os.path.join(output_dir,"graph.html")
    nt.write_html("graph.html", notebook=False, open_browser=False)
    print("View graph written at {}".format(out))
    os.chdir(cwd)
    
    return
