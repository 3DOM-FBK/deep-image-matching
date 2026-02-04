var highlightActive = false;

// initialize global variables.
var edges;
var nodes;
var allNodes;
var allEdges;
var container = document.getElementById('mynetwork');
var nodeColors;
var originalNodes;
var network;
var options, data;
var positions;
var net;
var filter = {
    item : '',
    property : '',
    value : []
};
var clusterIndex = 0;
  var clusters = [];
  var lastClusterZoomLevel = 0;
  var clusterFactor = 0.9;

// This method is responsible for drawing the graph, returns the drawn network
function drawGraph() {
  // parsing and collecting nodes and edges from the python
  nodes = new vis.DataSet({{nodes|tojson}});
  edges = new vis.DataSet({{edges|tojson}});

  nodeColors = {};
  allNodes = nodes.get({ returnType: "Object" });
  allEdges = edges.get({ returnType: "Object" });

  // adding nodes and edges to the graph
  data = {nodes: nodes, edges: edges};
  
  edgesIds = edges.getIds();
  
  var options = {{options|safe}};
  net = options;
  options = {
    "configure": {
        "enabled": false
    },
    // },
    // "groups":{
    //     "useDefaultGroups": false,
    //     "0": { "color":{ "border": "#2B7CE9", "background": "#97C2FC"} },
    //     "1": { color:{ border: "#FFA500", background: "#FFFF00"} },
    //     "2": { color:{ border: "#FA0A10", background: "#FB7E81"} },
    //     "3": { color:{ border: "#41A906", background: "#7BE141"} }
    // },
    "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 2,
        "font": {
            "size": 6
        }
    },
    "edges": {
        "color": {
        "inherit": 'from'
        },
        "smooth": {
            "type": "continuous",
            "forceDirection": "none",
            "roundness": 0.2
        }
    },
    "interaction": {
        "dragNodes": true,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false,
        "hideEdgesOnZoom": true,
        "selectConnectedEdges": true, //change to false for debugging
    },
    "physics": {
        "enabled": false,
        "stabilization": {
        "enabled": true,
        "fit": true,
        "iterations": 200,
        "onlyDynamicEdges": false,
        "updateInterval": 50,
        }
    }
    }

  network = new vis.Network(container, data, options);

  for (nodeId in allNodes) {
    nodeColors[nodeId] = network.groups.get(allNodes[nodeId].group).color;
  }

  positions = network.getPositions();

//   network.on("selectNode", function (params){
//             if (network.isCluster(params.nodes[0]) == true) {
//                 console.log(params.nodes[0]);
//                 network.openCluster(params.nodes[0], {releaseFunction: releaseFunction});
//             }else{
//                 showImagePopup(params);
//             }
//       });

  edgesColor = {};
  for (let edgeId of edgesIds){
    edgesColor[edgeId] = nodeColors[allEdges[edgeId].from];
  }

   network.on("click", function(params){
     neighbourhoodHighlight(params, edgesColor);
   });

  // Show edge weight in left panel. DEBUG only
  network.on("selectEdge", function(params){
        if(!params.edges[0].includes("clusterEdge:"))
            document.getElementById("edge_details").innerHTML = "" + edges.get(params.edges[0]).title;
  });
      
  network.on("dragStart", function (){
    if (document.getElementById("imagePopup") != null)
        if(imagePopup.style.display != 'none')
            closeImagePopup(imagePopup);
    
  });

  
  network.on("deselectNode", function (){
    if (document.getElementById("imagePopup") != null)
        if(imagePopup.style.display != 'none'){
            closeImagePopup(imagePopup);
        }
  });

  network.on("zoom", function (params){

    if (document.getElementById("imagePopup") != null)
        imagePopup.style.transform = "scale(" + params.scale + ")";
  });

//   network.once("stabilizationIterationsDone", function() {
//     network.setOptions( { physics: false } );
    
//     });
{% if nodes|length > 30 %}
network.setOptions( {
                nodes: {
                     opacity: 0.5
                    }
                });
{% endif %}

  return network;
}

function releaseFunction(){
  return positions;
}

function toggleSidePanel() {
const sidePanel = document.getElementById('sidePanel');
const mainContent = document.getElementById('mynetwork');
const reopenButton = document.getElementById('reopenButton');

if (sidePanel.style.width === '20%') {
    sidePanel.style.width = '0%';
    mainContent.style.flexGrow = '1';
    setTimeout(function(){
        reopenButton.style.display = 'block';
    }, 250);
} else {
    sidePanel.style.width = '20%';
    mainContent.style.flexGrow = '0';
    reopenButton.style.display = 'none';
}
}

function reopenSidePanel() {
const sidePanel = document.getElementById('sidePanel');
const reopenButton = document.getElementById('reopenButton');

sidePanel.style.width = '20%';
reopenButton.style.display = 'none';
}

function populateSidePanel(){
const sidePanel = document.getElementById('sidePanel');
const table = sidePanel.children[2];
const tbody = table.getElementsByTagName('tbody')[0];
tbody.rows[0].cells[0].innerHTML += net["properties"]["edges"];
tbody.rows[1].cells[0].innerHTML += net["properties"]["communities"].length;
tbody.rows[2].cells[0].innerHTML += net["properties"]["aligned"];
tbody.rows[3].cells[0].innerHTML += net["properties"]["not_aligned"];

j = 0
if (net["properties"]["communities"].length > 1){
    //document.getElementById("cluster").innerHTML += "<input class="comm_button" type='button' onclick='clusterByCommunity()' value='Cluster by community' />";
    tbody.rows[1].cells[0].innerHTML +="    ▼"
    tbody.rows[1].cells[0].style.cursor = "pointer";
    for (let index=0; index< net["properties"]["communities"].length; index++) {
        newrow = tbody.insertRow(j + 2);
        newrow.className = "comm";
        newcell = newrow.insertCell(0);
        newcell.innerHTML = "<input type='button' id='btn_" + index + "' class='comm_button' style='color:" + network.groups.get(index).color.border + ";' value='#" + (j+1) +  " size: "+ net['properties']['communities'][index] + "'/>";
        document.getElementById("btn_" + index).addEventListener("click", function() {
            clusterByCommunity(index, network.groups.get(index).color);
        }, false);
        j += 1;
    }
}

if (net["properties"]["not_aligned"] != 0){
    tbody.rows[3 + j].cells[0].innerHTML +="    ▼"
    tbody.rows[3 + j].cells[0].style.cursor = "pointer";
}

var i = 0;
for (nodeId in allNodes) {
    if( !allNodes[nodeId].aligned ){
        newrow = tbody.insertRow(i + j + 4)
        newrow.className = "na"
        newrow.insertCell(0).innerHTML = "[" + nodeId + "]  " + allNodes[nodeId].title;
        i += 1;
    }
}
}

function toggleRows(clickedRow, className) {
var rowsToToggle = clickedRow.nextElementSibling;

// Toggle the display property of all rows with class 'na' below the clicked row
while (rowsToToggle && rowsToToggle.classList.contains(className)) {
    if (rowsToToggle.style.display === 'none' || rowsToToggle.style.display === '') {
        rowsToToggle.style.display = 'table-row';
    } else {
        rowsToToggle.style.display = 'none';
    }

    rowsToToggle = rowsToToggle.nextElementSibling;
}
}

function closeImagePopup(imagePopup) {
imagePopup.style.display = 'none';
document.getElementById('mynetwork').removeChild(imagePopup);
}

// showing the popup
function showImagePopup(params) {
  // get the data from the vis.DataSet
  nodeId = params.nodes[0];
  var nodeData = nodes.get(nodeId);
    var imagePopup = document.createElement("div");
    imagePopup.className = 'imagePopup';
    imagePopup.id = 'imagePopup';
    container.appendChild(imagePopup);
  
    const path = document.baseURI;
    path_array = path.split('/');
    path_array.splice(0, 2);
    path_array.splice(-2);
    img_path = path_array.join('/').concat('/images/');
    imagePopup.style.backgroundImage = "url(" + img_path.concat(nodeData.title) + ")";
    

  // get the position of the node
  var posCanvas = network.getPositions([nodeId])[nodeId];

  // get the bounding box of the node
  var boundingBox = network.getBoundingBox(nodeId);

  //position tooltip:
  posCanvas.x = posCanvas.x + 0.5 * (boundingBox.right - boundingBox.left);

  // convert coordinates to the DOM space
  var posDOM = network.canvasToDOM(posCanvas);

  // Give it an offset
  posDOM.x += 10;
  posDOM.y -= 100;

  // show and place the tooltip.
  imagePopup.style.display = 'block';
  imagePopup.style.top = posDOM.y + 'px';
  imagePopup.style.left = posDOM.x + 'px';
}

function neighbourhoodHighlight(params, edgesColor) {
  allNodes = nodes.get({ returnType: "Object" });
  allEdges = edges.get({ returnType: "Object" });
  // if something is selected:
  if (params.nodes.length > 0) {
    if (network.isCluster(params.nodes[0]) == true) {
      network.openCluster(params.nodes[0], {releaseFunction: releaseFunction});
    }else{
      if(highlightActive === true){
        for (var edgeId in allEdges) {
          allEdges[edgeId].color = edgesColor[edgeId].border;
      }

      }
      highlightActive = true;
      showImagePopup(params);
      var selectedNode = params.nodes[0];

      var connectedEdges = network.getConnectedEdges(selectedNode);
      
      for (let edgeId in allEdges){
        if(!(connectedEdges.includes(allEdges[edgeId].id)))
          allEdges[edgeId].color = "rgba(200,200,200,0.5)";
      }
    }
    }else if (highlightActive === true) {
      // reset all nodes
      for (var edgeId in allEdges) {
        allEdges[edgeId].color = edgesColor[edgeId].border;
      }
      highlightActive = false;
    }
    // transform the object into an array
    var updateEdgeArray = [];
    for (let edgeId in allEdges) {
      if (allEdges.hasOwnProperty(edgeId)) {
        updateEdgeArray.push(allEdges[edgeId]);
      }
    }
    edges.update(updateEdgeArray);
}

function filterHighlight(params) {
  allNodes = nodes.get({ returnType: "Object" });
  // if something is selected:
  if (params.nodes.length > 0) {
    filterActive = true;
    let selectedNodes = params.nodes;

    // hiding all nodes and saving the label
    for (let nodeId in allNodes) {
      allNodes[nodeId].hidden = true;
      if (allNodes[nodeId].savedLabel === undefined) {
        allNodes[nodeId].savedLabel = allNodes[nodeId].label;
        allNodes[nodeId].label = undefined;
      }
    }

    for (let i=0; i < selectedNodes.length; i++) {
      allNodes[selectedNodes[i]].hidden = false;
      if (allNodes[selectedNodes[i]].savedLabel !== undefined) {
        allNodes[selectedNodes[i]].label = allNodes[selectedNodes[i]].savedLabel;
        allNodes[selectedNodes[i]].savedLabel = undefined;
      }
    }

  } else if (filterActive === true) {
    // reset all nodes
    for (let nodeId in allNodes) {
      allNodes[nodeId].hidden = false;
      if (allNodes[nodeId].savedLabel !== undefined) {
        allNodes[nodeId].label = allNodes[nodeId].savedLabel;
        allNodes[nodeId].savedLabel = undefined;
      }
    }
    filterActive = false;
  }

  // transform the object into an array
  var updateArray = [];
  if (params.nodes.length > 0) {
    for (let nodeId in allNodes) {
      if (allNodes.hasOwnProperty(nodeId)) {
        updateArray.push(allNodes[nodeId]);
      }
    }
    nodes.update(updateArray);
  } else {
    for (let nodeId in allNodes) {
      if (allNodes.hasOwnProperty(nodeId)) {
        updateArray.push(allNodes[nodeId]);
      }
    }
    nodes.update(updateArray);
  }
}

function selectNode(nodes) {
  network.selectNodes(nodes);
  neighbourhoodHighlight({ nodes: nodes });
  return nodes;
}

function selectNodes(nodes) {
  network.selectNodes(nodes);
  filterHighlight({nodes: nodes});
  return nodes;
}

function highlightFilter(filter) {
  let selectedNodes = []
  let selectedProp = filter['property']
  if (filter['item'] === 'node') {
    let allNodes = nodes.get({ returnType: "Object" });
    for (let nodeId in allNodes) {
      if (allNodes[nodeId][selectedProp] && filter['value'].includes((allNodes[nodeId][selectedProp]).toString())) {
        selectedNodes.push(nodeId)
      }
    }
  }
  else if (filter['item'] === 'edge'){
    let allEdges = edges.get({returnType: 'object'});
    // check if the selected property exists for selected edge and select the nodes connected to the edge
    for (let edge in allEdges) {
      if (allEdges[edge][selectedProp] && filter['value'].includes((allEdges[edge][selectedProp]).toString())) {
        selectedNodes.push(allEdges[edge]['from'])
        selectedNodes.push(allEdges[edge]['to'])
      }
    }
  }
  selectNodes(selectedNodes)
}

function clusterByCommunity(index, color) {
  //network.setData(data);
  var clusterOptionsByData;
  //for (var i = 0; i < network.groups["communities"].length; i++) {
    clusterOptionsByData = {
      joinCondition: function (childOptions) {
        return childOptions.group === index;
      },
      processProperties: function (clusterOptions, childNodes, childEdges) {
        clusterOptions.color = color;
        console.log(clusterOptions);
        return clusterOptions;
      },
      clusterNodeProperties: {
        borderWidth: 3,
        shape: "circle",
        color: color,
        label: "community:" + index,
      },
    };
    network.cluster(clusterOptionsByData);
  }
