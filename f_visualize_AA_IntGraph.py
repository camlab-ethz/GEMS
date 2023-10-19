from torch_geometric.utils import remove_self_loops
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.io as io
from sklearn.neighbors import NearestNeighbors





def visualize_graph(graph, 
                    title,
                    highlight=None,
                    highlight_symbol = 'x',
                    show_edges = True,
                    show_edge_attr=False, 
                    add_traces = [], 
                    linewidth=2, 
                    markersize=3):


    # To prepare a list of all the edges in the graph, and a list of same shape that countains the features of these edges
    # For this, remove double edges (undirected graph) and remove self_loops

    if show_edge_attr:
        edge_list, edge_attr = remove_self_loops(graph.edge_index, graph.edge_attr)
        edge_list = edge_list.T.tolist()
        edge_attr = edge_attr.tolist()

        edges=[]
        hoverinfo_edges = []
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))
                hoverinfo_edges.append(edge_attr[idx])

        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_nodes = graph.x.tolist()
        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]

        for n in range (len(hoverinfo_edges)):
            hoverinfo_edges[n] = [int(entry) if entry % 1 == 0 else round(entry, 4) for entry in hoverinfo_edges[n]]


    else: 
        edge_list, _ = remove_self_loops(graph.edge_index)
        edge_list = edge_list.T.tolist()
    
        edges=[]
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))

        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_edges = 'None'
        hoverinfo_nodes = graph.x.tolist()
        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]


    N = graph.x.shape[0]

    #Marker Colors based on atom type
    atomtypes = (graph.x[:,320:361] == 1).nonzero(as_tuple=True)[1].tolist() #identify the index of the first 1 in the feature matrix = atom type
    
    marker_color_mapping = {0:'rgb(0,0,0)', 3:'rgb(34,139,34)', 2:'rgb(65,105,225)', 5:'rgb(255,0,0)', 
                            4:'rgb(0,0,255)', 6:'rgb(0,255,0)', 5:'rgb(255,0,0)', 7:'rgb(0,0,0)',
                            10:'rgb(255,120,0)', 8:'rgb(0,0,0)', 11:'rgb(238,210,2)'}
    markercolor = [marker_color_mapping[atom] if atom in marker_color_mapping.keys() else 'rgb(0,0,0)' for atom in atomtypes]

    #Marker shape based on atom type (Fe, Cl and the atoms in highligh as crosses, the rest as circles)
    not_ions = [0,2,3,4,5,6,10,11]
    symbols = ['x' if atom not in not_ions else 'circle' for atom in atomtypes]
    if highlight != None:
        symbols = [highlight_symbol if index in highlight else symbol for index, symbol in enumerate(symbols)]


    # Prepare the coordinates of the nodes and edges
    atomcoords = graph.pos.numpy()

    Xn=[atomcoords[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[atomcoords[k][1] for k in range(N)]# y-coordinates
    Zn=[atomcoords[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        Xe+=[atomcoords[e[0]][0],atomcoords[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[atomcoords[e[0]][1],atomcoords[e[1]][1], None]# y-coordinates of edge ends
        Ze+=[atomcoords[e[0]][2],atomcoords[e[1]][2], None]# z-coordinates of edge ends


    # Configure Plot, trace1 = edges, trace2 = nodes
    trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(50,50,50)', width=linewidth),
               text=hoverinfo_edges,
               #textposition= 'middle center',
               hoverinfo = 'text'
               )

    trace2=go.Scatter3d(x=Xn,
                    y=Yn,
                    z=Zn,
                    mode='markers',
                    marker=dict(symbol=symbols,
                                size=markersize,
                                color=markercolor,
                                #colorscale = 'viridis',
                                line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                    text=hoverinfo_nodes,
                    hoverinfo='text'
                    )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title=title,
            width=2000,
            height=2000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
    )

    # Add the traces that are given in add_traces
    if show_edges:
        data = [trace1, trace2] + add_traces
    else: 
        data = [trace2] + add_traces


    # PLOT
    fig=go.Figure(data=data, layout=layout)
    io.renderers.default='notebook'

    iplot(fig, filename='3d-scatter-colorscale')





def graph_to_traces(graph,
                    symbol = 'circle',
                    highlight=None,
                    highlight_symbol = 'x',
                    show_edge_attr=False, 
                    linewidth=2, 
                    markersize=3):


    # To prepare a list of all the edges in the graph, and a list of same shape that countains the features of these edges
    # For this, remove double edges (undirected graph) and remove self_loops

    if show_edge_attr:
        edge_list, edge_attr = remove_self_loops(graph.edge_index, graph.edge_attr)
        edge_list = edge_list.T.tolist()
        edge_attr = edge_attr.tolist()

        edges=[]
        hoverinfo_edges = []
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))
                hoverinfo_edges.append(edge_attr[idx])

        print('Here')

        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_nodes = graph.x.tolist()
        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]

        for n in range (len(hoverinfo_edges)):
            hoverinfo_edges[n] = [int(entry) if entry % 1 == 0 else round(entry, 4) for entry in hoverinfo_edges[n]]


    else: 
        edge_list, _ = remove_self_loops(graph.edge_index)
        edge_list = edge_list.T.tolist()
    
        edges=[]
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))


        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_edges = 'None'
        hoverinfo_nodes = graph.x.tolist()
        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]


    N = graph.x.shape[0]

    #Marker Colors based on atom type
    atomtypes = (graph.x[:,:27] == 1).nonzero(as_tuple=True)[1].tolist() #identify the index of the first 1 in the feature matrix = atom type
    marker_color_mapping = {5:'rgb(34,139,34)', 6:'rgb(65,105,225)', 7:'rgb(255,0,0)', 15:'rgb(255,255,0)', 25:'rgb(0,0,0)', 16:'rgb(255,105,180)'}
    markercolor = [marker_color_mapping[atom] for atom in atomtypes]

    #Marker shape based on atom type (Fe, Cl and the atoms in highligh as crosses, the rest as circles)
    symbols = [highlight_symbol if atom==26 or atom==17 else symbol for atom in atomtypes]
    if highlight != None:
        symbols = [highlight_symbol if index in highlight else sym for index, sym in enumerate(symbols)]


    # Prepare the coordinates of the nodes and edges
    atomcoords = graph.pos.numpy()

    Xn=[atomcoords[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[atomcoords[k][1] for k in range(N)]# y-coordinates
    Zn=[atomcoords[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        Xe+=[atomcoords[e[0]][0],atomcoords[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[atomcoords[e[0]][1],atomcoords[e[1]][1], None]# y-coordinates of edge ends
        Ze+=[atomcoords[e[0]][2],atomcoords[e[1]][2], None]# z-coordinates of edge ends



    # Configure Plot, 

    # trace1 = edges
    trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(50,50,50)', width=linewidth),
               text=hoverinfo_edges,
               #textposition= 'middle center',
               hoverinfo = 'text'
               )

    # trace2 = nodes
    trace2=go.Scatter3d(x=Xn,
                    y=Yn,
                    z=Zn,
                    mode='markers',
                    marker=dict(symbol=symbols,
                                size=markersize,
                                color=markercolor,
                                #colorscale = 'viridis',
                                line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                    text=hoverinfo_nodes,
                    hoverinfo='text'
                    )
    
    return trace1, trace2






def get_interaction_trace(graph1, graph2, threshold, linetype = 'longdash', linecolor = 'rgb(125,125,125)', linewidth=1):

    '''Graph1 should be the smaller graph, the one we loop over'''

    graph2_coords = graph2.pos.numpy()

    knn = NearestNeighbors(n_neighbors=50)
    knn.fit(graph2_coords)

    threshold = threshold
    Xe = []
    Ye = []
    Ze = []

    # For every atom of graph1
    graph1_coords = graph1.pos.numpy()

    for atom in graph1_coords: 
        
        # Find the nearest neighbors and the distances
        dist, neighbors = knn.kneighbors([atom], return_distance=True)

        dist = dist[0]
        neighbors = neighbors[0]
    
        for index, neigh in enumerate(neighbors):
            if dist[index] < threshold:

                graph1_atom = atom
                graph2_atom = graph2_coords[neigh]

                Xe+=[graph1_atom[0], graph2_atom[0], None]# x-coordinates of edge ends
                Ye+=[graph1_atom[1], graph2_atom[1], None]
                Ze+=[graph1_atom[2], graph2_atom[2], None]


    interaction_trace = go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color=linecolor, width=linewidth, dash=linetype),
               hoverinfo='none'
               )
    
    return interaction_trace