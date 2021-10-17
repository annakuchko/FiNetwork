import logging
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib
from matplotlib.colors import to_hex
from collections import Counter
from numpy import cos, pi, sin
from finetwork.optimiser._centrality_metrics import _CentralityMetrics
from matplotlib.colors import LinearSegmentedColormap


def n_group_colorpallet(groups):
    groups = np.unique(list(groups.values()))
    n = len(groups)
    col_list = [
        '#FEA3A3', '#FFAD78', '#FFCF78', '#FFED78', '#F5FF78', '#D9FF78',
        '#B8FF78', '#78FF7E', '#78FFB2', '#78FFDD', '#78F1FF', '#97C0FF',
        '#B497FF', '#E797FF', '#FCA0F2', '#FCA0CF', '#A0FCEA', '#FFEECC'
        ]
    cmap = LinearSegmentedColormap.from_list(
        name='my_colormap', 
        colors=col_list,
        N=n)

    cols_dict = {}
    for i in range(cmap.N):
        cols_dict[groups[i]] = cmap(i)
        
    return cols_dict

def node_theta(nodelist, node):
    assert len(nodelist) > 0, "nodelist must be a list of items."
    assert node in nodelist, "node must be inside nodelist."

    i = nodelist.index(node)
    theta = -np.pi + i * 2 * np.pi / len(nodelist)

    return theta

def group_theta(node_length, node_idx):
    theta = -np.pi + node_idx * 2 * np.pi / node_length
    return theta



def text_alignment(x, y):
    if x == 0:
        ha = "center"
    elif x > 0:
        ha = "left"
    else:
        ha = "right"
    if y == 0:
        va = "center"
    elif y > 0:
        va = "bottom"
    else:
        va = "top"

    return ha, va


def get_cartesian(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return x, y


def circos_radius(n_nodes, node_r):
    A = 2 * np.pi / n_nodes
    B = (np.pi - A) / 2
    a = 2 * node_r
    return a * np.sin(B) / np.sin(A)

def to_proper_radians(theta):
    if theta > pi or theta < -pi:
        theta = theta % pi
    return theta


def despine(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


class _CircosPlot(object):
    def __init__(
        self,
        graph,
        node_order=None,
        node_size=0.5,
        node_grouping=None,
        group_order="alphabetically",
        node_color=None,
        node_labels=None,
        edge_width=None,
        edge_color=None,
        nodeprops=None,
        edgeprops=None,
        label_order=None,
        node_label_color=False,
        group_label_position=None,
        group_label_color=False,
        fontsize=14,
        fontfamily="serif",
        legend_handles=None,
        node_label_layout=None,
        group_label_offset=None,
        specified_layout=None,
        nodes_pos=None,
        limit_orig=None,
        lim=None,
        label_meta=None,
        partition=None,
        centrality_metrics='betweenness_centrality',
        **kwargs,
    ):
        # Set graph object
        self.graph = graph
        self.partition = partition
        self.label_order = label_order
        self.nodes = list(graph.nodes())  # keep track of nodes separately.
        # print('Label order: ', label_order)
        if label_order == None:
            key = lambda x : self.partition[x]
            (sorted(self.graph, key=key))
        else:
            key = label_order
        self.label_order = key
        self.nodes = sorted(self.graph, key=key)
        self.edges = list(graph.edges())
        # Set node arrangement
        self.node_order = node_order
        self.node_grouping = node_grouping
        self.group_order = group_order
        self.specified_layout = specified_layout
        self.nodes_pos = nodes_pos
        self.node_label_layout = node_label_layout
        self.limit_orig = limit_orig
        self.lim = lim
        self.centrality_metrics = centrality_metrics
        self.label_meta = label_meta
        
        
        d = dict(_CentralityMetrics(self.graph, self.centrality_metrics)._compute_metrics())
        dm = max(d.values())
        
        if dm<=0.25:
            adj_f=0.075
            
        elif dm>0.25 and dm<=0.45:
            adj_f=0.1
            
        elif dm>0.45 and dm<=0.65:
            adj_f=0.15
            
        elif dm>0.65 and dm<=0.85:
            adj_f=0.25
        
        elif dm>0.85 and dm<=1.05:
            adj_f=0.35
            
        else:
            adj_f = .4
        
        self.d_r = [v*adj_f for v in d.values()]
        if self.nodes_pos==None:
            node_r = max([v*adj_f for v in d.values()]) 
            nodes_pos = circos_radius(n_nodes=len(graph.nodes()), node_r=node_r)
            self.nodes_pos = nodes_pos
       
        
        # Set node colors
        self.node_color = node_color
        self.sm = None  # sm -> for scalarmappable. See https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots  # noqa
        logging.debug("INIT: {0}".format(self.sm))
        if self.node_color:
            self.node_colors = []
            self.compute_node_colors()
        else:
            self.node_colors = n_group_colorpallet(self.partition)
        self.node_labels = node_labels

        # Set edge properties
        self.edge_width = edge_width
        if self.edge_width:
            self.edge_widths = []
            self.compute_edge_widths()
        else:
            # self.edge_widths = [1.25] * len(self.edges)

            
            # if 'pearson'
            # sc_const = -2.
            
            # if 'pearson' and scaled 
            sc_const = -0.15
            
            # if 'atkinson'
            # sc_const=-0.1
            
            # if 'atkinson' and scaled
            # sc_const = 0 +without log

            
            self.edge_widths = [
                sc_const*np.log(w)/50*len(self.nodes) for u,v,w in self.graph.edges.data("weight")
                ]
        
        
        # edgeprops are matplotlib line properties. These can be set after
        # instantiation but before calling the draw() function.
        if edgeprops:
            self.edgeprops = edgeprops
        else:
            self.edgeprops = {"facecolor": '#3BF69F',
                              "alpha": 0.015}
        if node_label_color:
            self.node_label_color = self.node_colors
        else:
            self.node_label_color = ["#3BCAF6"] * len(self.nodes)
            
            
        self.edge_color = edge_color
        if self.edge_color:
            self.edge_colors = []
            self.compute_edge_colors()
        else:
            self.edge_colors = ["#3BF69F"] * len(self.edges)

        fs = 15 / 50 * len(self.nodes)
        figsize = (fs, fs)
        
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        self.figure = plt.figure(figsize=figsize, facecolor='black',
                                 )
        
        self.ax = self.figure.add_subplot(1, 1, 1)

        
        despine(self.ax)
        
        # We provide the following attributes that can be set by the end-user.
        # nodeprops are matplotlib patches properties.
        
        if nodeprops:
            self.nodeprops = nodeprops()
        else:
            self.nodeprops = {"edgecolor": '#3BF69F'}
       

        # Compute each node's positions.
        self.compute_node_positions()

        # Conditionally compute node label positions.
        self.compute_node_label_positions()

        # set group properties
        self.group_label_position = group_label_position
        self.groups = []
        if group_label_position:
            self.compute_group_label_positions()
            if group_label_color:
                self.compute_group_colors()
            else:
                self.group_label_color = ['#3BF69F'] * len(self.nodes)

        # set text properties
        valid_fonts = ["serif", "sans-serif", "fantasy", "monospace"]
        if fontfamily not in valid_fonts:
            raise ValueError(f"fontfamily should be one of {valid_fonts}")
        self.fontfamily = fontfamily
        self.fontsize = fontsize # * 50 / len(self.nodes)
        
        # Verify that the provided input is legitimate
        valid_node_label_layouts = (None, "rotation", "numbers")
        assert specified_layout in valid_node_label_layouts
        # Store the node label layout
        self.node_label_layout = specified_layout

        # Store the group label offset
        self.group_label_offset = group_label_offset
        # Add legend to plot
        if "group_legend" in kwargs.keys():
            if kwargs["group_legend"]:
                self.draw_legend()
    
    def _compute_positions(self):
        
        return self.nodes_pos


    def _draw(self, save=True, figname=None, title=None):
        self.ax.set_title(f'Graph for period {title}', 
                          color='white', fontsize=14)
        self.ax.patch.set_linewidth('1.2')      
        self.ax.set_facecolor('black')
        self.ax.axes.margins(0.15,tight=True)
        self.draw_nodes()
        self.draw_edges()
        if self.limit_orig==None:
            lim =  max(abs(np.array(self.ax.dataLim.bounds))) * 0.75
            self.lim = lim
        else:
            lim = self.limit_orig
            
        self.ax.set_ylim(-lim, lim)
        
        self.ax.set_xlim(-lim, lim)
        if hasattr(self, "groups") and self.groups:
            self.draw_group_labels()
        logging.debug("DRAW: {0}".format(self.sm))
        if self.sm:
            self.figure.subplots_adjust(right=0.8)
            cax = self.figure.add_axes([0.85, 0.2, 0.05, 0.6])
            self.figure.colorbar(self.sm, cax=cax)
  
        self.figure.tight_layout(pad = 0)
        # plt.show()
        plt.savefig(f'{figname}.png',
                     bbox_inches="tight", 
                     pad_inches=0)
        matplotlib.pyplot.close()

    def _get_lim_orig(self):
        return self.lim
    
    def _get_label_order(self):
        return self.label_order

    def compute_edge_widths(self):
        
        if isinstance(self.edge_width, str):
            edges = self.graph.edges
            self.edge_widths = [edges[n][self.edge_width] for n in self.edges]
        else:
            self.edge_widths = self.edge_width


   
    def compute_group_label_positions(self):
        """
        Computes the x,y positions of the group labels.
        """
        assert self.group_label_position in ["beginning", "middle", "end"]
        data = [self.graph.nodes[n][self.node_grouping] for n in self.nodes]
        node_length = len(data)
        groups = Counter(data)
        
        radius = self.nodes_pos
        xs = []
        ys = []
        has = []
        vas = []
        node_idcs = np.cumsum(list(groups.values()))
        node_idcs = np.insert(node_idcs, 0, 0)
        if self.group_label_position == "beginning":
            for idx in node_idcs[:-1]:
                x, y = get_cartesian(
                    r=radius, theta=group_theta(node_length, idx)
                )
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        elif self.group_label_position == "middle":
            node_idcs = node_idcs.reshape(len(node_idcs), 1)
            node_idcs = np.concatenate((node_idcs[:-1], node_idcs[1:]), axis=1)
            for idx in node_idcs:
                theta1 = group_theta(node_length, idx[0])
                theta2 = group_theta(node_length, idx[1] - 1)
                x, y = get_cartesian(r=radius, theta=(theta1 + theta2) / 2)
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        elif self.group_label_position == "end":
            for idx in node_idcs[1::]:
                x, y = get_cartesian(
                    r=radius, theta=group_theta(node_length, idx - 1)
                )
                ha, va = text_alignment(x, y)
                xs.append(x)
                ys.append(y)
                has.append(ha)
                vas.append(va)

        self.group_label_coords = {"x": xs, "y": ys}
        self.group_label_aligns = {"has": has, "vas": vas}
        self.groups = groups.keys()

    def compute_node_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        in the Circos plot.
        """
        xs = []
        ys = []
        radius = self.nodes_pos
        self.plot_radius =  radius
        self.nodeprops["radius"] = 0.0005
        self.nodeprops["linewidth"] = radius * 0.0
        for node in self.nodes:
            x, y = get_cartesian(r=radius, theta=node_theta(self.nodes, node))
            xs.append(x)
            ys.append(y)
        self.node_coords = {"x": xs, "y": ys}
        
        
    def compute_node_label_positions(self):
        """
        Uses the get_cartesian function to compute the positions of each node
        label in the Circos plot.

        This method is always called after the compute_node_positions
        method, so that the plot_radius is pre-computed.
        This will also add a new attribute, `node_label_rotation` to the object
        which contains the rotation angles for each of the nodes. Together with
        the node coordinates this can be used to add additional annotations
        with rotated text.
        """
        self.init_node_label_meta()
   
        for i, node in enumerate(self.nodes):
            
            # Define radius 'radius' and circumference 'theta'
            
            # multiplication factor 1.02 moved below
            radius = self.plot_radius + self.nodeprops["radius"]
            theta = node_theta(self.nodes, node)
            radius_adjustment = 1.15
            x, y = get_cartesian(r=radius*radius_adjustment, theta=theta)
            # ----- For numbered nodes -----

            # Node label x-axis coordinate
            tx, _ = get_cartesian(r=radius, theta=theta)
            
            # Create the quasi-circular positioning on the x axis
            tx *= 1 - np.log(np.cos(theta) * self.nonzero_sign(np.cos(theta)))
            # Move each node a little further away from the circos
            tx += self.nonzero_sign(x)
           
            # Node label y-axis coordinate numerator
            numerator = radius * (
                theta % (self.nonzero_sign(y) * self.nonzero_sign(x) * np.pi)
            )
            # Node label y-axis coordinate denominator
            denominator = self.nonzero_sign(x) * np.pi
            # Node label y-axis coordinate
            ty = 2 * (numerator / denominator)

            # ----- For rotated nodes -----
            
            # Computes the text rotation
            theta_deg = to_proper_radians(theta) / pi * 180
            if theta_deg >= -90 and theta_deg <= 90:  # right side
                rot = theta_deg
            else:  # left side
                rot = theta_deg - 180
            if self.label_meta == None:
                self.store_node_label_meta(x, y, tx, ty, rot)
            else:
                self.store_node_label_meta(x, y, tx, ty, rot)
                self.node_label_coords = self.label_meta
      
    def _get_label_meta(self):
        return self.node_label_coords

    
    @staticmethod
    def nonzero_sign(xy):
        """
        A sign function that won't return 0
        """
        return -1 if xy < 0 else 1


    def init_node_label_meta(self):
        """
        This function ensures that self.node_label_coords
        exist with the correct keys and empty entries
        This function should not be called by the user
        """

        # Reset node label coorc/align dictionaries
        self.node_label_coords = {"x": [], "y": [], "tx": [], "ty": []}
        self.node_label_aligns = {"has": [], "vas": []}
        self.node_label_rotation = []


    def store_node_label_meta(self, x, y, tx, ty, rot):
        self.node_label_coords["x"].append(x)
        self.node_label_coords["y"].append(y)
        self.node_label_coords["tx"].append(tx)
        self.node_label_coords["ty"].append(ty)

        # Computes the text alignment for x
        if x == 0:
            self.node_label_aligns["has"].append("center")
        elif x > 0:
            self.node_label_aligns["has"].append("left")
        else:
            self.node_label_aligns["has"].append("right")

        # Computes the text alignment for y
        if self.node_label_layout == "rotation" or y == 0:
            self.node_label_aligns["vas"].append("center")
        elif y > 0:
            self.node_label_aligns["vas"].append("bottom")
        else:
            self.node_label_aligns["vas"].append("top")

        self.node_label_rotation.append(rot)


    def draw_nodes(self):
        ec = self.nodeprops["edgecolor"]
        lw = self.nodeprops["linewidth"]
        d_r = self.d_r
        import matplotlib
        handles=[]
        for n_col in self.node_colors:
            colored_circle = matplotlib.patches.Circle(
                [],
                color=self.node_colors[n_col], 
                label=n_col)
            handles.append(colored_circle)
        labels = [h.get_label() for h in handles] 
        self.ax.legend(handles=handles, 
                       labels=labels,  
                       # prop={'size': 20},
                       fontsize=self.fontsize,
                       markerscale=2.5,
                       facecolor='black',
                       edgecolor='black',
                       labelcolor='mfc',
                       loc=8,
                       ncol=3,
                       borderaxespad=-5
                       )
        
        for i, node in enumerate(self.nodes):
            x = self.node_coords["x"][i]
            y = self.node_coords["y"][i]
            color = self.node_colors[self.partition[node]]
            node_patch = patches.Circle(
                (x, y), d_r[i],
                zorder=0, 
                ec=ec, 
                lw=lw,
                fc=color,
                alpha=1,
                in_layout=False
            )

            self.ax.add_patch(node_patch)
            
            if self.node_labels:
                label_x = self.node_label_coords["x"][i]
                label_y = self.node_label_coords["y"][i]
                label_tx = self.node_label_coords["tx"][i]
                label_ty = self.node_label_coords["ty"][i]
                label_ha = self.node_label_aligns["has"][i]
                label_va = self.node_label_aligns["vas"][i]
                # ----- Node label rotation layout -----

                if self.node_label_layout == "rotation":
                    rot = self.node_label_rotation[i]
                    
                    self.ax.text(
                        s=node,
                        x=label_x,
                        y=label_y,
                        ha=label_ha,
                        va=label_va,
                        rotation=rot,
                        rotation_mode="anchor",
                        # rotation_mode="deGfault",
                        color=color,
                        fontsize=self.fontsize,
                        family=self.fontfamily,
                    )

                # ----- Node label numbering layout -----

                elif self.node_label_layout == "numbers":

                    # Draw descriptions for labels
                    desc = "%s - %s" % ((i, node) if (x > 0) else (node, i))
                    self.ax.text(
                        s=desc,
                        x=label_tx,
                        y=label_ty,
                        ha=label_ha,
                        va=label_va,
                        color=color,
                        fontsize=self.fontsize,
                        family=self.fontfamily,
                    )

                    # Add numbers to nodes
                    self.ax.text(
                        s=i, x=label_x, y=label_y, ha="center", va="center"
                    )

                # Standard node label layout
                else:
                    # Draw node text straight from the nodes
                    self.ax.text(   
                        s=node,
                        x=label_x,
                        y=label_y,
                        ha=label_ha,
                        va=label_va,
                        color=color,
                        fontsize=self.fontsize,
                        family=self.fontfamily,
                    )


    def draw_edges(self):
        for i, (start, end) in enumerate(self.graph.edges()):
            start_theta = node_theta(self.nodes, start)
            end_theta = node_theta(self.nodes, end)
            verts = [
                get_cartesian(self.plot_radius, start_theta),
                (0, 0),
                get_cartesian(self.plot_radius, end_theta),
            ]
            color = self.edge_colors[i]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            lw = self.edge_widths[i]
            path = Path(verts, codes)
            patch = patches.PathPatch(
                path, lw=lw, ec=color, zorder=3, **self.edgeprops
            )
            self.ax.add_patch(patch)
            patch = patches.PathPatch(
                path, lw=lw, ec=color, zorder=5, fc='none', alpha=0.25
            )
            self.ax.add_patch(patch)

    def draw_group_labels(self):
        for i, label in enumerate(self.groups):
            label_x = self.group_label_coords["x"][i]
            label_y = self.group_label_coords["y"][i]
            label_ha = self.group_label_aligns["has"][i]
            label_va = self.group_label_aligns["vas"][i]
            color = self.group_label_color[i]
            self.ax.text(
                s=label,
                x=label_x,
                y=label_y,
                ha=label_ha,
                va=label_va,
                color=color,
                fontsize=self.fontsize,
                family=self.fontfamily,
            )


    def draw_legend(self):
        
        seen = set()
        colors_group = [
            x for x in self.node_colors if not (x in seen or seen.add(x))
        ]

        # Gets group labels
        labels_group = sorted(
            set([self.graph.nodes[n][self.node_color] for n in self.nodes])
        )

        # Create patchList to use as handle for plt.legend()
        patchlist = []
        for color, label in zip(colors_group, labels_group):
            # Convert RGBA to HEX value
            color = to_hex(color, keep_alpha=True)
            data_key = patches.Patch(color=color, label=label)
            patchlist.append(data_key)

        # Set the labels with the custom patchList
        self.ax.legend(
            handles=patchlist,
            loc="lower center",
            # Half number of columns for total of labels for the
            # groups
            ncol=int(len(labels_group) / 2),
            bbox_to_anchor=(0.5, -0.05),
        )

        # Make legend handle accessible for manipulation outside of the plot
        self.legend_handles = patchlist
