import numpy as np                                                                                                                                                                    

COLORMAPS = {                                                                                                                                                                         
    # ColorBrewer Set1 (vibrant, high contrast)
    "set1": {
        "plane":    np.array([0.894, 0.102, 0.110]),  # #E41A1C red
        "sphere":   np.array([0.216, 0.494, 0.722]),  # #377EB8 blue
        "cylinder": np.array([0.302, 0.686, 0.290]),  # #4DAF4A green
        "cone":     np.array([0.596, 0.306, 0.639]),  # #984EA3 purple
        "inr":      np.array([1.000, 0.498, 0.000]),  # #FF7F00 orange
        "bspline":  np.array([0.090, 0.745, 0.812]),  # #17BED0 cyan
    },                                                                                                                                                                                
    # ColorBrewer Set2 (softer, pastel)
    "set2": {
        "plane":    np.array([0.400, 0.761, 0.647]),  # #66C2A5 teal
        "sphere":   np.array([0.988, 0.553, 0.384]),  # #FC8D62 salmon
        "cylinder": np.array([0.553, 0.627, 0.796]),  # #8DA0CB periwinkle
        "cone":     np.array([0.906, 0.541, 0.765]),  # #E78AC3 pink
        "inr":      np.array([0.651, 0.847, 0.329]),  # #A6D854 lime
        "bspline":  np.array([0.553, 0.827, 0.780]),  # #8DD3C7 light teal
    },                                                                                                                                                                                
    # ColorBrewer Dark2 (muted, professional)
    "dark2": {
        "plane":    np.array([0.106, 0.620, 0.467]),  # #1B9E77 teal
        "sphere":   np.array([0.851, 0.373, 0.008]),  # #D95F02 orange
        "cylinder": np.array([0.459, 0.439, 0.702]),  # #7570B3 purple
        "cone":     np.array([0.906, 0.161, 0.541]),  # #E7298A magenta
        "inr":      np.array([0.400, 0.651, 0.118]),  # #66A61E green
        "bspline":  np.array([0.141, 0.580, 0.580]),  # #249494 dark cyan
    },
}

ACTIVE_COLORMAP = "set1"

def get_surface_color(surface_type):
    return COLORMAPS[ACTIVE_COLORMAP][surface_type]
