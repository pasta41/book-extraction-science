import seaborn as sns
from matplotlib.colors import ListedColormap

# Color palettes and colormaps used for different model families in plotting
# can slide a palette before passing to ListedColormap if want to reduce
# to a specific sub-part of the palette

llama1_palette = sns.color_palette("Reds", n_colors=1000)
llama1_cmap = ListedColormap(llama1_palette)
"""Red-based palette and colormap for Llama 1 models."""

llama2_palette = sns.color_palette("Purples", n_colors=1000)
llama2_cmap = ListedColormap(llama2_palette)
"""Purple-based palette and colormap for Llama 2 models."""

llama3_palette = sns.color_palette("Blues", n_colors=1000)
llama3_cmap = ListedColormap(llama3_palette)
"""Blue-based palette and colormap for Llama 3 models."""

pythia_palette = sns.color_palette("YlGn", n_colors=1000)
pythia_cmap = ListedColormap(pythia_palette)
"""Yellow-Green palette and colormap for Pythia family models."""

ds_palette = sns.color_palette("Greys", n_colors=1000)
ds_cmap = ListedColormap(ds_palette)
"""Grey-scale palette and colormap for DeepSeek models."""

phi_palette = sns.color_palette("Oranges", n_colors=1000)
phi_cmap = ListedColormap(phi_palette)
"""Orange palette and colormap for Phi models."""

gemma2_palette = sns.color_palette("Greens", n_colors=1000)
gemma2_cmap = ListedColormap(gemma2_palette)
"""Green palette and colormap for Gemma 2 models."""

qwen_palette = sns.color_palette("YlOrBr", n_colors=1000)
qwen_cmap = ListedColormap(qwen_palette)
"""Yellow-Orange-Brown palette and colormap for Qwen models."""

# Hatch patterns used in plotting for different bar styles
small_hatch = '/'
"""Small hatch pattern used for bar plots."""

medium_hatch = '+'
"""Medium hatch pattern used for bar plots."""

large_hatch = None
"""No hatch pattern, used for large bars or default style."""
