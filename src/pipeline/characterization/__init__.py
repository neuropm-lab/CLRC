"""Descriptive characterization of the summed NeuronChat network.

CLI-driven drivers that summarize the summed, cell-type-resolved
ligand-receptor network: which ligand-receptor interactions, brain regions
and cell types carry the most communication, and how communication decays
with fiber distance. Each driver ranks and prints its results and writes CSV
tables; none produce figures.

Drivers
-------
build_summed_matrix
    Sum the per-interaction NeuronChat matrices into one node-by-node matrix.
top_lr_interactions
    Rank ligand-receptor interactions by mean realized connectivity.
top_regions
    Rank brain regions by outgoing/incoming strength.
top_cells
    Rank cell types (and coarse classes) by outgoing/incoming strength.
distance_decay
    Build the cell-type-pair edge table joined with fiber length and quantify
    the distance-decay of connectivity per connection category.
"""
