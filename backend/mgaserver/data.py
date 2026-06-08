import pypsa
import pandas as pd
import pandera.pandas as pa
import numpy as np
import linopy
from pathlib import Path

DimensionSpec = dict[str, dict[str, dict[str, float]]]

# def load_points(path: Path, dimensions: DimensionSpec) -> pd.DataFrame:
# 	"""Load all PyPSA .nc files in a directory and extract values for each dimension.

# 	Returns a DataFrame with one row per .nc file, indexed by filename stem.
# 	"""
# 	component_attrs = {
# 		"Generator": lambda n: n.generators,
# 		"Load": lambda n: n.loads,
# 		"StorageUnit": lambda n: n.storage_units,
# 		"Line": lambda n: n.lines,
# 		"Bus": lambda n: n.buses,
# 		"Link": lambda n: n.links,
# 	}

# 	rows: dict[str, dict[str, float]] = {}
# 	for nc_file in sorted(path.glob("*.nc")):
# 		network = pypsa.Network()
# 		network.import_from_netcdf(nc_file)

# 		row: dict[str, float] = {}
# 		for dim_name, component_specs in dimensions.items():
# 			value = 0.0
# 			for component_type, attr_specs in component_specs.items():
# 				df = component_attrs[component_type](network)
# 				for attr, name_weights in attr_specs.items():
# 					col = df[attr]
# 					for comp_name, weight in name_weights.items():
# 						value += float(col[comp_name]) * weight
# 			row[dim_name] = value

# 		rows[nc_file.stem] = row

# 	return pd.DataFrame.from_dict(rows, orient="index")



