# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — D27 axis-flux discretisation repair
"""Use the integration grid's axis cell as its reference flux."""
import numpy as np
from fec_adapter.cross_machine import map_scalars


def electron_beta(flux, reader, shot, index):
    reading = reader.read(flux)
    if reading.contour is None:
        return float('nan')
    axis = reading.axis
    row = int(np.argmin(np.abs(shot.z_grid - axis[1])))
    column = int(np.argmin(np.abs(shot.r_grid - axis[0])))
    # normalised_beta seeds its connected plasma component at this exact cell.
    # Bilinear interpolation at a subcell axis can put that cell below depth zero.
    axis_flux = float(flux[row, column])
    edge_flux = float(np.mean(map_scalars._at(flux, shot.r_grid, shot.z_grid,
                                           reading.contour[:, 0], reading.contour[:, 1])))
    if not np.isfinite(edge_flux - axis_flux) or abs(edge_flux - axis_flux) < 1e-12:
        return float('nan')
    return map_scalars.normalised_beta(
        flux, shot.r_grid, shot.z_grid, axis, axis_flux, edge_flux,
        shot.thomson.r, shot.thomson.z, shot.thomson.te[index], shot.thomson.ne[index],
        float(shot.plasma_current[index]),
        map_scalars.vacuum_f('DIII-D', float(shot.tf_current[index]))) / map_scalars.PRESSURE_FACTOR
