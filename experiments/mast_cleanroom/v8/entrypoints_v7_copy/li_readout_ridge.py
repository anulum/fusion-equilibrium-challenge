# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Reactor Systems — real equal-family archive verification
"""Solve a ridge correction for one existing physical li readout with all other weights frozen."""
from dataclasses import replace
import numpy as np
from fec_adapter.diii_d_accuracy.network import silu

def hidden_features(network,features):
    """Return the original last hidden activations with original input scaling."""
    hidden=network.standardised(features)
    for weight,bias in zip(network.weights[:-1],network.biases[:-1],strict=True):
        hidden=silu(hidden@weight.T+bias)
    return hidden

def corrected_network(network,hidden,residual,ridge=0.001):
    """Fit only li_scored last-layer weights/bias by a fixed standardised ridge residual."""
    present=np.isfinite(residual)&np.isfinite(hidden).all(axis=1)
    x=np.asarray(hidden[present],dtype=np.float64);y=np.asarray(residual[present],dtype=np.float64)
    if len(y)<2:raise ValueError('Insufficient actual finite target rows')
    mean=x.mean(axis=0);spread=x.std(axis=0);spread=np.where(spread>0,spread,1.)
    z=(x-mean)/spread;target=y-y.mean()
    correction=np.linalg.solve(z.T@z/len(z)+ridge*np.eye(z.shape[1]),z.T@target/len(z))/spread
    bias=float(y.mean()-mean@correction)
    index=network.coordinates+network.label_names.index('li_scored')
    weights=[a.copy() for a in network.weights];biases=[a.copy() for a in network.biases]
    weights[-1][index]+=correction/network.output_scale[index];biases[-1][index]+=bias/network.output_scale[index]
    return replace(network,weights=tuple(weights),biases=tuple(biases)),{'rows':len(y),'ridge':ridge,'delta_weights_norm':float(np.linalg.norm(correction)),'delta_bias':bias}
