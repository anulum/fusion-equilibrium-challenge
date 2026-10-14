# SCPN Reactor Systems — M3c-1b step 1b-a (fec venv): Thomson Te 5 % markers of the step-1 frames (inputs only)
import glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'v8b'))
from edge_boundary import OFFSET, markers
from fec_adapter.shot_reader import read_shot
files = sorted(glob.glob('$DISK/_offroot_cache/fec_data_local/mast_public_test/*.parquet'))
frames = json.loads((HERE / 'step1_frames.json').read_text())
for fr in frames:
    shot = read_shot(Path(files[fr['shot']]))
    o = np.argsort(shot.thomson.r); r, te = shot.thomson.r[o], shot.thomson.te[fr['frame'], o]
    ok = np.isfinite(te) & (te > 0)
    m_in, m_out = markers(r[ok], te[ok])
    fr['z0'] = float(np.median(shot.thomson.z))
    fr['r_in'] = None if m_in is None else float(m_in - OFFSET)
    fr['r_out'] = None if m_out is None else float(m_out + OFFSET)
    cols = sorted(set(shot.coils.columns)); fr['circuits'] = cols
(HERE / 'step1b_frames.json').write_text(json.dumps(frames, indent=1))
print(sum(f['r_in'] is not None and f['r_out'] is not None for f in frames), 'of', len(frames), 'frames with both markers')
