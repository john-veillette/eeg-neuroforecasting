from mne.io.egi.egi import _read_header, _read_events
import pandas as pd
import numpy as np

def read_events(fpath):
    '''
    Reads neuroforecasting image events from our EGI raw files
    '''

    with open(fpath, 'rb') as fid:
        egi_info = _read_header(fid)
        egi_events = _read_events(fid, egi_info)
    event_codes = list(egi_info['event_codes'])
    event_dict = {ev: i for i, ev in enumerate(event_codes)}
    project_ids = [ev for ev in event_codes if 'PR' in ev]

    def get_idxs(ev):
        ev_bool = egi_events[event_dict[ev], :]
        ev_idxs = np.argwhere(ev_bool)
        return ev_idxs.flatten()

    # grab project IDs
    proj_events = []
    project_ids = [ev for ev in event_codes if 'PR' in ev]
    for ev in project_ids:
        idx = get_idxs(ev)[0]
        proj_events.append([ev, idx])
    proj_events = pd.DataFrame(proj_events, columns = ('project', 'sample'))
    proj_events.sort_values('sample', inplace = True)
    proj_events.reset_index(drop = True, inplace = True)

    # organize triggers/behavioral data (found at true event times)
    img_events = []
    img_ids = [ev for ev in event_codes if ev[0] == 'I' and ev not in ('IBEG', 'IEND')]
    for ev in img_ids:
        idxs = get_idxs(ev)
        for idx in idxs:
            category = 'face' if ev[1] == 'F' else 'place'
            if ev[2] == 'Y':
                choice = 1
            elif ev[2] == 'N':
                choice = 0
            else: # subject ran out of time
                choice = None
            funded = 1 if ev[3] == 'F' else 0
            img_events.append([idx, funded, choice, category])
    img_events = pd.DataFrame(img_events,
                           columns = ('sample', 'funded', 'choice', 'category'))
    img_events.sort_values('sample', inplace = True)
    img_events.reset_index(drop = True, inplace = True)

    # merge project labels and triggers/behaviorals
    img_events['project_id'] = proj_events.project
    return img_events, event_codes
