import numpy as np
import os.path as op
from pprint import pformat
# EEG utilities
import mne
from mne.preprocessing import ICA, create_eog_epochs
from pyprep.prep_pipeline import PrepPipeline
from autoreject import get_rejection_threshold
# BIDS utilities
from mne_bids import BIDSPath, read_raw_bids
from util.io.bids import DataSink
from bids import BIDSLayout

# constants / config
BIDS_ROOT = 'bids'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
HIGHPASS = .3 # low cutoff for filter
LOWPASS = 50. # high cutoff for filter

# gather our bearings
layout = BIDSLayout(BIDS_ROOT, derivatives = True)
subjects = layout.get_subjects()
subjects.sort()

already_done = layout.get_subjects(scope = 'preprocessing')

for i, sub in enumerate(subjects):

    if sub in already_done:
        continue

    np.random.seed(i)

    # grab the data
    bids_path = BIDSPath(
        root = BIDS_ROOT,
        subject = sub,
        task = 'kickstarter',
        datatype = 'eeg'
        )
    raw = read_raw_bids(bids_path, verbose = False)
    events, _ = mne.events_from_annotations(raw)

    # run PREP pipeline (highpass, notch, exclude bad chans, and re-reference)
    raw.load_data()
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": [raw.info['line_freq']]
        }
    prep = PrepPipeline(raw, prep_params, raw.get_montage(),
                            ransac = False, random_state = i)
    raw = None # this is now stored in PREP object
    prep.fit()
    raw = prep.raw # cleaned data

    # interpolate channels that are still noisy
    if sub == '109': # get channel we know prep misses for some reason
        raw.info['bads'] = ['E44']
    else:
        raw.info['bads'] = []
    bads = prep.noisy_channels_original
    bads['bad_after_PREP'] = raw.info['bads'] + prep.still_noisy_channels
    del prep # save memory
    raw.interpolate_bads() # chans manually marked bad

    # bandpass
    raw.filter(l_freq = HIGHPASS, h_freq = LOWPASS)

    def reref(dat): # rereference eye electrodes to become EOG
        dat[1,:] = (dat[1,:] - dat[0,:]) * -1
        return dat
    raw.apply_function(reref, picks = ['E25', 'E126'], channel_wise = False)
    raw.apply_function(reref, picks = ['E8', 'E127'], channel_wise = False)
    raw.set_channel_types({'E126': 'eog', 'E127': 'eog'})

    # segment and then downsample
    epochs = mne.Epochs(raw, events, tmin = -0.2, tmax = 0.8, baseline = None)
    epochs.load_data()
    del raw # free memory
    epochs.resample(sfreq = 2 * LOWPASS)

    # compute ICA components
    ica = ICA(n_components = 15, random_state = 0)
    ica.fit(epochs, picks = 'eeg')
    # and exclude ICA components that are correlated with EOG
    eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold = 1.96)
    ica.exclude = eog_indices
    ica.apply(epochs) # transforms in place

    # apply baseline correction *AFTER* ICA
    epochs.apply_baseline((-.2, 0.))
    # and remove EOG channels now that we're done with them
    epochs.drop_channels(['E126', 'E127'])

    # and then reject bad trials
    thres = get_rejection_threshold(epochs) # finds optimal by CV (5-fold)
    epochs.drop_bad(reject = thres)
    fig_erp = epochs.average().plot(spatial_colors = True, show = False)

    # save the cleaned data
    sink = DataSink(DERIV_ROOT, 'preprocessing')
    fpath = sink.get_path(
                    subject = sub,
                    task = 'kickstarter',
                    desc = 'clean',
                    suffix = 'epo',
                    extension = 'fif.gz'
                    )
    epochs.save(fpath)

    # generate a report
    report = mne.Report(verbose = True)
    report.parse_folder(op.dirname(fpath), pattern = '*epo.fif.gz',
                            render_bem = False)
    report.add_figs_to_section(
        fig_erp,
        captions = 'Average Evoked Response',
        section = 'evoked'
        )
    if ica.exclude:
        fig_ica = ica.plot_components(ica.exclude, show = False)
        report.add_figs_to_section(
            fig_ica,
            captions = 'Removed ICA Components',
            section = 'ICA'
            )
    html_lines = []
    for line in pformat(bads).splitlines():
        html_lines.append('<br/>%s' % line)
    html = '\n'.join(html_lines)
    report.add_htmls_to_section(html, captions = 'Interpolated Channels',
                            section = 'channels')
    crit = '<br/>threshold: {:0.2f} microvolts</br>'.format(thres['eeg'] * 1e6)
    report.add_htmls_to_section(crit, captions = 'Trial Rejection Criteria',
                            section = 'rejection')
    report.add_htmls_to_section(epochs.info._repr_html_(),
                            captions = 'Info',
                            section = 'info')
    report.save(op.join(sink.deriv_root, 'sub-%s.html'%sub), overwrite = True)
