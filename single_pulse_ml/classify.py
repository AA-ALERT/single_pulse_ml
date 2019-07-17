# Liam Connor 25 July 2018
# Script to classify single-pulses 
# using tensorflow/keras model. Output probabilities 
# can be saved and plotted

import optparse
import numpy as np
import h5py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
mpl.use('pdf', warn=False)

import frbkeras
import reader
import plot_tools


def classify(data, model, save_ranked=False, 
             plot_ranked=False, prob_threshold=0.5,
             fnout='ranked', nside=5, params=None,
             ranked_ind=None, ind_frb=None, 
             yaxlabel='Freq', tab=None, sb=False, DMgal=np.inf,
             ):

    if ranked_ind is not None:
        prob_threshold = 0.0

    if type(model)==str:
        print("Modelstring", model)
        model = frbkeras.load_model(model)
        
    mshape = model.input.shape
    dshape = data.shape
    
    if yaxlabel=='Freq':
        if dshape[1]>mshape[1]:
            print('Rebinning data in frequency')
            data = data.reshape((dshape[0], mshape[1], dshape[1]//int(mshape[1])) + dshape[2:])
            data = data.mean(2)
            dshape = data.shape

    # normalize data
    data = data.reshape(len(data), -1)
    data -= np.median(data, axis=-1)[:, None]
    data /= np.std(data, axis=-1)[:, None]

    # zero out nans
    data[data!=data] = 0.0
    data = data.reshape(dshape)

    if dshape[-1]!=1:
        data = data[..., None]

    if len(mshape)==3:
        data = data.mean(1)
        dshape = data.shape

    if tab is None:
        tab = -1*np.ones([len(data)])

    if yaxlabel=='Freq':
        print("ROLLING TIME AXIS TO MAX PIXEL")
        for ii, dd in enumerate(data[..., 0]):
            mx_pix = np.argmax(dd.mean(0))
            NTIME = dd.shape[1]
            dd = np.roll(dd, NTIME//2-mx_pix, axis=1)
            data[ii] = dd[..., None]

    if mshape[1]<dshape[1]:
        print('Mismatch axis 1')
        nm = int(mshape[1])
        nd = dshape[1]
        data = data[:, nd//2-nm//2:nd//2+nm//2]
    elif mshape[1]>dshape[1]:
        print("Error: Model expects:", mshape)
        print("Data has:", dshape) 

        return [], []

    if mshape[2]<dshape[2]:
        print('Mismatch axis 2')
        nm = int(mshape[2])
        nd = dshape[2]
        data = data[:, :, nd//2-nm//2:nd//2+nm//2]
    elif mshape[2]>dshape[2]:
        print("Error: Model expects:", mshape)
        print("Data has:", dshape) 

        return [],[]

    y_pred_prob = model.predict(data)
    y_pred_prob = y_pred_prob[:,1]

    if ind_frb is None:
        print("Getting ind")
        ind_frb = np.where(y_pred_prob>prob_threshold)[0]
    
    print("\n%d out of %d events with probability > %.2f:\n %s" % 
            (len(ind_frb), len(y_pred_prob), 
                prob_threshold, ind_frb))

    if len(ind_frb)==0:
        print("No events above the threshhold. Exiting.")
        return [],[]

    low_to_high_ind = np.argsort(y_pred_prob)

    if save_ranked is True:
        print("Need to fix the file naming")
#        fnout_ranked = fn_data.rstrip('.hdf5') + \
#                       'freq_time_candidates.hdf5'

        fnout_ranked = fnout + '.hdf5'

        g = h5py.File(fnout_ranked, 'w')
        g.create_dataset('data_frb_candidate', data=data[ind_frb])
        g.create_dataset('frb_index', data=ind_frb)
        g.create_dataset('probability', data=y_pred_prob)
        g.create_dataset('params', data=params)
        if sb:
            g.create_dataset('sb', data=tab[ind_frb])
        else:
            g.create_dataset('tab', data=tab[ind_frb])
        g.close()
        print("\nSaved them and all probabilities to: \n%s" % fnout_ranked)

    if plot_ranked is True:
        print('plotting')
        if save_ranked is False:
            argtup = (data[ind_frb], ind_frb, y_pred_prob)
            ranked_ind_ = plot_tools.plot_multiple_ranked(argtup, nside=nside, \
                                            fnfigout=fnout, ascending=False, 
                                            params=params[ind_frb], ranked_ind=ranked_ind,
                                            yaxlabel=yaxlabel, tab=tab[ind_frb], sb=sb, DMgal=DMgal)
        else:
            ranked_ind_ = plot_tools.plot_multiple_ranked(fnout_ranked, nside=nside, \
                                            fnfigout=fnout, ascending=False,
                                            params=params[ind_frb], ranked_ind=ranked_ind,
                                            yaxlabel=yaxlabel, tab=tab[ind_frb], sb=sb, DMgal=DMgal)
    else:
        ranked_ind_ = np.argsort(y_pred_prob)[::-1]

    return data, ind_frb, ranked_ind_, y_pred_prob


def run_main(fn_data, fn_model_freq, options, DMgal=np.inf):
    print("Using datafile %s" % fn_data)
    print("Using keras model in %s" % fn_model_freq)

    if options.sb:
        return_sb = True
        return_tab = False
    else:
        return_sb = False
        return_tab = True
    data_freq, y, data_dm, data_mb, params, beam = reader.read_hdf5(fn_data, return_tab=return_tab, return_sb=return_sb)
    
    dms = params[:, 1]

    NFREQ = data_freq.shape[1]
    NTIME = data_freq.shape[2]
    WIDTH = options.twindow

    # low time index, high time index
    tl, th = NTIME//2-WIDTH//2, NTIME//2+WIDTH//2

    if data_freq.shape[-1] > (th-tl):
        data_freq = data_freq[..., tl:th]

#    fn_fig_out = options.fnout + '_freq_time_dm%0.1f-%0.1f' % (dm_min, dm_max)
    fn_fig_out_freq = options.fnout + '_%0.1f-%0.1f_freq_time' % (dms.min(), dms.max())

    print("\nCLASSIFYING FREQ/TIME DATA\n")

    data_freq, ind_frb, ranked_ind_freq, y_prob_freq = classify(data_freq, fn_model_freq, 
                             save_ranked=options.save_ranked, 
                             plot_ranked=options.plot_ranked, 
                             prob_threshold=options.prob_threshold,
                             fnout=fn_fig_out_freq, params=params, 
                             nside=options.nside, yaxlabel='Freq', tab=beam, sb=options.sb, DMgal=DMgal)


    if len(ind_frb)==0:
        return

    if options.fn_model_dm is not None:
        if len(data_dm)>0:
            print("\nCLASSIFYING DM/TIME DATA\n)")
            fn_fig_out_dm = options.fnout + '_dm_time'

            data_dm, ind_frb_dm, ranked_ind_freq_, y_prob_dm = classify(data_dm,
                                 options.fn_model_dm, 
                                 save_ranked=options.save_ranked, 
                                 plot_ranked=False, 
                                 prob_threshold=options.prob_threshold,
                                 fnout=fn_fig_out_dm, params=params, 
                                 nside=options.nside, ind_frb=None,
                                 ranked_ind=None, yaxlabel='DM', DMgal=DMgal)

            # Remove candidates where DM/time probability is below 
            ind_remove = np.where(y_prob_dm[ind_frb]<options.prob_threshold_dm)[0]
            ind_frb = np.delete(ind_frb, ind_remove)
        else:
            print("No DM/time data to classify")

    if options.fn_model_time is not None:
        print("\nCLASSIFYING 1D TIME DATA\n)")
        fn_fig_out_time = options.fnout + '_time'
        # classify(data_freq, options.fn_model_time, 
        #      save_ranked=options.save_ranked, 
        #      plot_ranked=options.plot_ranked, 
        #      prob_threshold=options.prob_threshold,
        #      params=params, 
        #      nside=options.nside, ind_frb=ind_frb,
        #      ranked_ind=ranked_ind_freq, yaxlabel='', DMgal=DMgal)

        data_time, ind_frb_time, ranked_ind_freq_, y_prob_time = classify(data_time,
                                 options.fn_model_time, 
                                 save_ranked=options.save_ranked, 
                                 plot_ranked=False, 
                                 prob_threshold=options.prob_threshold,
                                 params=params, 
                                 nside=options.nside, ind_frb=None,
                                 ranked_ind=None, yaxlabel='DM', DMgal=DMgal)

    if options.fn_model_mb is not None:
        classify(data_mb, options.fn_model_mb, 
             save_ranked=options.save_ranked, 
             plot_ranked=options.plot_ranked, 
             prob_threshold=options.prob_threshold,
             fnout=options.fnout, params=params, ind_frb=ind_frb,
             nside=options.nsidem, ranked_ind=ranked_ind_freq, DMgal=DMgal)

    if options.plot_ranked is True:
        print('Plotting')
        if options.save_ranked is False:
            argtup = (data_freq[ind_frb], ind_frb, y_prob_freq)

            ranked_ind_ = plot_tools.plot_multiple_ranked(argtup, nside=options.nside, \
                                            fnfigout=fn_fig_out_freq, ascending=False, 
                                            params=params[ind_frb], ranked_ind=None,
                                            yaxlabel='Freq', tab=beam[ind_frb], 
                                            sb=options.sb, DMgal=DMgal)
        else:
            fnout_ranked = options.fnout + '.hdf5'
            ranked_ind_ = plot_tools.plot_multiple_ranked(fnout_ranked, \
                                            nside=options.nside, 
                                            fnfigout=fn_fig_out_freq, 
                                            ascending=False,
                                            params=params[ind_frb], 
                                            ranked_ind=None,
                                            yaxlabel='Freq', tab=beam[ind_frb], 
                                            sb=options.sb, DMgal=options.DMgal)

        if options.fn_model_dm is not None:
            argtup = (data_dm[ind_frb], ind_frb, y_prob_dm)

            ranked_ind_freq_final = np.argsort(y_prob_freq[ind_frb])[::-1]
            ranked_ind_dm = plot_tools.plot_multiple_ranked(argtup, nside=options.nside, \
                                          fnfigout=fn_fig_out_dm, ascending=False,
                                          params=params[ind_frb], ranked_ind=ranked_ind_freq_final,
                                          yaxlabel='DM', tab=beam[ind_frb], sb=options.sb, 
                                          DMgal=DMgal)


        if options.fn_model_time is not None:
            argtup = (data_time[ind_frb], ind_frb, y_prob_time)
            
            ranked_ind_freq_final = np.argsort(y_prob_freq[ind_frb])[::-1]
            ranked_ind_dm = plot_tools.plot_multiple_ranked(argtup, nside=options.nside, \
                                          fnfigout=fn_fig_out_time, ascending=False,
                                          params=params[ind_frb], ranked_ind=ranked_ind_freq_final,
                                          yaxlabel='DM', tab=beam[ind_frb], sb=options.sb, 
                                          DMgal=DMgal)


if __name__=="__main__":
    parser = optparse.OptionParser(prog="classify.py", \
                        version="", \
                        usage="%prog FN_DATA FN_MODEL [OPTIONS]", \
                        description="Apply DNN model to FRB candidates")

    parser.add_option('--fn_model_dm', dest='fn_model_dm', type='str', \
                        help="Filename of dm_time model. Default None", \
                        default=None)

    parser.add_option('--fn_model_time', dest='fn_model_time', type='str', \
                        help="Filename of 1d time model. Default None", \
                        default=None)

    parser.add_option('--fn_model_mb', dest='fn_model_mb', type='str', \
                        help="Filename of multibeam model. Default None", \
                        default=None)

    parser.add_option('--pthresh', dest='prob_threshold', type='float', \
                        help="probability treshold", default=0.5)

    parser.add_option('--pthresh_dm', dest='prob_threshold_dm', type=float,
                      default=0.0)

    parser.add_option('--save_ranked', dest='save_ranked', 
                        action='store_true', \
                        help="save FRB events + probabilities", \
                        default=False)

    parser.add_option('--plot_ranked', dest='plot_ranked', \
                        action='store_true',\
                        help="plot triggers", default=False)

    parser.add_option('--twindow', dest='twindow', type='int', \
                        help="time width, default 64", default=64)

    parser.add_option('--fnout', dest='fnout', type='str', \
                       help="beginning of figure names", \
                       default='ranked')

    parser.add_option('--nside', dest='nside', type='int', \
                       help="number of rows/cols of subplots per figure", \
                       default=5)

    parser.add_option('--DMgal', dest='DMgal', type='float', \
                       help="expected DM contribution from Milky Way",\
                       default=np.inf)

    parser.add_option('--synthesized_beams', dest='sb', action='store_true',
                       help="Use synthesized beams instead of TABs")

    options, args = parser.parse_args()

    assert len(args)==2, "Arguments are FN_DATA FN_MODEL [OPTIONS]"

    fn_data = args[0]
    fn_model_freq = args[1]

    run_main(fn_data, fn_model_freq, options, DMgal=options.DMgal)
    exit()

#    if options.DMgal > 0:
#        run_main(fn_data, fn_model_freq, options, dm_min=0., dm_max=options.DMgal)
#        run_main(fn_data, fn_model_freq, options, dm_min=options.DMgal)
#    else:
#        run_main(fn_data, fn_model_freq, options)








            
