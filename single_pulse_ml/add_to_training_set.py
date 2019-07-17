import sys

from datetime import datetime, date
import numpy as np 
import h5py 

import reader

if len(sys.argv) != 5:
	print("\nExpected fnh5_orig fnh5_new label_new [0/1] name [dm/freq]\n")
        exit()

fn_orig = sys.argv[1]
fn_new = sys.argv[2]
y_n_ = np.int(sys.argv[3])
name = sys.argv[4]

assert name in ['dm', 'freq'], "name must be dm or freq"

# fn_orig = './data/20190712-20068-freqtime.hdf5'
# fn_orig = './data/20190712-20068-dmtime.hdf5'
# fn_new = '/home/arts/software/ARTS-obs/external/arts-analysis/Crab/data/data_sb00_36_full.hdf5'
# name = 'dm'
# y_n_ = 1

data_freq_o, y_o, data_dm_o, data_mb_o, params_o = reader.read_hdf5(fn_orig)

try:
	print("Assuming ranked output of classifier")
	g = h5py.File(fn_new,'r')
	data = g['data_frb_candidate']
	if name=='dm':
		data_dm_n = data
	elif name=='freq':
		data_freq_n = data[..., 0]
except:
	data_freq_n, _, data_dm_n, data_mb_n, params_n = reader.read_hdf5(fn_new)

try:
	ntrig_o_f = len(data_freq_o)
	nfreq_o = data_freq_o.shape[1]
except:
	ntrig_o_f = 0 

try:
	ntrig_o_dm = len(data_dm_o)
	ndm_o = data_dm_o.shape[1]
except:
	ntrig_o_dm = 0 
	ndm_o = 0

try:
	ntrig_n_f = len(data_freq_n)
	nfreq_n = data_freq_n.shape[1]
except:
	ntrig_n_f = 0 

try:
	ntrig_n_dm = len(data_dm_n)
	ndm_n = data_dm_n.shape[1]
except:
	ntrig_n_dm = 0 
	ndm_n = 0

if name=='freq':
	# Assume all new triggers are either FP (0) or TP (1)
	y_n = np.ones([len(data_freq_n)])*y_n_
elif name=='dm':
	y_n = np.ones([len(data_dm_n)])*y_n_

if ntrig_o_f>0 and name is 'freq':
	if nfreq_n>nfreq_o:
	    print('Rebinning data in frequency')
	    data_freq_n = data_freq_n.reshape((ntrig_n_f, nfreq_o, nfreq_n//nfreq_o) + data_freq_n.shape[2:])
	    data_freq_n = data_freq_n.mean(2)
	    dshape = data_freq_n.shape

	# normalize data
	data_freq_n = data_freq_n.reshape(len(data_freq_n), -1)
	data_freq_n -= np.median(data_freq_n, axis=-1)[:, None]
	data_freq_n /= np.std(data_freq_n, axis=-1)[:, None]

	# zero out nans
	data_freq_n[data_freq_n!=data_freq_n] = 0.0
	data_freq_n = data_freq_n.reshape(dshape)

	data_freq_full = np.concatenate([data_freq_o, data_freq_n])
	y_full = np.concatenate([y_o, y_n])

	currentDate = date.today()
	fnout = './data/' + currentDate.strftime('%Y%-m%-d') + '-freqtime.hdf5'

	f = h5py.File(fnout, 'w')
	f.create_dataset('labels', data=y_full)

	print('\nWrote to %s' % fnout)
	print('All events are labelled %d\n' % np.mean(y_n))
	print('Class balance is:\n%0.2f RFI\n%0.2f FRB' % (y_full.sum()/float(len(y_full)), 1-y_full.sum()/float(len(y_full))))	

	if ntrig_n_f > 0:
		try:
			f.create_dataset('data_freq_time', data=data_freq_full)
		except:
			print("Could not write freq/time")

if ntrig_o_dm>0 and name is 'dm':
	dshape = data_dm_n.shape
	# normalize data
	data_dm_n = data_dm_n.reshape(len(data_dm_n), -1)
	data_dm_n -= np.median(data_dm_n, axis=-1)[:, None]
	data_dm_n /= np.std(data_dm_n, axis=-1)[:, None]

	# zero out nans
	data_dm_n[data_dm_n!=data_dm_n] = 0.0
	data_dm_n = data_dm_n.reshape(dshape)

	data_dm_full = np.concatenate([data_dm_o, data_dm_n])
	y_full = np.concatenate([y_o, y_n])

	currentDate = date.today()
	fnout = './data/' + currentDate.strftime('%Y%-m%-d') + '-dmtime.hdf5'

	f = h5py.File(fnout, 'w')
	f.create_dataset('labels', data=y_full)

	print('\nWrote to %s' % fnout)
	print('All events are labelled %d\n' % np.mean(y_n))


	if ntrig_n_dm > 0:
		try:
			f.create_dataset('data_dm_time', data=data_dm_full)
		except:
			print("Could not write dm/time")







