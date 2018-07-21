"""Loads data stored in the formats used by the Neuralynx recording systems."""

import numpy as np

def loadNVT(filename):
    """function to load NVT file"""
    f = open(filename, 'rb')	
    header = str(f.read(2 ** 14)).strip('\x00')
    dt = np.dtype([('swstx', np.uint16), ('swid', np.uint16), ('swdata_size', np.uint16),
                ('qwTimestamp', np.uint64), ('dwPoints', np.int32, (400,)), ('sncrc', np.int16), 
                ('extractedX', np.int32 ), ('extractedY', np.int32), ('extractedAngle', np.int32),
                ('dwTargets', np.int32, (50,))])
    data = np.fromfile(f, dt)
    return data


def loadNcs(filename, should_d2a=True, should_read_time=True, trim_zeros=True,
            return_header=False, auto_invert=True):
    """Loads a neuralynx .ncs electrode file.
    Keyword arguments:
    should_d2a -- convert from integer to microVolt units (default True)
    should_read_time -- read and return timestamps (default True)
    trim_zeros -- removes traling zeros from the trace (default True)
    return_header -- return the file header (default True)
    auto_invert -- invert signal if header shows it is (default True)
    Returns (in order):
    - Voltage trace (in uV if should_d2a = True)
    - Timestamps (double, microseconds) if should_read_time = True
    - Integer to microVolt conversion factor (double) if should_d2a = False
    Usage:
    csc, ts = nlxio.loadNcs('TT4E1.ncs')
    csc, ts, d2a = nlxio.loadNcs('TT4E1.ncs', should_d2a=False)
    csc = nlxio.loadNcs('TT4E1.ncs', should_read_time=False)
    csc, d2a = nlxio.loadNcs('TT4E1.ncs', should_d2a=False,
                                    should_read_time = false)
    """
    f = open(filename, 'rb')

    # Nlx files have a 16kbyte header
    header = str(f.read(2 ** 14)).strip('\x00')

    # The format for a .ncs files according the the neuralynx docs is
    # uint64 - timestamp in microseconds
    # uint32 - channel number
    # uint32 - sample freq
    # uint32 - number of valid samples
    # int16 x 512 - actual csc samples
    dt = np.dtype([('time', '<Q'), ('channel', '<i'), ('freq', '<i'),
                ('valid', '<i'), ('csc', '<h', (512,))])
    # five points for fast numpy dtype reading
    temp = np.fromfile(f, dt)

    # unpack the csc matrix
    csc = temp['csc'].reshape((temp['csc'].size,))

    # .ncs files have a timestamp every 512 data points, so lets interpolate
    # the timestamps in between those - this is messy code, but works
    if should_read_time:
    	ts = np.zeros(csc.shape)
        ts[::512] = temp['time']
        xo = np.arange(ts.size)
        ts = np.interp(xo, xo[::512], ts[::512])
        # the last few are just filled with a constant, so increment them
        ts[-511:] = ts[-512] + (1 + np.arange(511)) * (ts[512] - ts[0]) / 512.0

    # strip trailing zeros (an artifact of my badly written downsample mcode)
    if trim_zeros:
        csc = np.trim_zeros(csc, 'b')
        if should_read_time:
            ts = ts[:csc.size]

    # now scan the header for a2d information
    global a2d_conversion
    for line in header.split('\n'):
        if line.strip().startswith('-ADBitVolts'):
            a2d_conversion = 1e6 * np.array(map(float, line.split()[1:5]))
        if line.strip().startswith('-InputInverted'):
            if (line.strip().split(' ')[1] == 'True') and auto_invert:
                csc = - csc

    if (a2d_conversion is None) and should_d2a:
        raise IOError("ADBitVolts not found in .ncs header for " + filename)
        
        
    csc = a2d_conversion * csc.astype(np.float64)
    retvals = [csc]
    
    if should_read_time:
        retvals.append(temp)
    if not should_d2a:
        retvals.append(a2d_conversion)
    if return_header:
        retvals.append(header.split('\n'))

    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)


""" this works because the ncs timestamps are sorted array on basis of time"""
def nextGreaterElement(timestamps, eventStartTimestamp):
	for index,ts in enumerate(timestamps):
		if ts > eventStartTimestamp:
			return ts
	raise ValueError("Nothing Found")


def nextSmallerElement(timestamps, eventStopTimestamp):
	try: return max(ts for ts in timestamps if ts< eventStopTimestamp)
	except ValueError: return "Nothing Found"

def fileSplitterUsingEvents(ncsData, eventStartTimestamp, eventStartName, eventStopTimestamp, eventStopName):
        """Splits the ncs data on the basis of event start time and event stop time.
    Keyword arguments:
    ncsData -- refers to the data collected over a channel
    eventStartTimestamp --  start time for the event
    eventStartName --  event name for the starting timestamp
    eventStopTimestamp --  stop time for the event
    eventStopName -- event name for the stopping timesamp
    Returns (in order):
    - frequency data in the range between event start time and event stop time
    """
	#store all the timestamps in one single list
	ncsTimestamp = []
	for ts in ncsData[1]:
		ncsTimestamp.append(ts[0])

	#get time stamp next to start time
	eventStartTime = nextGreaterElement(ncsTimestamp, eventStartTimestamp)
	#get time stamp just before stop time
	eventStopTime = nextSmallerElement(ncsTimestamp, eventStopTimestamp)

	dataPoints = []
	for ts in ncsData[1]:
		if ts[0]>= eventStartTime and ts[0]<=eventStopTime:
                  for t in ts[4]:
                     dataPoints.append(t*1e6*a2d_conversion)            
	return dataPoints


def loadTetrodeNcs(filename, should_d2a=True, trim_zeros=True):
    if filename.find('%C') == -1:
        raise ValueError("Filename passed to loadTetrodeNcs must contain %C.")

    # This is kind of slow, not sure why, should try to speed it up
    csc = []
    d2a = []
    ts = None
    for i in range(1, 5):
        print(i)
        tmp = loadNcs(filename.replace('%C', str(i)),
                      should_d2a=should_d2a,
                      should_read_time=(ts is None),
                      trim_zeros=trim_zeros)
        if not should_d2a:
            csc.append(tmp[0])
            d2a.append(tmp[-1])
        else:
            if ts is None:
                csc.append(tmp[0])
            else:
                csc.append(tmp)

        if ts is None:
            ts = tmp[1]

    csc = np.vstack(csc).T

    if should_d2a:
        return csc, ts
    else:
        d2a = np.hstack(d2a)
        return csc, ts, d2a


def loadNtt(filename, should_d2a=True):
    """Loads a neuralynx .ntt tetrode spike file.
    Keyword arguments:
    should_d2a -- convert from integer to microVolt units (default True)
    Returns:
    - Spikes as (num_spikes, length_waveform, num_channels) array
    - Spike times as uint64, microseconds
    - Sampling frequency in waveforms, Hz
    Usage:
    ts, sp, fs = nlxio.loadNtt('TT13.ntt')
    """
    f = open(filename, 'rb')
    header = f.read(2 ** 14)
    
    # A tetrode spike record is as folows:
    # uint64 - timestamp                    bytes 0:8
    # uint32 - acquisition entity number    bytes 8:12
    # uint32 - classified cel number        bytes 12:16
    # 8 * uint32- params                    bytes 16:48
    # 32 * 4 * int16 - waveform points
    # hence total record size is 2432 bits, 304 bytes
    # header is supposed to be 16kbyte, i.e. 16 * 2^10 = 2^14

    # Read the header and find the conversion factors / sampling frequency
    a2d_conversion = None
    fs = None
    for line in header.split('\n'):
        line = line.replace('\t', ' ')
        if line.strip().startswith('-SamplingFrequency'):
            fs = float(line.strip().split(' ')[1].strip())
        if line.strip().startswith('-ADBitVolts'):
            a2d_conversion = 1e6 * np.array(list(map(float, line.split()[1:5])))
    
    f.seek(2 ** 14)    # start of the spike, records
    # Neuralynx write little endian for some dumb reason
    dt = np.dtype([('time', '<Q'), ('filer', '<i', 10),
        ('spikes', np.dtype('<h'), (32, 4))])
    temp = np.fromfile(f, dt)

    if (a2d_conversion is None) and should_d2a:
        raise IOError("ADBitVolts not found in .ntt header for " + filename)
    if fs is None:
        raise IOError("Frequency not found in .ntt header for " + filename)

    if should_d2a:
        # if we've already done d2a theres no reason to return the conv factor
        return temp['time'], temp['spikes'] * np.reshape(a2d_conversion, [1, 1, 4]), fs
    else:
        return temp['time'], temp['spikes'], fs


def loadNev(filename):
    """Loads a neuralynx .nev file.
    Returns four numpy arrays:
    - Timestamp (uint64)
    - EventID (uint16)
    - Nttl (uint16)
    - Event String (charx128)
    ts, eid, nttl, estr = nlxio.loadNev('Events.nev')
    """
    f = open(filename, 'rb')
    # There's nothing useful in the header for .nev files, so skip past it
    f.seek(2 ** 14)
    # An event record is as follows:
    # int16 - nstx - reserved
    # int16 - npkt_id - id of the originating system
    # int16 - npkt_datn nhon a_size - this value should always be 2
    # uint64 - timestamp, microseconds
    # int16 - nevent_id - ID value for event
    # int16 - nttl - decimal TTL value read from the TTL input port
    # int16 - ncrc - record crc check, not used in consumer applications
    # int16 - ndummy1 - reserved
    # int16 - ndummy2 - reserved
    # int32x8 - dnExtra - extra bit values for this event
    # string(128) - event string
    dt = np.dtype([('filler1', '<h', 3), ('time', '<Q'), ('id', '<h'),
                   ('nttl', '<h'), ('filler2', '<h', 3), ('extra', '<i', 8),
                   ('estr', np.dtype('a128'))])
    temp = np.fromfile(f, dt)

    return temp['time'], temp['id'], temp['nttl'], temp['estr']