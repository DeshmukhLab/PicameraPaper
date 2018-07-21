# -*- coding: utf-8 -*-
"""
@author: rajat

INPUT: ttl filenames, neuralynx event (.nev) filename and picam video frame timestamp filenames
OUTPUT: video frame timestamps and TTL pulse timestamps adjusted for temporal drifts

We use 3 second ON 2 second OFF 3 second ON 2 second OFF transition to mark the start of frames
followed by 1 second ON/OFF transitions which goes on till the end of the recording session

"""

import pandas as pd
import numpy as np
import glob
import neuralynxio as nlxio
import sys

#function to binary search for a given key in data 
def binarySearch(data,val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) / 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind

#function to calculate TTL transition time 3 second ON 2 second OFF transition for picam., returns 4 transition time 
def calcPiTTLStartTime(piTTLState, piTTLTimestamp):
    pi_syncPulseTime = []
    #inter TTL time diff for ttl timestamps
    piTTLTimeDiff = np.ediff1d(piTTLTimestamp)
    #3 second, TTL high pulse detection
    diff3_picam=abs(piTTLTimeDiff-3000000)
    #find the indices of high pulse event
    trigger3_picam=np.where(diff3_picam<10000)[0]
    #2 second, TTL low pulse detection
    diff2_picam=abs(piTTLTimeDiff-2000000)
    #find the indices of high pulse event
    trigger2_picam=np.where(diff2_picam<10000)[0]
    #detect the timestamps for each on and off transition
    #if else condition to ensure that 3 second ON, 2 second OFF, 3 second ON, 2 second OFF transition are used
    if trigger3_picam[0]+1==trigger2_picam[0]:
            if trigger2_picam[0]+1==trigger3_picam[1]:
                if trigger3_picam[1]+1==trigger2_picam[1]:
                    pi_syncPulseTime.append(piTTLTimestamp[trigger3_picam[0]])
                    pi_syncPulseTime.append(piTTLTimestamp[trigger2_picam[0]])
                    pi_syncPulseTime.append(piTTLTimestamp[trigger3_picam[1]])
                    pi_syncPulseTime.append(piTTLTimestamp[trigger2_picam[1]])
    #return the time for the sync pulse (3ON,2OFF transition)
    return np.array(pi_syncPulseTime)

#function to calculate TTL transition time 3 second ON 2 second OFF transition for nlx, returns 4 transition time 
def calcNlxTTLStartTime(nlxEventTimestamps,nlxEventID,nlxTTLState,nlxEventString):
    nlx_syncPulseTime = []
    #event id starts from 11 for Nlx, make it zero
    nlxEventID = nlxEventID - 11
    #confirm where 'TTL Input' string is in the event string
    nlxTTLeventFlag = np.array(['TTL Input' in nlxevent_str for nlxevent_str in nlxEventString])
    #get the timestamps for each nlx timestamps
    nlxTTLTimestamps = nlxEventTimestamps[np.where(nlxEventID==0) and np.where(nlxTTLeventFlag==True)]
    #inter TTL time diff for nlx ttl timestamps 
    nlxTTLTimeDiff = np.ediff1d(nlxTTLTimestamps)
    #3 second HIGH pulse detection
    diff3_nlx = abs(nlxTTLTimeDiff-3000000)
    #find the indices of high pulse event
    trigger3_nlx = np.where(diff3_nlx<10000)[0]
    #2 second, TTL low pulse detection
    diff2_nlx = abs(nlxTTLTimeDiff-2000000)
    #find the indices of high pulse event
    trigger2_nlx=np.where(diff2_nlx<10000)[0]
    #detect the timestamps for each on and off transition
    #if else condition to ensure that 3 second ON, 2 second OFF, 3 second ON, 2 second OFF transition are used
    if trigger3_nlx[0]+1==trigger2_nlx[0]:
            if trigger2_nlx[0]+1==trigger3_nlx[1]:
                if trigger3_nlx[1]+1==trigger2_nlx[1]:
                    nlx_syncPulseTime.append(nlxTTLTimestamps[trigger3_nlx[0]])
                    nlx_syncPulseTime.append(nlxTTLTimestamps[trigger2_nlx[0]])
                    nlx_syncPulseTime.append(nlxTTLTimestamps[trigger3_nlx[1]])
                    nlx_syncPulseTime.append(nlxTTLTimestamps[trigger2_nlx[1]])    

    return np.array(nlx_syncPulseTime)

#function to calculate offset between neuralynx and picamera
def offsetPiTimestamps(nlxSyncPulseTime,piSyncPulseTime,piTTLTimestamps,piTTLCamGPUTimestamps,piFrameTimestamps):
    #calculate all offsets between Pi GPU timestampswhen TTL start events occured and Nlx timestamps 
    #and get mean offset across all timestamps
    pi_nlx_offset = np.mean(nlxSyncPulseTime - piSyncPulseTime)
    #Find the offset between camera GPU timestamps i.e. frame timestamps (TimeStamps variance)
    #for first TTL event and corresponsing TTL timestamps 
    ind = binarySearch(piTTLCamGPUTimestamps,piTTLTimestamps[0])
    GPUTSoffset = piTTLCamGPUTimestamps[ind] - piTTLTimestamps[0]
    #update the frame timestamps to account for start time offset
    piFrameTimestamps = piFrameTimestamps + GPUTSoffset + pi_nlx_offset
    piTTLTimestamps = piTTLTimestamps + pi_nlx_offset
    return piTTLTimestamps, piFrameTimestamps

##get both frame and ttl timestamps only for the maze event
def getTimestampsForMaze(camTTL_ipstate,camTTL_ts,camFrame_ts,startMazeTime,endMazeTime):
    #find the start and end maze index for ttl time
    startMazeInd = binarySearch(camTTL_ts, startMazeTime)
    endMazeInd = binarySearch(camTTL_ts, endMazeTime)
    #select ttl times corresponding to start and end maze time
    camTTL_ts = camTTL_ts[startMazeInd:endMazeInd]
    camTTL_ipstate = camTTL_ipstate[startMazeInd:endMazeInd]
    #find the start and end maze index for frame time
    startMazeInd = binarySearch(camFrame_ts, startMazeTime)
    endMazeInd = binarySearch(camFrame_ts, endMazeTime)
    #select frame timestamps corresponding to start and end maze time
    camFrame_ts = camFrame_ts[startMazeInd:endMazeInd]
    return camTTL_ipstate, camTTL_ts, camFrame_ts, startMazeInd, endMazeInd

#function to adjust for temporal drifts
def getTemporalDriftTimes(camframe_ts,camttl_ts,nlx_ts):
    camframe_ts_ = np.copy(camframe_ts)
    #for each TTL on/off transition
    for i in range(1,len(nlx_ts)):
        #calculate the  offset between nlx ttl time and pi ttl time
        pi_nlx_ttl_offset = nlx_ts[i] - camttl_ts[i]
        #offset the future pi ttl times to account for drifts
        camttl_ts[i+1:] = camttl_ts[i+1:] + pi_nlx_ttl_offset
        #find the nearest frame timestamps for the current ttl transition
        ind = binarySearch(camframe_ts_, nlx_ts[i])
        #add the nlx and pi ttl offset to all the future frame timestamps
        camframe_ts_[ind:] = camframe_ts_[ind:] + pi_nlx_ttl_offset
        #adjust the frame timestmaps to account for the drifts wi
        camframe_ts_[:ind] = np.linspace(camframe_ts_[0],camframe_ts_[ind-1], len(camframe_ts_[:ind]))
    return camframe_ts_, camttl_ts

#function to calculate temporally adjusted timestamps 
def getTemporalAdjustedTimes(nlxEventFilename, timestampFiles, ttlFiles):
    #load Neuralynx Events.nev file to get TTL timestamps
    nlxEventData = nlxio.loadNev(nlxEventFilename)
    #load nlx event timestamps
    nlxEventTimestamps, nlxEventID, nlxTTL_ipstate, nlxEventString = nlxEventData[0], nlxEventData[1], nlxEventData[2], nlxEventData[3]
    #calculate sync pulse 3 second ON 2 second OFF x 2 time transitio, so 4 transition timestamps
    nlxSyncPulseTime = calcNlxTTLStartTime(nlxEventTimestamps, nlxEventID, nlxTTL_ipstate, nlxEventString)
    #calculate start and end maze epochs for neuralynx
    #This has to be changed as per user's need and maze 
    startMazeInd = np.where(np.array(['start maze 1' in nlxevent_str for nlxevent_str in nlxEventString])==True)[0][0]
    endMazeInd = np.where(np.array(['end maze 1' in nlxevent_str for nlxevent_str in nlxEventString])==True)[0][0]
    #select ttl times corresponding to start and end maze time
    nlxEventTimestamps_maze = nlxEventTimestamps[startMazeInd:endMazeInd]
    nlxTTL_ipstate_maze = nlxTTL_ipstate[startMazeInd:endMazeInd]
    nlxEventString_maze = nlxEventString[startMazeInd:endMazeInd]
    #get start maze and end maze time
    startMazeTime = nlxEventTimestamps[startMazeInd]
    endMazeTime = nlxEventTimestamps[endMazeInd]
    #confirm where 'TTL Input' string is in the event string
    nlxTTLeventFlag = np.array(['TTL Input' in nlxevent_str for nlxevent_str in nlxEventString_maze])
    #get the timestamps for each nlx timestamps and nlx TTL input state when TTL event occured 
    nlxEventTimestamps_maze = nlxEventTimestamps_maze[np.where(nlxTTLeventFlag==True)]
    nlxTTL_ipstate_maze = nlxTTL_ipstate_maze[np.where(nlxTTLeventFlag==True)]
    
    
    #ERROR! if the number of ttl file and frame timestamps are not same
    if len(ttlFiles)!=len(timestampFiles):
        sys.exit('Number of TTL file and Frame timestapmps files are not same')
    
    #load the ttl timestamps data for each camera
    cam1TTL_data = pd.read_csv(ttlFiles[0],usecols=[0,1,2])
    cam2TTL_data = pd.read_csv(ttlFiles[1],usecols=[0,1,2])
    cam3TTL_data = pd.read_csv(ttlFiles[2],usecols=[0,1,2])
    cam4TTL_data = pd.read_csv(ttlFiles[3],usecols=[0,1,2])
    cam5TTL_data = pd.read_csv(ttlFiles[4],usecols=[0,1,2])
    cam6TTL_data = pd.read_csv(ttlFiles[5],usecols=[0,1,2])
    cam7TTL_data = pd.read_csv(ttlFiles[6],usecols=[0,1,2])
    cam8TTL_data = pd.read_csv(ttlFiles[7],usecols=[0,1,2])
    
    #get TTL input state as well as TTL timestamps for each pulse transition
    cam1TTL_ipstate, cam1TTL_ts, cam1TTL_camts = np.array(cam1TTL_data['InputState']), np.array(cam1TTL_data['GPUTimestamp']), np.array(cam1TTL_data['CameraGPUTimestamp'])
    cam2TTL_ipstate, cam2TTL_ts, cam2TTL_camts = np.array(cam2TTL_data['InputState']), np.array(cam2TTL_data['GPUTimestamp']), np.array(cam2TTL_data['CameraGPUTimestamp'])
    cam3TTL_ipstate, cam3TTL_ts, cam3TTL_camts = np.array(cam3TTL_data['InputState']), np.array(cam3TTL_data['GPUTimestamp']), np.array(cam3TTL_data['CameraGPUTimestamp'])
    cam4TTL_ipstate, cam4TTL_ts, cam4TTL_camts = np.array(cam4TTL_data['InputState']), np.array(cam4TTL_data['GPUTimestamp']), np.array(cam4TTL_data['CameraGPUTimestamp'])
    cam5TTL_ipstate, cam5TTL_ts, cam5TTL_camts = np.array(cam5TTL_data['InputState']), np.array(cam5TTL_data['GPUTimestamp']), np.array(cam5TTL_data['CameraGPUTimestamp'])
    cam6TTL_ipstate, cam6TTL_ts, cam6TTL_camts = np.array(cam6TTL_data['InputState']), np.array(cam6TTL_data['GPUTimestamp']), np.array(cam6TTL_data['CameraGPUTimestamp'])
    cam7TTL_ipstate, cam7TTL_ts, cam7TTL_camts = np.array(cam7TTL_data['InputState']), np.array(cam7TTL_data['GPUTimestamp']), np.array(cam7TTL_data['CameraGPUTimestamp'])
    cam8TTL_ipstate, cam8TTL_ts, cam8TTL_camts = np.array(cam8TTL_data['InputState']), np.array(cam8TTL_data['GPUTimestamp']), np.array(cam8TTL_data['CameraGPUTimestamp'])
    
    #load the frame timestamps for each camera 
    cam1Frame_ts = np.array(pd.read_csv(timestampFiles[0],usecols=[1]))[:,0]
    cam2Frame_ts = np.array(pd.read_csv(timestampFiles[1],usecols=[1]))[:,0]
    cam3Frame_ts = np.array(pd.read_csv(timestampFiles[2],usecols=[1]))[:,0]
    cam4Frame_ts = np.array(pd.read_csv(timestampFiles[3],usecols=[1]))[:,0]
    cam5Frame_ts = np.array(pd.read_csv(timestampFiles[4],usecols=[1]))[:,0]
    cam6Frame_ts = np.array(pd.read_csv(timestampFiles[5],usecols=[1]))[:,0]
    cam7Frame_ts = np.array(pd.read_csv(timestampFiles[6],usecols=[1]))[:,0]
    cam8Frame_ts = np.array(pd.read_csv(timestampFiles[7],usecols=[1]))[:,0]
    
    #calculate the timestmaps for each on/off transition marking the start of frame
    #we have total 4 such transition
    cam1SyncPulseTime = calcPiTTLStartTime(cam1TTL_ipstate, cam1TTL_ts)
    cam2SyncPulseTime = calcPiTTLStartTime(cam2TTL_ipstate, cam2TTL_ts)
    cam3SyncPulseTime = calcPiTTLStartTime(cam3TTL_ipstate, cam3TTL_ts)
    cam4SyncPulseTime = calcPiTTLStartTime(cam4TTL_ipstate, cam4TTL_ts)
    cam5SyncPulseTime = calcPiTTLStartTime(cam5TTL_ipstate, cam5TTL_ts)
    cam6SyncPulseTime = calcPiTTLStartTime(cam6TTL_ipstate, cam6TTL_ts)
    cam7SyncPulseTime = calcPiTTLStartTime(cam7TTL_ipstate, cam7TTL_ts)
    cam8SyncPulseTime = calcPiTTLStartTime(cam8TTL_ipstate, cam8TTL_ts)
    
    #calculate all frame timestamps with offset correction
    cam1TTL_ts_offsetadj, cam1Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam1SyncPulseTime,cam1TTL_ts,cam1TTL_camts,cam1Frame_ts)
    cam2TTL_ts_offsetadj, cam2Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam2SyncPulseTime,cam2TTL_ts,cam2TTL_camts,cam2Frame_ts)
    cam3TTL_ts_offsetadj, cam3Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam3SyncPulseTime,cam3TTL_ts,cam3TTL_camts,cam3Frame_ts)
    cam4TTL_ts_offsetadj, cam4Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam4SyncPulseTime,cam4TTL_ts,cam4TTL_camts,cam4Frame_ts)
    cam5TTL_ts_offsetadj, cam5Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam5SyncPulseTime,cam5TTL_ts,cam5TTL_camts,cam5Frame_ts)
    cam6TTL_ts_offsetadj, cam6Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam6SyncPulseTime,cam6TTL_ts,cam6TTL_camts,cam6Frame_ts)
    cam7TTL_ts_offsetadj, cam7Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam7SyncPulseTime,cam7TTL_ts,cam7TTL_camts,cam7Frame_ts)
    cam8TTL_ts_offsetadj, cam8Frame_ts_offsetadj = offsetPiTimestamps(nlxSyncPulseTime,cam8SyncPulseTime,cam8TTL_ts,cam8TTL_camts,cam8Frame_ts)
    
    #find the TTL state, TTL timestmaps, frame timestmaps for the maze
    cam1TTL_ipstate_maze, cam1TTL_ts_maze, cam1Frame_ts_maze, cam1StartMazeInd, cam1EndMazeInd = getTimestampsForMaze(cam1TTL_ipstate,cam1TTL_ts_offsetadj,cam1Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam2TTL_ipstate_maze, cam2TTL_ts_maze, cam2Frame_ts_maze, cam2StartMazeInd, cam2EndMazeInd = getTimestampsForMaze(cam2TTL_ipstate,cam2TTL_ts_offsetadj,cam2Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam3TTL_ipstate_maze, cam3TTL_ts_maze, cam3Frame_ts_maze, cam3StartMazeInd, cam3EndMazeInd = getTimestampsForMaze(cam3TTL_ipstate,cam3TTL_ts_offsetadj,cam3Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam4TTL_ipstate_maze, cam4TTL_ts_maze, cam4Frame_ts_maze, cam4StartMazeInd, cam4EndMazeInd  = getTimestampsForMaze(cam4TTL_ipstate,cam4TTL_ts_offsetadj,cam4Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam5TTL_ipstate_maze, cam5TTL_ts_maze, cam5Frame_ts_maze, cam5StartMazeInd, cam5EndMazeInd  = getTimestampsForMaze(cam5TTL_ipstate,cam5TTL_ts_offsetadj,cam5Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam6TTL_ipstate_maze, cam6TTL_ts_maze, cam6Frame_ts_maze, cam6StartMazeInd, cam6EndMazeInd = getTimestampsForMaze(cam6TTL_ipstate,cam6TTL_ts_offsetadj,cam6Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam7TTL_ipstate_maze, cam7TTL_ts_maze, cam7Frame_ts_maze, cam7StartMazeInd, cam7EndMazeInd  = getTimestampsForMaze(cam7TTL_ipstate,cam7TTL_ts_offsetadj,cam7Frame_ts_offsetadj,startMazeTime,endMazeTime)
    cam8TTL_ipstate_maze, cam8TTL_ts_maze, cam8Frame_ts_maze, cam8StartMazeInd, cam8EndMazeInd  = getTimestampsForMaze(cam8TTL_ipstate,cam8TTL_ts_offsetadj,cam8Frame_ts_offsetadj,startMazeTime,endMazeTime)
    
    ##check if the number of frames timestamps and TTL timestamps match across camera 
    try:
        assert len(cam1Frame_ts_maze)==len(cam2Frame_ts_maze)==len(cam3Frame_ts_maze)==len(cam4Frame_ts_maze)==len(cam5Frame_ts_maze)==len(cam6Frame_ts_maze)==len(cam7Frame_ts_maze)==len(cam8Frame_ts_maze)
        assert len(cam1TTL_ts_maze)==len(cam2TTL_ts_maze)==len(cam3TTL_ts_maze)==len(cam4TTL_ts_maze)==len(cam5TTL_ts_maze)==len(cam6TTL_ts_maze)==len(cam7TTL_ts_maze)==len(cam8TTL_ts_maze)==len(nlxTTL_ipstate_maze)
    except AssertionError:
        sys.exit('ERROR!: number of frame timestamps or TTL timestmaps does not match across cameras')

    #adjust for the temporal drift between picam and nlx timestamps for each picamera
    cam1Frame_ts_driftadj,cam1TTL_ts_driftadj = getTemporalDriftTimes(cam1Frame_ts_maze,cam1TTL_ts_maze,nlxEventTimestamps_maze)
    cam2Frame_ts_driftadj,cam2TTL_ts_driftadj = getTemporalDriftTimes(cam2Frame_ts_maze,cam2TTL_ts_maze,nlxEventTimestamps_maze)
    cam3Frame_ts_driftadj,cam3TTL_ts_driftadj = getTemporalDriftTimes(cam3Frame_ts_maze,cam3TTL_ts_maze,nlxEventTimestamps_maze)
    cam4Frame_ts_driftadj,cam4TTL_ts_driftadj = getTemporalDriftTimes(cam4Frame_ts_maze,cam4TTL_ts_maze,nlxEventTimestamps_maze)
    cam5Frame_ts_driftadj,cam5TTL_ts_driftadj = getTemporalDriftTimes(cam5Frame_ts_maze,cam5TTL_ts_maze,nlxEventTimestamps_maze)
    cam6Frame_ts_driftadj,cam6TTL_ts_driftadj = getTemporalDriftTimes(cam6Frame_ts_maze,cam6TTL_ts_maze,nlxEventTimestamps_maze)
    cam7Frame_ts_driftadj,cam7TTL_ts_driftadj = getTemporalDriftTimes(cam7Frame_ts_maze,cam7TTL_ts_maze,nlxEventTimestamps_maze)
    cam8Frame_ts_driftadj,cam8TTL_ts_driftadj = getTemporalDriftTimes(cam8Frame_ts_maze,cam8TTL_ts_maze,nlxEventTimestamps_maze)
    
    #save and return the start and end maze index for each camera, adjusted frame timestamps, TTL timestamps for each camera 
    frame_ts_adj = [cam1Frame_ts_driftadj,cam2Frame_ts_driftadj,cam3Frame_ts_driftadj,cam4Frame_ts_driftadj,cam5Frame_ts_driftadj,
                    cam6Frame_ts_driftadj,cam7Frame_ts_driftadj,cam8Frame_ts_driftadj]
    ttl_ts_adj = [cam1TTL_ts_driftadj,cam2TTL_ts_driftadj,cam3TTL_ts_driftadj,cam4TTL_ts_driftadj,cam5TTL_ts_driftadj,
                  cam6TTL_ts_driftadj,cam7TTL_ts_driftadj,cam8TTL_ts_driftadj]
      
    return frame_ts_adj, ttl_ts_adj


#********MAIN CODE******************
def main():    
    #load the TTL and frame timestamps file"
    timestampFiles = glob.glob('cam*timestamp*.csv')
    ttlFiles = glob.glob('cam*ttl*.csv')
    #neuralynx file name
    nlxEventFilename = 'Events.nev' 
    
    #get the frame timestamps and TTL timestamps to account for the drift
    camFrame_ts_adjusted, camTTL_ts_adjusted = getTemporalAdjustedTimes(nlxEventFilename, timestampFiles, ttlFiles)
    #save the frame timestamps and ttl timestamps for each picamera to csv
    for i in range(len(timestampFiles)):
        df = pd.concat([pd.Series(camFrame_ts_adjusted[i]),pd.Series(camTTL_ts_adjusted[i])],1,keys=['FrameTimestamp','TTLTimestamp'])
        filename = timestampFiles[i].split('_')[0] + '_adjTime.csv'
        df.to_csv(filename, index=False)


main()        