import sys
import os
import cv2
import numpy as np
import random

class Recognition:
    # current allowed numbers / letters tested: [ 0 - 9 ]
    _ALPHABET_ = 10
    
    # current number of samples per alphabet entry: 8
    _ALPHABET_EACH_ = 4
    
    # current allowed image size
    _SIZ_ = 16
    
    # colors to be weighted, grayscale for now
    _W_ = 255
    _G_ = 127
    _B_ = 0
    
    # weighting
    _HVY_ = 3 # if needed
    _MID_ = 2
    _LGT_ = 1
    _NON_ = 0
    
    def SETUP():
        ARR_IMG_N = np.zeros((Recognition._ALPHABET_, Recognition._SIZ_, Recognition._SIZ_))
        ARR_IMG_N_X = np.zeros((Recognition._ALPHABET_, Recognition._ALPHABET_EACH_, Recognition._SIZ_, Recognition._SIZ_))
        
        for i in range(Recognition._ALPHABET_):
            for j in range(Recognition._ALPHABET_EACH_):
                IMG_N = cv2.imread(f'resources/N-{i}.png') # solo original sample for each number
                #IMG_N = cv2.imread(f'resources/N-{i}-{chr(ord('A') + j)}.png')
                if len(IMG_N.shape) == 3:
                    IMG_N = cv2.cvtColor(IMG_N, cv2.COLOR_BGR2GRAY)
                IMG_N = IMG_N.astype(np.float32)
                
                HGT, WID = IMG_N.shape
                
                for ROW in range(HGT):
                    for COL in range(WID):
                        PXL = IMG_N[ROW, COL].copy()
                        ARR_IMG_N[i][ROW][COL] = (ARR_IMG_N[i][ROW][COL] + PXL) / 2
                        if ARR_IMG_N[i][ROW][COL] > 239: # avg white -> full white ... better weighting
                            ARR_IMG_N[i][ROW][COL] = Recognition._W_
            #print(ARR_IMG_N[i]) #
            print(f'---[ END: {i} ]---') #
        print('---[ END: SETUP ]---') #
        
        return ARR_IMG_N
    
    def INPUT(IMG_PATH):
        ARR_IMG_N = Recognition.SETUP()
        ARR_SCR_N_CNT = np.zeros(Recognition._ALPHABET_)
        
        ARR_IMG = np.zeros((Recognition._SIZ_, Recognition._SIZ_))
        IMG = cv2.imread(IMG_PATH)
        if len(IMG.shape) == 3:
            IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        IMG = IMG.astype(np.float32)
        
        TOP, BOT, LEF, RIG = 0, IMG.shape[0], 0, IMG.shape[1]
        while TOP < BOT and np.all(IMG[TOP] == Recognition._W_):
            TOP += 1
        while BOT > TOP and np.all(IMG[BOT - 1] == Recognition._W_):
            BOT -= 1
        while LEF < RIG and np.all(IMG[:, LEF] == Recognition._W_):
            LEF += 1
        while RIG > LEF and np.all(IMG[:, RIG - 1] == Recognition._W_):
            RIG -= 1
        if TOP > 0:
            TOP -= 1
        if BOT < IMG.shape[0]:
            BOT += 1
        if LEF > 0:
            LEF -= 1
        if RIG < IMG.shape[0]:
            RIG += 1
        IMG_EDIT = IMG[TOP:BOT, LEF:RIG]
        
        cv2.imwrite(f'resources\\extra\\MAIN.png', IMG_EDIT.astype(np.uint8)) ######
        
        HGT, WID = IMG_EDIT.shape
        print(f'[TRIM] H:{HGT}/W:{WID} (MAIN)') #
        
        ARR_IMG_N_EDIT = np.zeros(Recognition._ALPHABET_, dtype=object)
        for i in range(Recognition._ALPHABET_):
            ARR_IMG_N_EDIT[i] = cv2.resize(ARR_IMG_N[i], (WID, HGT), interpolation=cv2.INTER_CUBIC)
            ARR_IMG_N_EDIT[i] = ((ARR_IMG_N_EDIT[i] - 127) * 1.5) + 127 # contrast
            ARR_IMG_N_EDIT[i] = np.clip(ARR_IMG_N_EDIT[i], 0, 255)
            ARR_IMG_N_EDIT[i][ARR_IMG_N_EDIT[i] > 144] = 255
            
            #cv2.imwrite(f'resources\\extra\\{i}-PRE.png', ARR_IMG_N_EDIT[i].astype(np.uint8)) ######
            
            TOP, BOT, LEF, RIG = 0, ARR_IMG_N[i].shape[0], 0, ARR_IMG_N[i].shape[1]
            while TOP < BOT and np.all(ARR_IMG_N[i][TOP] > 180):
                TOP += 1
            while BOT > TOP and np.all(ARR_IMG_N[i][BOT - 1] > 180):
                BOT -= 1
            while LEF < RIG and np.all(ARR_IMG_N[i][:, LEF] > 180):
                LEF += 1
            while RIG > LEF and np.all(ARR_IMG_N[i][:, RIG - 1] > 180):
                RIG -= 1
            ARR_IMG_N_EDIT[i] = ARR_IMG_N[i][TOP:BOT, LEF:RIG]
            
            cv2.imwrite(f'resources\\extra\\{i}-PRE.png', ARR_IMG_N_EDIT[i].astype(np.uint8)) ######
            
            ARR_IMG_N_EDIT[i] = cv2.resize(ARR_IMG_N_EDIT[i], (WID, HGT), interpolation=cv2.INTER_CUBIC)
            ARR_IMG_N_EDIT[i][ARR_IMG_N_EDIT[i] <= 127] = 0
            #ARR_IMG_N_EDIT[i][(ARR_IMG_N_EDIT[i] > 96) & (ARR_IMG_N_EDIT[i] < 160)] = 127
            ARR_IMG_N_EDIT[i][ARR_IMG_N_EDIT[i] > 127] = 255
            
            cv2.imwrite(f'resources\\extra\\{i}-PST.png', ARR_IMG_N_EDIT[i].astype(np.uint8)) ######
        
        for ROW in range(HGT):
            for COL in range(WID):
                PXL = IMG_EDIT[ROW, COL].copy()
                ARR_IMG[ROW][COL] = Recognition._LGT_
                for i in range(Recognition._ALPHABET_):
                    if PXL < Recognition._W_:
                        if ARR_IMG_N_EDIT[i][ROW][COL] == Recognition._B_:
                            ARR_SCR_N_CNT[i] += 1.00
                        if ARR_IMG_N_EDIT[i][ROW][COL] == Recognition._G_:
                            ARR_SCR_N_CNT[i] += 0.50
                    else:
                        if ARR_IMG_N_EDIT[i][ROW][COL] != Recognition._W_:
                            ARR_SCR_N_CNT[i] -= 0.75
                    print(ARR_SCR_N_CNT) #
        #print(ARR_IMG) #
        print('---[ END: RECOGNITION ]---')
        
        #for i in range(Recognition._ALPHABET_):
            #print(f'{i}: {ARR_SCR_N_CNT[i]}') #
        print(f'\nI think your number is <{np.argmax(ARR_SCR_N_CNT)}> !') #

if __name__ == '__main__':
    np.set_printoptions(linewidth=6*16)
    Recognition.INPUT('N-SAMPLE.png')