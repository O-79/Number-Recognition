import sys
import os
import cv2
import numpy as np
import random

class Recognition:
    # current allowed numbers / letters tested: [ 0 - 9 ]
    _ALPHABET_ = 10
    
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
        for i in range(Recognition._ALPHABET_):
            IMG_N = cv2.imread(f'resources/N-{i}.png').astype(np.int16)
            HGT, WID, CLR = IMG_N.shape
            for ROW in range(HGT):
                for COL in range(WID):
                    PXL = IMG_N[ROW, COL].copy()
                    B = PXL[0]
                    G = PXL[1]
                    R = PXL[2]
                    if (B, G, R) == (Recognition._B_, Recognition._B_, Recognition._B_):
                        ARR_IMG_N[i][ROW][COL] = Recognition._MID_;
            # print(ARR_IMG_N[i])
        # print('---')
        return ARR_IMG_N
    
    def INPUT(IMG_PATH):
        ARR_IMG_N = Recognition.SETUP()
        ARR_SCR_N_CNT = np.zeros(Recognition._ALPHABET_)
        
        ARR_IMG = np.zeros((Recognition._SIZ_, Recognition._SIZ_))
        IMG = cv2.imread(IMG_PATH).astype(np.int16)
        HGT, WID, CLR = IMG.shape
        for ROW in range(HGT):
            for COL in range(WID):
                PXL = IMG[ROW, COL].copy()
                B = PXL[0]
                G = PXL[1]
                R = PXL[2]
                if (B, G, R) == (Recognition._B_, Recognition._B_, Recognition._B_):
                    ARR_IMG[ROW][COL] = Recognition._MID_;
                    for i in range(Recognition._ALPHABET_):
                        if ARR_IMG[ROW][COL] - ARR_IMG_N[i][ROW][COL] == 0:
                            ARR_SCR_N_CNT[i] += 1
        # print(ARR_IMG)
        # print('---')
        # for i in range(Recognition._ALPHABET_):
            # print(f'{i}: {ARR_SCR_N_CNT[i]}')
        
        print(np.argmax(ARR_SCR_N_CNT))

if __name__ == '__main__':
    Recognition.INPUT('N-SAMPLE.png')