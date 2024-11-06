# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 01:13:30 2023

@author: RYU
"""

#.py file for reading .mat files

import torch
import numpy as np
import scipy.io
import h5py

# reading data
class MatReader():
    #.mat 파일을 읽기 위하여 Class 정의
    #reader = MatReader(.mat파일 경로) 등의 꼴로 변수 선언
    #.mat 파일을 읽는 module로는 2가지가 있음 (scipy.io.loadmat() 및 h5py.File())
    #보통 전자로 읽는편인데, v7.3이후로는 scipy.io.loadmat()가 못 읽어들이는 경우가 있음 (NotImplementedError: Please use HDF reader for matlab v7.3 files)
    #그럴 때는 h5py.File()을 통해 읽어야함 (즉, 비교적 최신 version의 mat파일은 h5py.File()을 통해서만 읽을 수 있음)
    
    #본 code에서는 우선 scipy.io.loadmat()으로 .mat파일을 읽는 것을 try하되,
    #Error 발생시 except구문 (예외처리구문)을 통해 h5py으로 파일을 읽음 (메소드 _load_file() 참고)
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()
    
    #.mat 파일을 불러오자 01
    #처음 init()코드에서 볼 수 있듯이, reader = MatReader(.mat파일 경로)으로 reader 변수를 생성하면,
    #이 method에 입각하여 reader변수에 .data 및 .old_mat 속성을 추가한다.
    def _load_file(self):
        #우선 scipy.io.loadmat()로 읽기 시도
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True #만약 잘 읽었다면, 이는 .mat파일의 version이 구버전이라는 뜻

        #만약 scipy.io.loadmat()로 못 읽어들인다면, h5py를 시도
        except:
            self.data = h5py.File(self.file_path, mode='r')
            self.old_mat = False #이 경우, .mat파일의 version이 신버전이라는 뜻
    
    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        #self.data : 통째로 불러온 .mat파일
        #self.data[field] : 'coeff', 'sol' 등, 우리가 불러오고자 하는 변수
        x = self.data[field]
        if not self.old_mat: #h5py로 읽는 경우
            x = x[()] #이 code를 통해 np_array로 변환이 된다.
            axes_range = range(len(x.shape) - 1, -1, -1) #[len(x.shape)-1, len(x.shape)-2, ... , 1, 0]
            x = np.transpose(x, axes=axes_range) #shape : From (a1, a2, ..., an) --> To (an, ..., a2, a1) 

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float