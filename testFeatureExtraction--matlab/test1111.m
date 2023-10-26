close all
clc 
clear all

strPath='D:\study\����ʶ��\2014-07-04-���ѡ��������\';

%%
%��ȡ���ļ���
dirList = dir(strPath);  
isSubDir = [dirList(:).isdir];    %# returns logical vector  
nameFolds = {dirList(isSubDir).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 
nSubDir=length(nameFolds);

fileList=dir(fullfile(strcat(strPath,nameFolds{1},'\*.avi')));  %ȡ��һ���ļ���
nFile=length(fileList); 

%ȡ��һ���ļ�������ĵ�һ����Ƶ
readerObj = VideoReader(strcat(strPath,nameFolds{1},'\',fileList(1).name));
vidFrames = read(readerObj);  %��ȡ����֡
numFrames = get(readerObj, 'NumberOfFrames'); % ֡��

%%
%������ز�������ȡ�Ҷ�Ƶ�α�Ĳ���
%��getgrayfrequency ��getgraymatrix���õ�
d = 1;      %����뾶

%��ȡ�Ҷ�Ƶ�α�Ĳ���  ���� getgrayfrequency ���õ�
a = 0;      %���������ص�ҶȲ�ֵ 
t = 1;      %���������ص�ҶȱȽ�����(1��ʾ:<=a;2��ʾ:==a) 

%С���ֽ�Ĳ��������� waveletDecomposition ���õ�
w = 'sym4'; %С��������
cfn = 1;    %coef2ϵ���������ͣ�1:ca 2:ch 3:cv 4:cd
cl = 1;     %ϵ�����ڷֽ���

%%
for  nf=1:1
%         eval(['im =',str_avi,'(:,:,nf*3-2:nf*3);' ]);  %RGB��ͨ����ȡ
       im = vidFrames(:,:,:,nf);   %��С600*800*3��ÿ�����ص��RGBȡֵ
       im1 = im(:,:,1);
       im2 = im(:,:,2);
       im3 = im(:,:,3);
%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%  
        nowFrame = im; %��ǰ֡��һ֡ͼƬ600*800*3 
        if nf ~= 1
            velocity = solveVelocity(nowFrame,preFrame);  %11 �ٶ�
        else 
            velocity = 0;
        end
        result = getFeatures(nowFrame,velocity,d,a,t,w,cfn,cl);
        preFrame = nowFrame;
%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%        
        XX(nf,:) = result;
end
X = XX;
