close all
clc 
clear all

strPath='D:\study\工况识别\2014-07-04-锑粗选工况分类\';

%%
%获取子文件夹
dirList = dir(strPath);  
isSubDir = [dirList(:).isdir];    %# returns logical vector  
nameFolds = {dirList(isSubDir).name}';  
nameFolds(ismember(nameFolds,{'.','..'})) = []; 
nSubDir=length(nameFolds);

fileList=dir(fullfile(strcat(strPath,nameFolds{1},'\*.avi')));  %取第一个文件夹
nFile=length(fileList); 

%取第一个文件夹里面的第一个视频
readerObj = VideoReader(strcat(strPath,nameFolds{1},'\',fileList(1).name));
vidFrames = read(readerObj);  %读取所有帧
numFrames = get(readerObj, 'NumberOfFrames'); % 帧数

%%
%纹理相关参数，获取灰度频次表的参数
%在getgrayfrequency 、getgraymatrix中用到
d = 1;      %邻域半径

%获取灰度频次表的参数  仅在 getgrayfrequency 中用到
a = 0;      %与中心像素点灰度差值 
t = 1;      %与中心像素点灰度比较类型(1表示:<=a;2表示:==a) 

%小波分解的参数，仅在 waveletDecomposition 中用到
w = 'sym4'; %小波基类型
cfn = 1;    %coef2系数矩阵类型：1:ca 2:ch 3:cv 4:cd
cl = 1;     %系数所在分解层次

%%
for  nf=1:1
%         eval(['im =',str_avi,'(:,:,nf*3-2:nf*3);' ]);  %RGB三通道提取
       im = vidFrames(:,:,:,nf);   %大小600*800*3，每个像素点的RGB取值
       im1 = im(:,:,1);
       im2 = im(:,:,2);
       im3 = im(:,:,3);
%%%%%%%%%%%%% 代码主体 %%%%%%%%%%%%%%%%%%%%%%%  
        nowFrame = im; %当前帧，一帧图片600*800*3 
        if nf ~= 1
            velocity = solveVelocity(nowFrame,preFrame);  %11 速度
        else 
            velocity = 0;
        end
        result = getFeatures(nowFrame,velocity,d,a,t,w,cfn,cl);
        preFrame = nowFrame;
%%%%%%%%%%%%% 代码主体 %%%%%%%%%%%%%%%%%%%%%%%        
        XX(nf,:) = result;
end
X = XX;
