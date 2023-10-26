close all
clc 
clear all

%% 加载数据
str_avi = 'X';
numData = '16';
strL = ['VedioData',numData];
strS = ['样本4\FEMyData',numData];
load(strL);

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
eval([str_avi,'=uint8(',str_avi,');']);
eval(['[~,~,numFrames] = size(',str_avi,');']);
numFrames = numFrames/3;


for  nf=1:2
        eval(['im =',str_avi,'(:,:,nf*3-2:nf*3);' ]);  %RGB三通道提取
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

