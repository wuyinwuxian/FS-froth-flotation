close all
clc 
clear all

%% ��������
str_avi = 'X';
numData = '16';
strL = ['VedioData',numData];
strS = ['����4\FEMyData',numData];
load(strL);

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
eval([str_avi,'=uint8(',str_avi,');']);
eval(['[~,~,numFrames] = size(',str_avi,');']);
numFrames = numFrames/3;


for  nf=1:2
        eval(['im =',str_avi,'(:,:,nf*3-2:nf*3);' ]);  %RGB��ͨ����ȡ
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

