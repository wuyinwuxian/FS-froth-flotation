function result = getFeatures(nowFrame,velocity,d,a,t,w,cfn,cl)
    [m1,n1,l] = size(nowFrame);                  %大小600*800*3，l是小波分解层次（也是通道数）
    grayIm = int16(rgb2gray(nowFrame));          %为什么要变成16位整型？？

    grayMean = sloveGrayMean(grayIm,m1,n1);      %1  灰度均值
    hueMean = sloveHueMean(nowFrame,m1,n1);      %3  色调均值  范围0~1 
    rim1 = sum(sum(nowFrame(:,:,1)))/m1/n1;      %5  R均值
    gim1 = sum(sum(nowFrame(:,:,2)))/m1/n1;      %6  G均值
    bim1 = sum(sum(nowFrame(:,:,3)))/m1/n1;      %7  B均值
    R_per = rim1/grayMean;                       %10 相对红色分量

    %小波分解后的低频、高频（对角高）系数
    [im1,im4] = waveletDecomposition(grayIm,l,w,cfn,cl);  
    graylevel1 = int16(max(max(im1)))+1;  
    graylevel4 = int16(max(max(im4)))+1; 

    coarseness = sloveCoarseness(grayIm,graylevel1,d,a,t);            %14 泡沫粗度
    nonUnifiormaity = sloveNonUnifiormaity(grayIm,graylevel1,d,a,t);  %15 数值非均匀度
    secondMoment = sloveSecondMoment(grayIm,graylevel1,d,a,t);        %16 二阶矩
    LF_Coarseness = sloveCoarseness(im1,graylevel1,d,a,t);            %17 低频子图粗度
    HF_energy = sloveHF_energy(im4,graylevel4,d,a,t);                 %18 高频能量

    DL = Fen(grayIm);                    %分水岭
    L1 = bwlabel(DL); % 计算DL（二值图像）有几个连通区域，默认按8连通找，然后编号1...n（区域个数）

    bubble_load = sloveBubble_load(DL,grayIm,m1,n1);    %9  承载率
    scale =sloveScale(L1);                              %8  长宽比
    bubble_size_mean = sloveBubble_size_mean(L1,m1,n1); %2  泡沫大小均值
    [radius,bubble_size] = slove_R_Bsize(L1);
    bubble_size_std = std(radius);                      %4  大小分布方差
    bubble_kurtois = kurtosis(radius);                  %12 陡峭度
    bubble_sknew = skewness(radius);                    %13 偏斜度
    
    result = [grayMean,bubble_size_mean,hueMean,bubble_size_std,rim1,gim1,bim1,...%'1灰度','2泡沫平均大小','3色调','4大小标准差','5红色均值','6绿色均值','7蓝色均值'
                   scale,bubble_load,R_per,velocity,bubble_kurtois,bubble_sknew,...%,'8长宽比','9承载率','10相对红色分量','11速度','12陡峭度','13偏斜度'
                   coarseness,nonUnifiormaity,secondMoment,LF_Coarseness,HF_energy,...%,'14粗度','15非均匀度','16二阶矩','17低频子图粗度','18高频能量'
                   bubble_size];%泡沫大小
end