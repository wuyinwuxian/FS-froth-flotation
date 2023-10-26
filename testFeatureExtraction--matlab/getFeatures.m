function result = getFeatures(nowFrame,velocity,d,a,t,w,cfn,cl)
    [m1,n1,l] = size(nowFrame);                  %��С600*800*3��l��С���ֽ��Σ�Ҳ��ͨ������
    grayIm = int16(rgb2gray(nowFrame));          %ΪʲôҪ���16λ���ͣ���

    grayMean = sloveGrayMean(grayIm,m1,n1);      %1  �ҶȾ�ֵ
    hueMean = sloveHueMean(nowFrame,m1,n1);      %3  ɫ����ֵ  ��Χ0~1 
    rim1 = sum(sum(nowFrame(:,:,1)))/m1/n1;      %5  R��ֵ
    gim1 = sum(sum(nowFrame(:,:,2)))/m1/n1;      %6  G��ֵ
    bim1 = sum(sum(nowFrame(:,:,3)))/m1/n1;      %7  B��ֵ
    R_per = rim1/grayMean;                       %10 ��Ժ�ɫ����

    %С���ֽ��ĵ�Ƶ����Ƶ���ԽǸߣ�ϵ��
    [im1,im4] = waveletDecomposition(grayIm,l,w,cfn,cl);  
    graylevel1 = int16(max(max(im1)))+1;  
    graylevel4 = int16(max(max(im4)))+1; 

    coarseness = sloveCoarseness(grayIm,graylevel1,d,a,t);            %14 ��ĭ�ֶ�
    nonUnifiormaity = sloveNonUnifiormaity(grayIm,graylevel1,d,a,t);  %15 ��ֵ�Ǿ��ȶ�
    secondMoment = sloveSecondMoment(grayIm,graylevel1,d,a,t);        %16 ���׾�
    LF_Coarseness = sloveCoarseness(im1,graylevel1,d,a,t);            %17 ��Ƶ��ͼ�ֶ�
    HF_energy = sloveHF_energy(im4,graylevel4,d,a,t);                 %18 ��Ƶ����

    DL = Fen(grayIm);                    %��ˮ��
    L1 = bwlabel(DL); % ����DL����ֵͼ���м�����ͨ����Ĭ�ϰ�8��ͨ�ң�Ȼ����1...n�����������

    bubble_load = sloveBubble_load(DL,grayIm,m1,n1);    %9  ������
    scale =sloveScale(L1);                              %8  �����
    bubble_size_mean = sloveBubble_size_mean(L1,m1,n1); %2  ��ĭ��С��ֵ
    [radius,bubble_size] = slove_R_Bsize(L1);
    bubble_size_std = std(radius);                      %4  ��С�ֲ�����
    bubble_kurtois = kurtosis(radius);                  %12 ���Ͷ�
    bubble_sknew = skewness(radius);                    %13 ƫб��
    
    result = [grayMean,bubble_size_mean,hueMean,bubble_size_std,rim1,gim1,bim1,...%'1�Ҷ�','2��ĭƽ����С','3ɫ��','4��С��׼��','5��ɫ��ֵ','6��ɫ��ֵ','7��ɫ��ֵ'
                   scale,bubble_load,R_per,velocity,bubble_kurtois,bubble_sknew,...%,'8�����','9������','10��Ժ�ɫ����','11�ٶ�','12���Ͷ�','13ƫб��'
                   coarseness,nonUnifiormaity,secondMoment,LF_Coarseness,HF_energy,...%,'14�ֶ�','15�Ǿ��ȶ�','16���׾�','17��Ƶ��ͼ�ֶ�','18��Ƶ����'
                   bubble_size];%��ĭ��С
end