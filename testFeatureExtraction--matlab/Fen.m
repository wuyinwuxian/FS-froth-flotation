function  DL  = Fen(im)
    KYS=7;  %开运算阈值
    K = imadjust(im,stretchlim(im));  %对比度增强
    Ds = imsubtract(K,10);            %像素小于10的置零，即将边缘加深，
    se = strel('disk',KYS);           %构造结构元素，用于开运算
    Ioc = imopen(Ds, se);             %开运算
    Iobr = imreconstruct(Ioc, Ds);    %开重构
    Iobrd = imdilate(Iobr, se);       %膨胀操作
    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));%膨胀后取反重构
    Iobrcbr = imcomplement(Iobrcbr);  %取反操作
    Iec = imcomplement(Iobrcbr);      %滤波后取反
    DL = watershed(Iec);              %分水岭变换
end