function [im1,im4] = waveletDecomposition(grayIm,l,w,cfn,cl)
[C,S] = wavedec2(grayIm,l,w);          %进行3层小波分解
if cl~=0
    switch cfn
        case 1
            im1 = appcoef2(C,S,w,l);   % appcoef2 提取低频系数
        case 2
            im1 = detcoef2('h',C,S,cl);
        case 3
            im1 = detcoef2('v',C,S,cl);
        case 4
            im1 = detcoef2('d',C,S,cl);
    end
%指定提取那部分小波系数，a--近似图像 ;h--水平高频分量;v--垂直高频分量;d--对角高
%     im2 = detcoef2('h',C,S,l); 
%     im3 = detcoef2('v',C,S,l);
    im4 = detcoef2('d',C,S,l);        % detcoef2 对小波变换的高频部分系数进行提取
end

im1 = abs(int16(im1));
% im2 = abs(int16(im2));
% im3 = abs(int16(im3));
im4 = abs(int16(im4));
        