function [im1,im4] = waveletDecomposition(grayIm,l,w,cfn,cl)
[C,S] = wavedec2(grayIm,l,w);          %����3��С���ֽ�
if cl~=0
    switch cfn
        case 1
            im1 = appcoef2(C,S,w,l);   % appcoef2 ��ȡ��Ƶϵ��
        case 2
            im1 = detcoef2('h',C,S,cl);
        case 3
            im1 = detcoef2('v',C,S,cl);
        case 4
            im1 = detcoef2('d',C,S,cl);
    end
%ָ����ȡ�ǲ���С��ϵ����a--����ͼ�� ;h--ˮƽ��Ƶ����;v--��ֱ��Ƶ����;d--�ԽǸ�
%     im2 = detcoef2('h',C,S,l); 
%     im3 = detcoef2('v',C,S,l);
    im4 = detcoef2('d',C,S,l);        % detcoef2 ��С���任�ĸ�Ƶ����ϵ��������ȡ
end

im1 = abs(int16(im1));
% im2 = abs(int16(im2));
% im3 = abs(int16(im3));
im4 = abs(int16(im4));
        