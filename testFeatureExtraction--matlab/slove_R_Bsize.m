function [radius,bubble_size] = slove_R_Bsize(L1)
    structArea1=regionprops(L1,'Area');  %����ÿ���������������
    area2=[structArea1.Area];            %�õ�����ĭ��С�ֲ�ͳ��
    area2 = area2/max(area2);            
    [bubble_size,~] = hist(area2,20);    %��ĭ��С
    radius = sqrt(area2).*0.3;           %�뾶
end