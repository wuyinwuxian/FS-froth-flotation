function [radius,bubble_size] = slove_R_Bsize(L1)
    structArea1=regionprops(L1,'Area');  %计算每个区域的像素总数
    area2=[structArea1.Area];            %得到的泡沫大小分布统计
    area2 = area2/max(area2);            
    [bubble_size,~] = hist(area2,20);    %泡沫大小
    radius = sqrt(area2).*0.3;           %半径
end