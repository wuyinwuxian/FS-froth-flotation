function scale =sloveScale(L1)
    structChangzhou1 = regionprops(L1,'MajorAxisLength'); %区域的长轴
    changZhou = [structChangzhou1.MajorAxisLength];
    structDuanzhou1 = regionprops(L1,'MinorAxisLength');  %区域的短轴
    duanZhou = [structDuanzhou1.MinorAxisLength];
    scale = mean(changZhou./duanZhou);                    %8长宽比
end