function scale =sloveScale(L1)
    structChangzhou1 = regionprops(L1,'MajorAxisLength'); %����ĳ���
    changZhou = [structChangzhou1.MajorAxisLength];
    structDuanzhou1 = regionprops(L1,'MinorAxisLength');  %����Ķ���
    duanZhou = [structDuanzhou1.MinorAxisLength];
    scale = mean(changZhou./duanZhou);                    %8�����
end