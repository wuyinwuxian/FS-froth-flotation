function bubble_load = sloveBubble_load(DL,grayIm,m1,n1)
    bgm = (DL == 0);                     %求出边缘分割线
    bgm = bwmorph(bgm,'bridge');         %像素连接操作
    number_2 = length(bgm);
    %泡沫透明的部分=灰度值较小的像素减去分割线部分
    number_l=length(find(grayIm<80));    %设定阈值灰度小于80为有矿 
    transpantNum=number_l- number_2;
    if transpantNum<0
        transpantNum=0;
    end
    bubble_load=1-transpantNum/m1/n1;    %9承载率
end