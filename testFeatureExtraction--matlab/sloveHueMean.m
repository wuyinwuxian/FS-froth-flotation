function hue1 = sloveHueMean(Frame_1,m1,n1)          
    hsv_im=rgb2hsv(Frame_1);                %色调
    hue1= sum(sum(hsv_im(:,:,2)))/m1/n1;    %3 色调均值  范围0~1
end