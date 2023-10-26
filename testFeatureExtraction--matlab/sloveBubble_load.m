function bubble_load = sloveBubble_load(DL,grayIm,m1,n1)
    bgm = (DL == 0);                     %�����Ե�ָ���
    bgm = bwmorph(bgm,'bridge');         %�������Ӳ���
    number_2 = length(bgm);
    %��ĭ͸���Ĳ���=�Ҷ�ֵ��С�����ؼ�ȥ�ָ��߲���
    number_l=length(find(grayIm<80));    %�趨��ֵ�Ҷ�С��80Ϊ�п� 
    transpantNum=number_l- number_2;
    if transpantNum<0
        transpantNum=0;
    end
    bubble_load=1-transpantNum/m1/n1;    %9������
end