function  DL  = Fen(im)
    KYS=7;  %��������ֵ
    K = imadjust(im,stretchlim(im));  %�Աȶ���ǿ
    Ds = imsubtract(K,10);            %����С��10�����㣬������Ե���
    se = strel('disk',KYS);           %����ṹԪ�أ����ڿ�����
    Ioc = imopen(Ds, se);             %������
    Iobr = imreconstruct(Ioc, Ds);    %���ع�
    Iobrd = imdilate(Iobr, se);       %���Ͳ���
    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));%���ͺ�ȡ���ع�
    Iobrcbr = imcomplement(Iobrcbr);  %ȡ������
    Iec = imcomplement(Iobrcbr);      %�˲���ȡ��
    DL = watershed(Iec);              %��ˮ��任
end