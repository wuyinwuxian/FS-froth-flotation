function bubble_size_mean = sloveBubble_size_mean(L1,m1,n1)
    numRegion = double(max(L1(:)));                %������ͨ����ĸ���
    bubble_size_mean =sqrt(m1*n1/numRegion).*0.3;  %2 ��ĭ��С��ֵ
end
