function bubble_size_mean = sloveBubble_size_mean(L1,m1,n1)
    numRegion = double(max(L1(:)));                %计算连通区域的个数
    bubble_size_mean =sqrt(m1*n1/numRegion).*0.3;  %2 泡沫大小均值
end
