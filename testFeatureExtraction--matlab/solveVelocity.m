function velocity = solveVelocity(nowFrame,preFrame)
    %1上下帧图像宏块匹配过程
    centralIm1 = nowFrame(301:400,401:500);   %红色通道的 
    %对前帧泡沫图像中心宏块灰度值行列累计求和
    sumRowCol1=[sum(centralIm1,1),sum(centralIm1,2)'];
    sumDiff=zeros(101,101);
    minDiff=inf;
    minDy=0;
    minDx=0;
    for dy=-100:100
        for dx=-150:150
            centralIm2=preFrame(301+dy:400+dy,401+dx:500+dx);
            sumRowCol2=[sum(centralIm2,1),sum(centralIm2,2)'];
            dSum=sum(abs(sumRowCol1-sumRowCol2));
            sumDiff(dy+101,dx+151)=dSum;
            if dSum<minDiff
                minDiff=dSum;
                minDy=dy;
                minDx=dx;
            end 
        end
    end
    %速度特征的求取
    velocity=sqrt(minDx.^2+minDy.^2); %11速度特征
end