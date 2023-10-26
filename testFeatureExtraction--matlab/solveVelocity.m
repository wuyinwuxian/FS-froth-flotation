function velocity = solveVelocity(nowFrame,preFrame)
    %1����֡ͼ����ƥ�����
    centralIm1 = nowFrame(301:400,401:500);   %��ɫͨ���� 
    %��ǰ֡��ĭͼ�����ĺ��Ҷ�ֵ�����ۼ����
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
    %�ٶ���������ȡ
    velocity=sqrt(minDx.^2+minDy.^2); %11�ٶ�����
end