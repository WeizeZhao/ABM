clear all
%Set random seed.
rng(71849)
%Set Parameter values:
%number of agent in the system.
N=1000;
%Number of in-group and out-group samples.
K=10;
%Gamma value in the utility function (see appendix. for explanation, default value is 10. 
gamma=10;
%Number of iteration.
R=20000;
%Value of w_AUT, the motivation strength of maintaining private attitudes.
w1=0.2;
%Value of w', default value is 0.5.
wgi2=0.5;
w2=(1-w1)*wgi2;
w3=(1-w1)*(1-wgi2);
%Value of m, determining the priority of each dimension in group.
%classification.
GSweight=[0.5 0.5];
%Proportation of stubborn agent in the system, default value is 0.
fixprop=0;
%Stubborn agents' constant expressed attitude values.
fixaut2=0.8;
fixagent=randperm(N, fixprop*N);
%The sum of two parameters in the beta distribution describing authentic
%preference (see Appendix.). Default value is 10 for both dimensions.
sumage1=10;
sumage2=10;

%Generate a uniformly distributed private attitudes (also initially
%expressed attitudes) on both dimensions.
alpha1=1.001;
beta1=10-alpha1;
limit1=betainv(0.5,alpha1,beta1);
limit2=1-limit1;
%Correlation matrix for private attitudes on both dimensions, default to be
%[1 0; 0 1], i.e. two independent issues.
rho1=[1 0;
    0 1 ];
Z1 = mvnrnd([0 0], rho1, N);
U1 = normcdf(Z1,0,1);
X1 = [unifinv(U1(:,1),limit1,limit2) unifinv(U1(:,2),limit1,limit2)];

int1=X1(:,1);
int2=X1(:,2);
int2(fixagent)=fixaut2;

agealpha1=zeros(N,1);
agebeta1=zeros(N,1);
agealpha2=zeros(N,1);
agebeta2=zeros(N,1);

%Generate each agent's authentic preference distribution.
for t=1:1:N
    fun=@(x)abs(betainv(0.5,x,sumage1-x)-int1(t));
    x0=0.1;
    x=fminsearchbnd(fun,x0,1.001,sumage1-1.001);
    agealpha1(t)=x;
    agebeta1(t)=sumage1-x;
end
for t=1:1:N
    fun=@(x)abs(betainv(0.5,x,sumage2-x)-int2(t));
    x0=0.1;
    x=fminsearchbnd(fun,x0,1.001,sumage2-1.001);
    agealpha2(t)=x;
    agebeta2(t)=sumage2-x;
end
%In the initial system, each agent's expressed attitudes equal to their
%underlying private attitudes on both dimensions.
choice1=int1;
choice2=int2;

%To avoid social norm distributions become too sharp or flat, we set two
%higher and lower thresholds for the sum of two parameters in the beta
%distributions describing social norms. Default values are 20 and 3. 
UL=20;
LL=3;

%These two N*R matrices record the dynamics of all agents' expressed attitudes on both
%dimensions.
for aa=1:1:R
    allchoice1(:,aa)=int1;
end
for aa=1:1:R
    allchoice2(:,aa)=int2;
end

%The function calculating disutility value from gap between expressed attitude and distribution median. 
function disutility=utilitya(x,alpha1,beta1,gamma)
disutility=exp(abs(gamma*(betacdf(x,alpha1,beta1)-0.5)));
end


round=0;
while round<R
    round=round+1;
    %Randomly select an activated agent for each iteration.
    i=ceil(rand*N);
    %Determine each agent' identitie on each dimension based on whether
    %their expressed attitude is smaller or larger than .5.
    leftgroup1=find(choice1<0.5);
    rightgroup1=find(choice1>=0.5);
    leftgroup2=find(choice2<0.5);
    rightgroup2=find(choice2>=0.5);

    samplesized1=ceil(GSweight(1)*K);
    samplesized2=K-samplesized1;
    
    %In the example of stubborn agents given in the Result section, all agent
    %converge to a non-neutral consensus, so that the number of agent with
    %opposite identity on that dimension equals 0, and there will be no
    %out-group influence.
    if length(rightgroup2)>=samplesized2 && length(leftgroup2)>=samplesized2
        samplsized2=samplesized2;
    else
        samplesized2=0;
    end

    %Selection of in-group and out-group samples based on the activated
    %agent's identity on each dimension.
    if ismember(i,leftgroup1)==1 && ismember(i,leftgroup2)==1
        ingroupsamplelocation1=randperm(length(leftgroup1),samplesized1);
        ingroupsamplelocation2=randperm(length(leftgroup2),samplesized2);
        ingroupsample=unique([leftgroup1(ingroupsamplelocation1);leftgroup2(ingroupsamplelocation2)]);
        outgroupsamplelocation1=randperm(length(rightgroup1),samplesized1);
        outgroupsamplelocation2=randperm(length(rightgroup2),samplesized2);
        outgroupsample=unique([rightgroup1(outgroupsamplelocation1);rightgroup2(outgroupsamplelocation2)]);
    end
    if ismember(i,leftgroup1)==1 && ismember(i,rightgroup2)==1
        ingroupsamplelocation1=randperm(length(leftgroup1),samplesized1);
        ingroupsamplelocation2=randperm(length(rightgroup2),samplesized2);
        ingroupsample=unique([leftgroup1(ingroupsamplelocation1);rightgroup2(ingroupsamplelocation2)]);
        outgroupsamplelocation1=randperm(length(rightgroup1),samplesized1);
        outgroupsamplelocation2=randperm(length(leftgroup2),samplesized2);
        outgroupsample=unique([rightgroup1(outgroupsamplelocation1);leftgroup2(outgroupsamplelocation2)]);
    end
    if ismember(i,rightgroup1)==1 && ismember(i,leftgroup2)==1
        ingroupsamplelocation1=randperm(length(rightgroup1),samplesized1);
        ingroupsamplelocation2=randperm(length(leftgroup2),samplesized2);
        ingroupsample=unique([rightgroup1(ingroupsamplelocation1);leftgroup2(ingroupsamplelocation2)]);
        outgroupsamplelocation1=randperm(length(leftgroup1),samplesized1);
        outgroupsamplelocation2=randperm(length(rightgroup2),samplesized2);
        outgroupsample=unique([leftgroup1(outgroupsamplelocation1);rightgroup2(outgroupsamplelocation2)]);
    end
    if ismember(i,rightgroup1)==1 && ismember(i,rightgroup2)==1
        ingroupsamplelocation1=randperm(length(rightgroup1),samplesized1);
        ingroupsamplelocation2=randperm(length(rightgroup2),samplesized2);
        ingroupsample=unique([rightgroup1(ingroupsamplelocation1);rightgroup2(ingroupsamplelocation2)]);
        outgroupsamplelocation1=randperm(length(leftgroup1),samplesized1);
        outgroupsamplelocation2=randperm(length(leftgroup2),samplesized2);
        outgroupsample=unique([leftgroup1(outgroupsamplelocation1);leftgroup2(outgroupsamplelocation2)]);
    end
            
    ingroupsamplechoice1=choice1(ingroupsample);
    ingroupsamplechoice2=choice2(ingroupsample);
    outgroupsamplechoice1=choice1(outgroupsample);
    outgroupsamplechoice2=choice2(outgroupsample);

    %Calculate the optimal value of expressed attitude on dimension one.  
    x=0:0.001:1;
    y11=utilitya(x,agealpha1(i),agebeta1(i),gamma);
    expalphain1=((1-mean(ingroupsamplechoice1))*(mean(ingroupsamplechoice1)^2))/(std(ingroupsamplechoice1)^2) -mean(ingroupsamplechoice1);
    expbetain1=((1-mean(ingroupsamplechoice1))/mean(ingroupsamplechoice1))*expalphain1;
    if expalphain1+expbetain1>=UL
        sumab1=expalphain1+expbetain1;
        expalphain1=UL*(expalphain1/sumab1);
        if expalphain1<1
            expalphain1=1.001;
        end
        expbetain1=UL*(expbetain1/sumab1);
        if expbetain1<1
            expbetain1=1.001;
        end
    end
     if expalphain1+expbetain1<=LL
        sumab1=expalphain1+expbetain1;
        expalphain1=LL*(expalphain1/sumab1);
        if expalphain1<1
            expalphain1=1.001;
        end
        expbetain1=LL*(expbetain1/sumab1);
        if expbetain1<1
            expbetain1=1.001;
        end
    end
    y21=utilitya(x,expalphain1,expbetain1,gamma);

    if isempty(outgroupsamplechoice1)~=1
    expalphaout1=((1-mean(outgroupsamplechoice1))*(mean(outgroupsamplechoice1)^2))/(std(outgroupsamplechoice1)^2) -mean(outgroupsamplechoice1);
    expbetaout1=((1-mean(outgroupsamplechoice1))/mean(outgroupsamplechoice1))*expalphaout1;
    if expalphaout1+expbetaout1>=UL
        sumab2=expalphaout1+expbetaout1;
        expalphaout1=UL*(expalphaout1/sumab2);
        if expalphaout1<1
            expalphaout1=1.001;
        end
        expbetaout1=UL*(expbetaout1/sumab2);
        if expbetaout1<1
            expbetaout1=1.001;
        end
    end
    if expalphaout1+expbetaout1<=LL
        sumab2=expalphaout1+expbetaout1;
        expalphaout1=LL*(expalphaout1/sumab2);
        if expalphaout1<1
            expalphaout1=1.001;
        end
        expbetaout1=LL*(expbetaout1/sumab2);
        if expbetaout1<1
            expbetaout1=1.001;
        end
    end
    y31=utilitya(x,expalphaout1,expbetaout1,gamma);
    else
        expalphaout1=0;
        expbetaout1=0;
        y31=0;
    end

    %Calculate the optimal value of expressed attitude on dimension two.
    y12=utilitya(x,agealpha2(i),agebeta2(i),gamma);
    expalphain2=((1-mean(ingroupsamplechoice2))*(mean(ingroupsamplechoice2)^2))/(std(ingroupsamplechoice2)^2) -mean(ingroupsamplechoice2);
    expbetain2=((1-mean(ingroupsamplechoice2))/mean(ingroupsamplechoice2))*expalphain2;
    if expalphain2+expbetain2>=UL
        sumab3=expalphain2+expbetain2;
        expalphain2=UL*(expalphain2/sumab3);
        if expalphain2<1
            expalphain2=1.001;
        end
        expbetain2=UL*(expbetain2/sumab3);
        if expbetain2<1
            expbetain2=1.001;
        end
    end
     if expalphain2+expbetain2<=LL
        sumab3=expalphain2+expbetain2;
        expalphain2=LL*(expalphain2/sumab3);
        if expalphain2<1
            expalphain2=1.001;
        end
        expbetain2=LL*(expbetain2/sumab3);
        if expbetain2<1
            expbetain2=1.001;
        end
    end
    y22=utilitya(x,expalphain2,expbetain2,gamma);

    if isempty(outgroupsamplechoice2)~=1
    expalphaout2=((1-mean(outgroupsamplechoice2))*(mean(outgroupsamplechoice2)^2))/(std(outgroupsamplechoice2)^2) -mean(outgroupsamplechoice2);
    expbetaout2=((1-mean(outgroupsamplechoice2))/mean(outgroupsamplechoice2))*expalphaout2;
    if expalphaout2+expbetaout2>=UL
        sumab4=expalphaout2+expbetaout2;
        expalphaout2=UL*(expalphaout2/sumab4);
        if expalphaout2<1
            expalphaout2=1.001;
        end
        expbetaout2=UL*(expbetaout2/sumab4);
        if expbetaout2<1
            expbetaout2=1.001;
        end
    end
     if expalphaout2+expbetaout2<=LL
        sumab4=expalphaout2+expbetaout2;
        expalphaout2=LL*(expalphaout2/sumab4);
        if expalphaout2<1
            expalphaout2=1.001;
        end
        expbetaout2=LL*(expbetaout2/sumab4);
        if expbetaout2<1
            expbetaout2=1.001;
        end
     end
     y32=utilitya(x,expalphaout2,expbetaout2,gamma);
    else
        expalphaout2=0;
        expbetaout2=0;
        y32=0;
    end

    y1=w1*y11+w2*y21-w3*y31;
    y2=w1*y12+w2*y22-w3*y32;
    [y1star,x1star]=min(y1);
    choice1(i)=x(x1star);
    allchoice1(i,round:R)=choice1(i);
    [y2star,x2star]=min(y2);
    
    %If the activated agent is a "stubborn agent", then the expressed
    %attitude on the second dimension will be replaced by a constant
    %value.
    if ismember(i,fixagent)==1
        choice2(i)=fixaut2;
    else
        choice2(i)=x(x2star);
    end
        
    allchoice2(i,round:R)=choice2(i);

end

%Remaining codes for data visualization. 
leftgroup11=find(choice1<=0.5);
leftgroup21=find(choice2<=0.5);
rightgroup11=find(choice1>0.5);
rightgroup21=find(choice2>0.5);

choice1l=choice1(leftgroup1);
choice1r=choice1(rightgroup1);
choice2l=choice2(leftgroup1);
choice2r=choice2(rightgroup1);

 figure(1)
        range1=[0:0.05:1];
        subplot(4,4,1:3)
        yy1l=hist(choice1l,range1);
        yy1r=hist(choice1r,range1);
        yy1all=(sum(yy1l)+sum(yy1r));
        yy1l=yy1l/yy1all;
        yy1r=yy1r/yy1all;
        b1=bar([yy1l' yy1r'],'stacked');
        ylim([0,0.5]);
        axis off
        subplot(4,4,[8,12,16])
        yy2l=hist(choice2l,range1);
        yy2r=hist(choice2r,range1);
        yy2all=(sum(yy2l)+sum(yy2r));
        yy2l=yy2l/yy2all;
        yy2r=yy2r/yy2all;
        b2=barh([yy2l' yy2r'],'stacked');
        xlim([0,0.5]);
        axis off
        subplot (4,4,[5:7,9:11,13:15])
        scatter(int1,int2,5,[0.5 0.5 0.5],'filled')
        for ar=1:1:N
            start_point=[int1(ar) int2(ar)]
            end_point=[choice1(ar) choice2(ar)]
            plot_arrow(start_point, end_point,0.002,[0.5 0.5 0.5])
        end
        hold on
        scatter(choice1(leftgroup1),choice2(leftgroup1),20,[0 0.4470 0.7410],'filled')
        scatter(choice1(rightgroup1),choice2(rightgroup1),20,[0.8500 0.3250 0.0980],'filled')
        xlabel('choice1')
        ylabel('choice2')
        grid on
        xlim([0,1])
        ylim([0,1])
        title(['round =' num2str(round)]);

choice1l=int1(leftgroup1);
choice1r=int1(rightgroup1);
choice2l=int2(leftgroup1);
choice2r=int2(rightgroup1);
figure(2)
        subplot(4,4,1:3)
        yy1l=hist(choice1l,range1);
        yy1r=hist(choice1r,range1);
        yy1all=(sum(yy1l)+sum(yy1r));
        yy1l=yy1l/yy1all;
        yy1r=yy1r/yy1all;
        b1=bar([yy1l' yy1r'],'stacked');
        ylim([0,0.5]);
        axis off
        subplot(4,4,[8,12,16])
        yy2l=hist(choice2l,range1);
        yy2r=hist(choice2r,range1);
        yy2all=(sum(yy2l)+sum(yy2r));
        yy2l=yy2l/yy2all;
        yy2r=yy2r/yy2all;
        b2=barh([yy2l' yy2r'],'stacked');
        xlim([0,0.5]);
        axis off
        subplot (4,4,[5:7,9:11,13:15])
        scatter(int1,int2,5,[0.5 0.5 0.5],'filled')
        xlabel('choice1')
        ylabel('choice2')
        grid on
        xlim([0,1])
        ylim([0,1])
        title(['round =' num2str(0)]); 
