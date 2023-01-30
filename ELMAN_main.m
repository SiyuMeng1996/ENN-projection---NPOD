
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  This code are used for predicting NPOD seasonality  %%%%%%
%%%%%%  Input data SST, SSH, SR, WS, PRE, CV                %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% ���ж��ٴ�
time=300;
MPEE=[];
INTEN_2=[];
fifi=nan*ones(300,95*12);
TYYYY='AREA'; %Ԥ�����


tititi = 206;

KOKO=nan*ones(tim,95);
KOKOll=nan*ones(tim,95);
KOKOhh=nan*ones(tim,95);

for timmm=1:time
     
%% ѡ����������
Mode = 'desert';
XX   =  8*23 ;
YY   =  8*45 ;

jinanin=ones(tititi+95*12,9);
jinanin(1:tititi,4)=T_OD_WS(1:tititi)';
jinanin(1:tititi,5)=T_OD_PRE(1:tititi)';
jinanin(1:tititi,6)=T_OD_SST(1:tititi)';
jinanin(1:tititi,7)=T_OD_SSH(1:tititi)';
jinanin(1:tititi,8)=T_OD_SR(1:tititi)';
jinanin(1:tititi,9)=T_OD_CS(1:tititi)';

eval(['jinanin(tititi+1:end,4)=','TI_',PERI,'_WS''']);
eval(['jinanin(tititi+1:end,5)=','TI_',PERI,'_PRE''']);
eval(['jinanin(tititi+1:end,6)=','TI_',PERI,'_SST''']);
eval(['jinanin(tititi+1:end,7)=','TI_',PERI,'_SSH''']);
eval(['jinanin(tititi+1:end,8)=','TI_',PERI,'_SR''']);
eval(['jinanin(tititi+1:end,9)=','TI_',PERI,'_CV''']);

if strcmp(TYYYY,'AREA') %Ԥ���������ǿ��
jinanout=nanmean(AREA_M)*ones(tititi+95*12,3);
jinanout(1:tititi,3)=AREA_M(1:tititi);
else
jinanout=nanmean(INTEN_M)*ones(tititi+95*12,3);
jinanout(1:tititi,3)=INTEN_M(1:tititi)';
end
%% �����������
save in jinanin
save out jinanout

%% ��������
clc
close all
load in.mat jinanin
load out.mat jinanout
jinans=[jinanin(:,4:8),jinanout(:,3)];

%% Elman����������
samp=jinans;
[m,n]=size(samp);
[all,mu,std]=zscore(samp);%��һ������ֵ���
samp=all;

ycn=1;%�������?
xcn=size(samp,2)-ycn;%��������
test_cn=95*12;%�˴��޸���֤��ݵ�����?
[sampin,sampout,confirm_in,confirm_out] = ceate_train_test_data(samp,ycn,test_cn,4);

nod1num= 28;%��һ�������ڵ���
nod2num=5;%�ڶ��������ڵ���
outnum=ycn;   %�����ڵ���
TF1='tansig'; %sig ������Ϊ������Ĵ��ݺ���?
TF2='tansig'; %sig ������Ϊ������Ĵ��ݺ���?
TFout='purelin';%purelin��������Ժ�����Ϊ�����Ĵ��ݺ���?
BTF='traingdx'; %traingdx  
[ t ]=threshold( xcn );%�����ÿ���еķ��?
net=newelm(t,[nod1num,nod2num,outnum],{TF1,TF2,TFout},BTF);

net=init(net);%��ʼ��
net.trainParam.epochs=2000;%ѵ������
net.trainParam.goal =1.0e-5; %��Сȫ�����?
net.trainParam.max_fail=20;%�����֤ʧ�ܴ���?
sampin=sampin' ;%ת�þ���
sampout=sampout';%ת�þ���
[net,tr,Y,E,Pf,Af]=train(net,sampin,sampout);%ѵ������
confirm_in=confirm_in'; %����ת��
confirm_out=confirm_out'; %����ת��
simout=sim(net,confirm_in);% ������֤���?
 
%����һ��
simout=simout'; %ת��
confirm_out=confirm_out';%ת��
simout      = FanGuiYiHua(simout,mu(xcn+1:xcn+ycn),std(xcn+1:xcn+ycn));
confirm_out = FanGuiYiHua(confirm_out,mu(xcn+1:xcn+ycn),std(xcn+1:xcn+ycn)) ; 
%��ʾ�ԱȽ��?     
result_duibi(confirm_out,simout); 
disp('ELMAN��MPE  RMSE')
MPE= sum(abs(simout-confirm_out)./confirm_out)/length(simout)*100 %����������
RMSE=sqrt(sum((simout-confirm_out).^2 ) /length(simout) )  %������
[R,stats] = shandiantu( confirm_out,simout );
R 
R2=stats(1) % ��ʾR2
MPEE=[MPEE,MPE];

kk_1=simout';
fifi(timmm,:)=kk_1;%��ģ��������ݴ��浽fifi�ļ�

Cx_11_m=[1:95];%���
Cx_22_m=[1:95];%���ֵ
Cx_33_m=[1:95];%��Сֵ

for year = 1:95
XXX=[1:1:12];
YYY=kk_1(1+12*(year-1):12*year);
Cx_11_m(year)=max(YYY)-min(YYY);
Cx_22_m(year)=max(YYY);
Cx_33_m(year)=min(YYY);
end

KOKO(timmm,:)=Cx_11_m;
KOKOll(timmm,:)=Cx_22_m;
KOKOhh(timmm,:)=Cx_33_m;


% ����Ϊ��ͼ����

figure
set(gcf,'unit','pixels','position',[0,10,1000,300])
hold on

plot([1:1140], kk_1,'-.^','color','k','linewidth',1,'markersize',2)
set(gca,'ycolor','k');

grid on
box on
set(gca,'xlim',[0 1141],'xtick',[-12:120:1140])
set(gca,'XTickLabel',{'2006','2015','2025','2035','2045','2055','2065','2075','2085','2095','2100'})
xtickangle(40)
set(gca,'GridLineStyle',':','LineWidth',2,'GridColor','k')
saveas(gcf,['PLOT_RCP85_',num2str(timmm),'.jpg'], 'jpg')

end

eval(['FINA_data_',TYYYY,'=fifi;']);
eval(['FINA_mean_',TYYYY,'=squeeze(nanmean(fifi));']);

