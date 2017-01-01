function MCR=CalMCR(MID_list_true,MID_list_inferred,new_order)
chain_length=size(MID_list_inferred,2);
MCR=zeros(chain_length,1);
true_with_out_nan = MID_list_true(~isnan(MID_list_true));

for i=1:chain_length
    I1=MID_list_inferred(:,i)==1;
    I2=MID_list_inferred(:,i)==2;
    I3=MID_list_inferred(:,i)==3;
    MID_list_inferred(I1,i)=new_order(1);
    MID_list_inferred(I2,i)=new_order(2);
    MID_list_inferred(I3,i)=new_order(3);
    temp = MID_list_inferred(:,i);
    I = true_with_out_nan ~= temp(~isnan(temp));
    MCR(i)=sum(I)/length(I);
end