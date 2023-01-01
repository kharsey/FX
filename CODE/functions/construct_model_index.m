function [ model_index ] = construct_model_index(gamma,number_g,idx_step,model_index)

for i=1:number_g
    
   block_length=size(model_index,1)/prod(idx_step(1:i));
   number_blocks=size(model_index,1)/block_length;
   z=[];
   for ij=1:length(gamma{i})
       pp=gamma{i};
       z=[z; repmat(pp(ij),block_length,1)];
   end
   model_index(:,i)=repmat(z,number_blocks/idx_step(i),1);
end
   
end

